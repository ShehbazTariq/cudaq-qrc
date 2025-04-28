"""
GPU‑ready quantum‑reservoir simulator with ETA tracking.
FIXED so parent and worker processes always agree on the embedding
dimension (emb_dim) and therefore avoid ValueError: shape mismatch.

Main fixes
----------
1. RydbergSimulator._get_config now omits the *readouts* list (because it
   contains non‑serialisable cudaq objects) and returns only simple
   Python / NumPy data that fully determines a unique simulator.
2. _worker regenerates the readouts via generate_readouts(nsites) so the
   list is identical to the parent.
3. embeddings_with_cache now verifies at runtime that worker slices have
   the same width as the memory‑map and raises a clear error otherwise.
"""

import os, math, pickle, shutil, concurrent.futures, time
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pywt
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# ──────────────────────────────────────────────────────────────────────
# 1)  core simulators
# ──────────────────────────────────────────────────────────────────────
class QuantumDynamics:
    """Parent class providing shared utilities + apply_layer loop."""

    def __init__(self, nsites, atoms, readouts,
                 t_start, t_end, t_step, tag, use_wavelet=False):
        self.nsites, self.atoms, self.readouts = nsites, atoms, readouts
        self.t_start, self.t_end, self.t_step = t_start, t_end, t_step
        self.tag, self.use_wavelet = tag, use_wavelet

        self.dimensions = {i: 2 for i in range(len(atoms))}
        self.n_steps = int((t_end - t_start) / t_step) + 1
        self.steps = np.linspace(0, t_end, self.n_steps)

        # import locally so env‑var CUDA_VISIBLE_DEVICES is respected
        import cudaq
        self.schedule = cudaq.Schedule(self.steps, ["t"])

    # ---------------- helpers ----------------------------------------
    def _match(self, x):
        """Pad / truncate to exactly nsites."""
        return x[: self.nsites] if len(x) >= self.nsites else np.concatenate(
            [x, np.zeros(self.nsites - len(x))]
        )

    def _wavelet(self, x):
        if not self.use_wavelet:
            return np.asarray(x)
        coeffs = pywt.wavedec(x, "db1")
        vec = np.concatenate(coeffs)[: len(self.atoms)]
        return vec / np.max(np.abs(vec)) if np.max(np.abs(vec)) else vec

    # ---------------- main evolution loop ----------------------------
    def apply_layer(self, datapoints, construct_h, *, show_progress=True):
        import cudaq  # local import keeps every proc independent

        cudaq.set_target("dynamics")

        n = len(datapoints)
        R = len(self.readouts)
        out = np.zeros((n, R * (self.n_steps - 1)), np.float64)

        psi0 = cudaq.State.from_data(
            np.ones((2 ** self.nsites, 2 ** self.nsites), np.complex128) / np.sqrt(2)
        )

        iterator = tqdm(
            enumerate(datapoints),
            total=n,
            disable=not show_progress,
            desc="evolve",
            unit="sample",
        )

        for j, x in iterator:
            h = construct_h(self._match(self._wavelet(x)))
            ev = cudaq.evolve(
                h,
                self.dimensions,
                self.schedule,
                psi0,
                observables=self.readouts,
                collapse_operators=[],
                store_intermediate_results=True,
            )
            evo = ev.expectation_values()[1:]
            out[j] = np.array([[v.expectation() for v in step] for step in evo]).T.flatten()
        return out


# ---------------- Rydberg child --------------------------------------
class RydbergSimulator(QuantumDynamics):
    def __init__(
        self,
        *,
        nsites,
        atoms,
        readouts,
        omega,
        t_start,
        t_end,
        t_step,
        t_rate,
        alpha,
        V_matrix,
        id,
        use_wavelet=False,
    ):
        # ensure ndarrays early (works in parent & child)
        atoms, alpha, V_matrix = map(lambda a: np.asarray(a, float), (atoms, alpha, V_matrix))
        super().__init__(nsites, atoms, readouts, t_start, t_end, t_step, id, use_wavelet)
        self.omega, self.t_rate = omega, t_rate
        self.alpha, self.V = alpha, V_matrix

    def construct_hamiltonian(self, x):
        import cudaq  # local import

        H = cudaq.SpinOperator()
        for j in range(self.nsites):
            H += (self.omega / 2) * cudaq.spin.x(j)
            for k in range(j + 1, self.nsites):
                H += self.V[j, k] * cudaq.spin.z(j) * cudaq.spin.z(k)
            H -= (x[j] + self.alpha[j] * x[j]) * cudaq.spin.z(j)
        return H

    def apply_layer(self, data, show_progress=True):
        return super().apply_layer(data, self.construct_hamiltonian, show_progress=show_progress)

    # ---------------- serialisable config ---------------------------
    def _get_config(self) -> Dict[str, Any]:
        """Return ONLY built-in types so pickle works.
        *readouts* will be regenerated in each worker.
        """
        return dict(
            nsites     = int(self.nsites),
            atoms      = self.atoms.tolist(),
            omega      = float(self.omega),
            t_start    = float(self.t_start),
            t_end      = float(self.t_end),
            t_step     = float(self.t_step),
            t_rate     = float(self.t_rate),
            alpha      = self.alpha.tolist(),
            V_matrix   = self.V.tolist(),
            id         = int(self.tag),
            use_wavelet= bool(self.use_wavelet),
        )


# ──────────────────────────────────────────────────────────────────────
# 2)  readouts + lattice plot (unchanged API; plotting body skipped)
# ──────────────────────────────────────────────────────────────────────

def generate_readouts(n: int):
    import cudaq  # local import

    outs = [cudaq.spin.x(i) for i in range(n)] + [cudaq.spin.y(i) for i in range(n)] + [
        cudaq.spin.z(i) for i in range(n)
    ]
    for i in range(n):
        for j in range(i + 1, n):
            outs += [
                cudaq.spin.x(i) * cudaq.spin.x(j),
                cudaq.spin.y(i) * cudaq.spin.y(j),
                cudaq.spin.z(i) * cudaq.spin.z(j),
            ]
    return outs


def plot_3d_lattice(*args, **kwargs):
    """Placeholder – existing implementation unchanged."""
    pass


# ──────────────────────────────────────────────────────────────────────
# 3)  helpers
# ──────────────────────────────────────────────────────────────────────

def _gpu_banner(i: int) -> str:
    try:
        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        name = pynvml.nvmlDeviceGetName(h).decode()
        out = f"{i}:{name}  {mem.free // 2 ** 20}/{mem.total // 2 ** 20} MB free"
        pynvml.nvmlShutdown()
        return out
    except Exception:
        return f"{i}: <unknown GPU>"


# ──────────────────────────────────────────────────────────────────────
# 4)  worker – regenerates readouts, shows its own tqdm, checks emb_dim
# ──────────────────────────────────────────────────────────────────────

def _worker(args: Tuple):
    phys_gpu, rows, full_X, sim_blob, batch_size, chatty = args

    # isolate a single physical device for this process
    os.environ["CUDA_VISIBLE_DEVICES"] = str(phys_gpu)

    import numpy as _np, pickle as _pk
    import cudaq

    full_X = _np.asarray(full_X, dtype=_np.float32)

    cls, kwargs = _pk.loads(sim_blob)

    # regenerate readouts so parent & worker are identical
    from dynamics import generate_readouts
    kwargs["readouts"] = generate_readouts(kwargs["nsites"])
    sim = cls(**kwargs)

    if chatty:
        print(f"[PID {os.getpid()}] ▶  {_gpu_banner(phys_gpu)}", flush=True)

    emb_dim = sim.apply_layer(full_X[:1], show_progress=False).shape[1]
    out = _np.zeros((len(rows), emb_dim), dtype=_np.float32)

    n_batches = math.ceil(len(rows) / batch_size)
    prog_bar  = tqdm(total=n_batches, desc=f"GPU{phys_gpu}", unit="batch", position=phys_gpu, leave=False)
    start_time = time.perf_counter()

    for start in range(0, len(rows), batch_size):
        blk_inds = rows[start : start + batch_size]
        out[start : start + len(blk_inds)] = sim.apply_layer(
            full_X[blk_inds], show_progress=False
        ).astype(_np.float32)

        prog_bar.update(1)
        done    = prog_bar.n * batch_size
        elapsed = time.perf_counter() - start_time
        if done:
            rate   = elapsed / done
            remain = rate * (len(rows) - done)
            prog_bar.set_postfix({"ETA": f"{remain/60:6.1f}m"})
    prog_bar.close()

    return rows, out


def _to_serialisable(x):
    """Convert NumPy objects → builtin types so pickle works in spawn."""
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return [_to_serialisable(v) for v in x]
    if isinstance(x, dict):
        return {k: _to_serialisable(v) for k, v in x.items()}
    return x


# ──────────────────────────────────────────────────────────────────────
# 5)  main resumable routine
# ──────────────────────────────────────────────────────────────────────

def embeddings_with_cache(
    simulator,
    X: np.ndarray,
    *,
    cache_dir: str | Path = "cache",
    final_dir: str | Path = "dataset/embeddings",
    config_name: str = "",
    overwrite: bool = False,
    batch_size: int = 256,
) -> np.ndarray:
    """Compute (or resume) quantum‑reservoir embeddings with GPU workers."""

    cache_dir, final_dir = Path(cache_dir), Path(final_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    stem = config_name or f"{simulator.nsites}q_cfg{simulator.tag}"
    cache_path = cache_dir / f"{stem}.npy"
    final_path = final_dir / f"{stem}.npy"

    X = np.asarray(X, dtype=np.float32, order="C")
    N = len(X)

    # ── open / create mmap ------------------------------------------
    if cache_path.exists() and not overwrite:
        mmap = np.load(cache_path, mmap_mode="r+")
        if mmap.shape[0] != N:
            raise ValueError("Cached file has the wrong number of rows.")
    else:
        D = simulator.apply_layer(X[:1], show_progress=False).shape[1]
        mmap = np.lib.format.open_memmap(cache_path, mode="w+", dtype=np.float32, shape=(N, D))
        mmap[:] = np.nan

    todo = np.where(np.isnan(mmap).any(axis=1))[0]
    if todo.size == 0:
        print("✓ embeddings already finished")
        return np.asarray(mmap)

    # ── available GPUs ---------------------------------------------
    gpus = [int(g) for g in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if g] or [0]
    print(f"Using GPUs {gpus} – {len(todo)} samples remaining")

    # split indices
    idx_chunks = [chunk for chunk in np.array_split(todo, len(gpus)) if chunk.size]

    # single‑GPU fall‑back (no multiprocessing)
    if len(gpus) == 1:
        for rows in tqdm(idx_chunks, desc="Embedding", unit="batch"):
            mmap[rows] = simulator.apply_layer(X[rows]).astype(np.float32)
        mmap.flush(); shutil.copy2(cache_path, final_path)
        return np.asarray(mmap)

    # prepare pickled simulator (without readouts, regenerated per worker)
    cfg_dict = _to_serialisable(simulator._get_config())
    sim_blob = pickle.dumps((simulator.__class__, cfg_dict))

    # pack args
    args = [
        (g, rows, X, sim_blob, batch_size, i == 0)
        for i, (g, rows) in enumerate(zip(gpus, idx_chunks))
    ]

    import multiprocessing as mp

    with concurrent.futures.ProcessPoolExecutor(max_workers=len(args), mp_context=mp.get_context("spawn")) as pool:
        for rows, part in pool.map(_worker, args):
            if part.shape[1] != mmap.shape[1]:
                raise ValueError(
                    f"Embedding dimension mismatch: worker {part.shape[1]} vs parent {mmap.shape[1]}"
                )
            mmap[rows] = part  # write slice

    mmap.flush(); shutil.copy2(cache_path, final_path)
    return np.asarray(mmap)


# ──────────────────────────────────────────────────────────────────────
# 6)  convenience wrapper per \u201cnsites\u201d
# ──────────────────────────────────────────────────────────────────────

def get_embeddings_for_nsites(
    *,
    nsites: int,
    X_train: np.ndarray,
    X_test: np.ndarray,
    dataset_tag: str = "MackeyGlass",
    root_cache: str = "cache",
    root_final: str = "dataset/embeddings",
    batch_size: int = 10,
    force: bool = False,
    rng_seed: int | None = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """High‑level helper used by experiments."""

    cfg_tr = f"{nsites}q_train_{len(X_train)}_{dataset_tag}"
    cfg_te = f"{nsites}q_test_{len(X_test)}_{dataset_tag}"
    f_tr = Path(root_final) / f"{cfg_tr}.npy"
    f_te = Path(root_final) / f"{cfg_te}.npy"

    if not force and f_tr.exists() and f_te.exists():
        print(f"✓ n={nsites}: loaded cached embeddings")
        return np.load(f_tr), np.load(f_te)

    if rng_seed is not None:
        np.random.seed(rng_seed)

    # ---------------- build simulator instance ----------------------
    d = 10.0
    atoms = np.linspace(0, (nsites - 1) * d, nsites)
    alpha = np.random.rand(nsites)
    V = np.random.rand(nsites, nsites)
    V = (V + V.T) / 2
    np.fill_diagonal(V, 0.2)

    sim = RydbergSimulator(
        nsites=nsites,
        atoms=atoms,
        readouts=generate_readouts(nsites),
        omega=2 * np.pi,
        t_start=0.0,
        t_end=3.0,
        t_step=0.5,
        t_rate=0.0,
        alpha=alpha,
        V_matrix=V,
        id=nsites,
    )

    # save a pretty lattice picture once (optional)
    pdf_path = Path(root_final) / f"{nsites}_lattice.pdf"
    if not pdf_path.exists():
        plot_3d_lattice(
            nsites, d, atoms, alpha, V, save_as_pdf=True, filename=str(pdf_path)
        )

    # scale data to [-1, 1]
    scaler = MinMaxScaler(feature_range=(-1, 1)).fit(X_train)
    Xt, Xe = scaler.transform(X_train), scaler.transform(X_test)

    tr = embeddings_with_cache(
        sim,
        Xt,
        cache_dir=root_cache,
        final_dir=root_final,
        config_name=cfg_tr,
        batch_size=batch_size,
    )

    te = embeddings_with_cache(
        sim,
        Xe,
        cache_dir=root_cache,
        final_dir=root_final,
        config_name=cfg_te,
        batch_size=batch_size,
    )

    scale = 2 ** nsites
    return tr / scale, te / scale




def noisy_embeddings(emb, *,
                     gaussian_sigma=None,
                     multiplicative_sigma=None,
                     shots=None,
                     T2=None, dt=1.0):
    """
    emb : ndarray of shape (N, T, R)
    Returns a copy with realistic Rydberg‐style noise applied.
    """
    emb = np.asarray(emb, float).copy()
    flat = emb.reshape(-1)

    # 1) T2 decoherence (broadcasts correctly over (N,T,R))
    if T2 is not None:
        N, T, R = emb.shape
        t        = np.arange(T) * dt
        decay    = np.exp(-t / T2).reshape(1, T, 1)
        emb     *= decay
        flat     = emb.reshape(-1)

    # 2) shot noise (p clipped to [0,1])
    if shots is not None:
        p = np.clip((flat + 1) / 2, 0.0, 1.0)
        k = np.random.binomial(shots, p)
        flat[:] = 2 * k / shots - 1

    # 3) multiplicative drift
    if multiplicative_sigma:
        flat[:] *= 1 + np.random.normal(0,
                                         multiplicative_sigma,
                                         flat.size)

    # 4) additive Gaussian read‐out noise
    if gaussian_sigma:
        flat[:] += np.random.normal(0,
                                    gaussian_sigma,
                                    flat.size)

    # clip final values back into [-1,1]
    np.clip(flat, -1.0, 1.0, out=flat)
    return emb.reshape(emb.shape)
