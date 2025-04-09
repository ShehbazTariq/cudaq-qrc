import numpy as np
from sklearn.metrics import mean_squared_error
import random
from joblib import Parallel, delayed

# Global cache: (mask_string -> mse)
EVALUATION_CACHE = {}

def enforce_max_features(mask: np.ndarray, max_features_selected: int) -> None:
    """
    Ensures that `mask` has no more than `max_features_selected` bits = 1.
    If the mask has more, randomly turn off bits until it meets the requirement.
    """
    num_features_on = np.sum(mask)
    if num_features_on > max_features_selected:
        on_indices = np.where(mask == 1)[0]
        np.random.shuffle(on_indices)
        excess = num_features_on - max_features_selected
        mask[on_indices[:excess]] = 0

def reshape_data(X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Reshape the data based on the mask for both 2D and 3D cases.
    """
    if X.ndim == 2:
        # Directly subset features
        return X[:, mask.astype(bool)]
    elif X.ndim == 3:
        # Subset observables and flatten time_steps * selected_observables
        selected_observables = np.where(mask == 1)[0]
        if len(selected_observables) == 0:
            raise ValueError("No observables selected. Mask must have at least one active observable.")
        X_selected = X[:, :, selected_observables]
        return X_selected.reshape(X.shape[0], -1)
    else:
        raise ValueError("Unsupported data dimensionality. Only 2D and 3D are supported.")

def evaluate_subset(
    mask: np.ndarray,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    model
) -> float:
    """
    Evaluate MSE of selected features (binary 'mask') using a given regression model.
    Results are cached to avoid re-training the same subset.
    """
    global EVALUATION_CACHE
    
    mask_key = ''.join(str(bit) for bit in mask)
    if mask_key in EVALUATION_CACHE:
        return EVALUATION_CACHE[mask_key]
    
    if not np.any(mask):
        EVALUATION_CACHE[mask_key] = 1e10
        return 1e10
    
    # Apply mask to reshape data
    X_sub_train = reshape_data(X_train, mask)
    X_sub_test = reshape_data(X_test, mask)
    
    model.fit(X_sub_train, y_train)
    preds = model.predict(X_sub_test)
    mse_val = mean_squared_error(y_test, preds)
    
    EVALUATION_CACHE[mask_key] = mse_val
    return mse_val

def genetic_algorithm_feature_selection(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    base_mse: float,
    qrc_full_mse: float,
    model,
    pop_size: int = 20,
    n_gen: int = 20,
    crossover_rate: float = 0.7,
    mutation_rate: float = 0.1,
    elitism_count: int = 2,
    n_jobs: int = -1,
    max_features_selected: int = None
) -> list:
    """
    Genetic Algorithm for feature subset selection supporting a custom model.
    """
    global EVALUATION_CACHE
    EVALUATION_CACHE.clear()
    
    if X_train.ndim == 2:
        n_features = X_train.shape[1]
    elif X_train.ndim == 3:
        n_features = X_train.shape[2]  # Features are observables
    else:
        raise ValueError("Unsupported data dimensionality. Only 2D and 3D are supported.")
    
    population = []
    for _ in range(pop_size):
        mask = np.random.randint(0, 2, size=n_features, dtype=int)
        if max_features_selected is not None:
            enforce_max_features(mask, max_features_selected)
        population.append((mask, None))
    
    best_mask = None
    best_mse = float('inf')
    iteration_results = []
    
    def compute_fitness(ind: np.ndarray) -> float:
        return -evaluate_subset(ind, X_train, X_test, y_train, y_test, model)
    
    for gen in range(n_gen):
        masks = [p[0] for p in population]
        fitness_values = Parallel(n_jobs=n_jobs)(
            delayed(compute_fitness)(m) for m in masks
        )
        population = [(masks[i], fitness_values[i]) for i in range(pop_size)]
        population.sort(key=lambda x: x[1], reverse=True)
        
        gen_best_mask, gen_best_fitness = population[0]
        gen_best_mse = -gen_best_fitness
        if gen_best_mse < best_mse:
            best_mse = gen_best_mse
            best_mask = gen_best_mask.copy()
        
        print(
            f"Iteration {gen+1}/{n_gen}: "
            f"base_mse={base_mse:.8f}, "
            f"qrc_full_mse={qrc_full_mse:.8f}, "
            f"best_mse={best_mse:.8f}"
        )
        iteration_results.append({
            "iteration": gen+1,
            "best_mask": best_mask.copy(),
            "best_mse": best_mse
        })
        
        new_population = population[:elitism_count]
        while len(new_population) < pop_size:
            cand1 = random.sample(population, 2)
            parent1 = max(cand1, key=lambda x: x[1])[0]
            cand2 = random.sample(population, 2)
            parent2 = max(cand2, key=lambda x: x[1])[0]
            
            child1 = parent1.copy()
            child2 = parent2.copy()
            if random.random() < crossover_rate:
                for i in range(n_features):
                    if random.random() < 0.5:
                        child1[i], child2[i] = child2[i], child1[i]
            
            for i in range(n_features):
                if random.random() < mutation_rate:
                    child1[i] = 1 - child1[i]
                if random.random() < mutation_rate:
                    child2[i] = 1 - child2[i]
            
            if max_features_selected is not None:
                enforce_max_features(child1, max_features_selected)
                enforce_max_features(child2, max_features_selected)
            
            new_population.append((child1, None))
            if len(new_population) < pop_size:
                new_population.append((child2, None))
        
        population = new_population
    
    return iteration_results