import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interp1d


# ------------------------------- RSM ------------------------------
def continuous_reference_set_method(alternatives, criteria_types, a, b, n_extra=10):
    rng = np.random.default_rng()
    M = alternatives.shape[1]
    extra_points = np.zeros((n_extra, M))
    for i in range(M):
        low = min(a[i], b[i])
        high = max(a[i], b[i])
        extra_points[:, i] = rng.uniform(low, high, n_extra)
    all_points = np.vstack([alternatives, extra_points])
    def normalize(points, a, b, criteria_types):
        norm_alt = np.zeros_like(points, dtype=float)
        for i in range(len(criteria_types)):
            if criteria_types[i] == -1:  # Minimalizacja
                norm_alt[:, i] = (points[:, i] - a[i]) / (b[i] - a[i])
            else:  # Maksymalizacja
                norm_alt[:, i] = (b[i] - points[:, i]) / (b[i] - a[i])
        return norm_alt
    def scoring_function(points, a, b, criteria_types):
        norm_alt = normalize(points, a, b, criteria_types)
        distance_to_a = np.linalg.norm(norm_alt - np.zeros((len(points), len(a))), axis=1)
        distance_to_b = np.linalg.norm(norm_alt - np.ones((len(points), len(a))), axis=1)
        C = distance_to_b / (distance_to_a + distance_to_b)
        return C
    scores = scoring_function(all_points, a, b, criteria_types)
    ranking = np.argsort(-scores)  # sortujemy malejąco po scores
    return ranking, scores, all_points


def discrete_reference_set_method(alternatives, criteria_types, a, b):
    def normalize(alternatives, a, b, criteria_types):
        norm_alt = np.zeros_like(alternatives, dtype=float)
        for i in range(len(criteria_types)):
            if criteria_types[i] == -1:
                norm_alt[:, i] = (alternatives[:, i] - a[i]) / (b[i] - a[i])
            else:
                norm_alt[:, i] = (b[i] - alternatives[:, i]) / (b[i] - a[i])
        return norm_alt
    def scoring_function(alternatives, a, b, criteria_types):
        norm_alt = normalize(alternatives, a, b, criteria_types)
        distance_to_a = np.linalg.norm(norm_alt - np.zeros((len(alternatives), len(a))), axis=1)
        distance_to_b = np.linalg.norm(norm_alt - np.ones((len(alternatives), len(a))), axis=1)
        C = distance_to_b / (distance_to_a + distance_to_b)
        return C
    scores = scoring_function(alternatives, a, b, criteria_types)
    ranking = np.argsort(-scores)
    return ranking, scores


# ------------------------------- TOPSIS ------------------------------
def fuzzy_topsis(decision_matrix, weights, criteria_types):
    norm_matrix = np.zeros_like(decision_matrix, dtype=float)
    for j in range(decision_matrix.shape[1]):
        if criteria_types[j] == 1:
            max_value = np.max(decision_matrix[:, j, 2])
            norm_matrix[:, j, 0] = decision_matrix[:, j, 0] / max_value
            norm_matrix[:, j, 1] = decision_matrix[:, j, 1] / max_value
            norm_matrix[:, j, 2] = decision_matrix[:, j, 2] / max_value
        else:
            min_value = np.min(decision_matrix[:, j, 0])
            norm_matrix[:, j, 0] = min_value / decision_matrix[:, j, 2]
            norm_matrix[:, j, 1] = min_value / decision_matrix[:, j, 1]
            norm_matrix[:, j, 2] = min_value / decision_matrix[:, j, 0]
    weighted_matrix = np.zeros_like(norm_matrix, dtype=float)
    for j in range(norm_matrix.shape[1]):
        weighted_matrix[:, j, 0] = norm_matrix[:, j, 0] * weights[j]
        weighted_matrix[:, j, 1] = norm_matrix[:, j, 1] * weights[j]
        weighted_matrix[:, j, 2] = norm_matrix[:, j, 2] * weights[j]
    fpis = np.zeros((decision_matrix.shape[1], 3), dtype=float)
    fnis = np.zeros((decision_matrix.shape[1], 3), dtype=float)
    for j in range(weighted_matrix.shape[1]):
        if criteria_types[j] == 1:
            fpis[j] = np.max(weighted_matrix[:, j, :], axis=0)
            fnis[j] = np.min(weighted_matrix[:, j, :], axis=0)
        else:
            fpis[j] = np.min(weighted_matrix[:, j, :], axis=0)
            fnis[j] = np.max(weighted_matrix[:, j, :], axis=0)
    distance_to_fpis = np.sqrt(np.sum(np.power(weighted_matrix - fpis, 2), axis=(1, 2)))
    distance_to_fnis = np.sqrt(np.sum(np.power(weighted_matrix - fnis, 2), axis=(1, 2)))
    with np.errstate(divide='ignore', invalid='ignore'):
        similarity_to_ideal = distance_to_fnis / (distance_to_fpis + distance_to_fnis)
        similarity_to_ideal[np.isnan(similarity_to_ideal)] = 0
    ranking = np.argsort(similarity_to_ideal)[::-1]
    return similarity_to_ideal, ranking


def topsis(decision_matrix, weights, criteria_types):
    norm_matrix = decision_matrix / np.sqrt((decision_matrix ** 2).sum(axis=0))
    weighted_matrix = norm_matrix * weights
    ideal_best = np.where(criteria_types == 1, np.max(weighted_matrix, axis=0), np.min(weighted_matrix, axis=0))
    ideal_worst = np.where(criteria_types == 1, np.min(weighted_matrix, axis=0), np.max(weighted_matrix, axis=0))
    dist_to_ideal = np.sqrt(((weighted_matrix - ideal_best) ** 2).sum(axis=1))
    dist_to_worst = np.sqrt(((weighted_matrix - ideal_worst) ** 2).sum(axis=1))
    score = dist_to_worst / (dist_to_ideal + dist_to_worst)
    ranking = np.argsort(score)[::-1]
    return ranking, score


#------------------------------------- UTA ------------------------------------ 

def UTA_function_continuous(values, reference_points, lambda_param=1, use_polynomial=False):
    """
    Aproksymuje funkcje użyteczności dla wersji ciągłej.

    :param values: Lista wartości alternatyw w przestrzeni kryteriów (ciągłe wartości)
    :param reference_points: Lista punktów referencyjnych (najlepszy, najgorszy, neutralny)
    :param lambda_param: Współczynnik λ (preferencje decydenta)
    :param use_polynomial: Flaga, czy używać interpolacji wielomianowej zamiast liniowej
    :return: Lista funkcji użyteczności dla wszystkich alternatyw
    """
    utility_functions = []

    # Budowa funkcji użyteczności dla każdego kryterium
    for i in range(len(reference_points[0])):
        ref_values = [ref[i] for ref in reference_points]  # Punkty referencyjne dla kryterium i
        utility_values = [-1, 0, 1]  # Wartości użyteczności odpowiadające punktom referencyjnym

        if use_polynomial:
            # Interpolacja wielomianowa za pomocą np.polyfit
            coeffs = np.polyfit(ref_values, utility_values, deg=min(len(ref_values)-1, 2))
            def poly_func(x, coeffs=coeffs):
                return np.polyval(coeffs, x)
            utility_functions.append(poly_func)
        else:
            # Tworzenie funkcji aproksymującej (ciągła interpolacja liniowa)
            interp_func = interp1d(ref_values, utility_values, kind='linear', fill_value="extrapolate")
            utility_functions.append(interp_func)

    # Wyznaczanie wartości funkcji użyteczności dla alternatyw
    utility_values = []
    for val in values:
        alternative_utility = []
        for i in range(len(val)):
            utility_value = utility_functions[i](val[i]) * lambda_param  # Obliczanie użyteczności dla kryterium i
            alternative_utility.append(utility_value)
        utility_values.append(alternative_utility)

    return np.array(utility_values)

# Funkcja optymalizacji w wersji ciągłej
def UTAstar_continuous(values, reference_points, lambda_param=1, use_polynomial=False):
    """
    Metoda UTA Star dla wersji ciągłej (wielokryterialnej).

    :param values: Lista wartości alternatyw w przestrzeni kryteriów
    :param reference_points: Lista punktów referencyjnych dla każdego kryterium
    :param lambda_param: Współczynnik λ (preferencje decydenta)
    :param use_polynomial: Flaga, czy używać interpolacji wielomianowej zamiast liniowej
    :return: Najlepsza alternatywa oraz jej użyteczność
    """
    # Wyznaczanie bounds na podstawie wartości alternatyw
    bounds = [(np.min(values[:, i]), np.max(values[:, i])) for i in range(values.shape[1])]

    # Funkcja celu: Maksymalizacja sumy użyteczności
    def objective_function(x):
        utility_values = UTA_function_continuous([x], reference_points, lambda_param=lambda_param, use_polynomial=use_polynomial)
        return -np.sum(utility_values)  # Negujemy, ponieważ minimalizujemy w optymalizacji

    # Początkowy punkt startowy (środek przedziałów)
    initial_guess = [(b[0] + b[1]) / 2 for b in bounds]

    # Ograniczenia w formacie scipy.optimize
    constraints = [(b[0], b[1]) for b in bounds]

    # Optymalizacja
    result = minimize(objective_function, initial_guess, bounds=constraints, method='SLSQP')

    if result.success:
        best_solution = result.x
        best_utility = -result.fun  # Odpowiada maksymalnej użyteczności
        best_utility_vals = UTA_function_continuous([best_solution], reference_points, lambda_param=lambda_param, use_polynomial=use_polynomial)
        return best_solution, best_utility, best_utility_vals
    else:
        raise ValueError("Optymalizacja nie powiodła się.")
        

# Funkcja do wyznaczania funkcji użyteczności dla wersji dyskretnej
def UTA_function_discrete(values, reference_points, lambda_param=1):
    """
    Aproksymuje funkcje użyteczności dla wersji dyskretnej.
    
    :param values: Lista wartości alternatyw w przestrzeni kryteriów
    :param reference_points: Lista punktów referencyjnych (najlepszy, najgorszy, neutralny)
    :param lambda_param: Współczynnik λ (preferencje decydenta)
    :return: Lista funkcji użyteczności dla wszystkich alternatyw
    """
    utility_values = []
    
    # Iterujemy przez alternatywy
    for val in values:
        alternative_utility = []
        
        # Iterujemy przez każde kryterium
        for i in range(len(val)):
            ref_values = [ref[i] for ref in reference_points]  # Zbiór punktów referencyjnych dla kryterium i
            # Interpolacja liniowa dla każdego kryterium
            interp_func = interp1d(ref_values, [-1, 0, 1], kind='linear', fill_value="extrapolate")
            utility_value = interp_func(val[i]) * lambda_param  # Użycie λ
            alternative_utility.append(utility_value)
        
        utility_values.append(alternative_utility)
    
    return np.array(utility_values)

# Funkcja do porównania alternatyw (dyskretna wersja)
def UTAstar_discrete(values, reference_points, lambda_param=1):
    """
    Metoda UTA Star dla wersji dyskretnej (wielokryterialnej) uwzględniająca preferencje decydenta.
    
    :param values: Lista wartości alternatyw w przestrzeni kryteriów
    :param reference_points: Lista punktów referencyjnych dla każdego kryterium
    :param lambda_param: Współczynnik λ (preferencje decydenta)
    :return: Wybrana alternatywa
    """
    # Wyznaczanie funkcji użyteczności
    utility_values = UTA_function_discrete(values, reference_points, lambda_param)
    
    # Wybór rozwiązania kompromisowego - wybieramy alternatywę z maksymalną sumą użyteczności
    ranking_values = np.sum(utility_values, axis=1)
    best_solution_idx = np.argmax(ranking_values)  # Wybieramy alternatywę z największą sumą funkcji użyteczności
    return best_solution_idx, utility_values, ranking_values