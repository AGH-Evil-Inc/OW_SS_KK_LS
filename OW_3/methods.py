import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interp1d


def continuous_reference_set_method(objective_functions, directions, a, b, bounds, x0=None):
    # Kod wklejony z poprzednich przykładów
    num_criteria = len(objective_functions)
    def F(x):
        return np.array([f(x) for f in objective_functions])
    def normalize(f_values, a, b, directions):
        norm_values = np.zeros_like(f_values)
        for i in range(len(f_values)):
            if directions[i] == -1:  # Minimalizacja
                norm_values[i] = (f_values[i] - a[i]) / (b[i] - a[i])
            else:  # Maksymalizacja
                norm_values[i] = (b[i] - f_values[i]) / (b[i] - a[i])
        return norm_values
    def scoring_function(x):
        f_values = F(x)
        f_norm = normalize(f_values, a, b, directions)
        distance_to_a = np.linalg.norm(f_norm - np.zeros(len(f_norm)))
        distance_to_b = np.linalg.norm(f_norm - np.ones(len(f_norm)))
        C = distance_to_b / (distance_to_a + distance_to_b)
        return C
    def objective(x):
        return -scoring_function(x)
    if x0 is None:
        x0 = np.zeros(len(bounds))
    result = minimize(
        objective,
        x0=x0,
        bounds=bounds,
        method='SLSQP'
    )
    optimal_x = result.x
    optimal_f_values = F(optimal_x)
    optimal_score = scoring_function(optimal_x)
    return result, optimal_f_values, optimal_score


def discrete_reference_set_method(alternatives, directions, a, b):
    def normalize(alternatives, a, b, directions):
        norm_alt = np.zeros_like(alternatives, dtype=float)
        for i in range(len(directions)):
            if directions[i] == -1:
                norm_alt[:, i] = (alternatives[:, i] - a[i]) / (b[i] - a[i])
            else:
                norm_alt[:, i] = (b[i] - alternatives[:, i]) / (b[i] - a[i])
        return norm_alt
    def scoring_function(alternatives, a, b, directions):
        norm_alt = normalize(alternatives, a, b, directions)
        distance_to_a = np.linalg.norm(norm_alt - np.zeros((len(alternatives), len(a))), axis=1)
        distance_to_b = np.linalg.norm(norm_alt - np.ones((len(alternatives), len(a))), axis=1)
        C = distance_to_b / (distance_to_a + distance_to_b)
        return C
    scores = scoring_function(alternatives, a, b, directions)
    ranking = np.argsort(-scores)
    return ranking, scores


# def fuzzy_topsis(decision_matrix, weights, criteria_type):
#     decision_matrix = np.array(decision_matrix)
#     # Zakładamy, że decision_matrix jest w formacie m x n x 3
#     norm_matrix = np.zeros_like(decision_matrix, dtype=float)
#     for j in range(decision_matrix.shape[1]):
#         if criteria_type[j] == 'max':
#             max_value = np.max(decision_matrix[:, j, 2])
#             norm_matrix[:, j, 0] = decision_matrix[:, j, 0] / max_value
#             norm_matrix[:, j, 1] = decision_matrix[:, j, 1] / max_value
#             norm_matrix[:, j, 2] = decision_matrix[:, j, 2] / max_value
#         else:
#             min_value = np.min(decision_matrix[:, j, 0])
#             norm_matrix[:, j, 0] = min_value / decision_matrix[:, j, 2]
#             norm_matrix[:, j, 1] = min_value / decision_matrix[:, j, 1]
#             norm_matrix[:, j, 2] = min_value / decision_matrix[:, j, 0]
#     weighted_matrix = np.zeros_like(norm_matrix, dtype=float)
#     for j in range(norm_matrix.shape[1]):
#         weighted_matrix[:, j, 0] = norm_matrix[:, j, 0] * weights[j]
#         weighted_matrix[:, j, 1] = norm_matrix[:, j, 1] * weights[j]
#         weighted_matrix[:, j, 2] = norm_matrix[:, j, 2] * weights[j]
#     fpis = np.zeros((decision_matrix.shape[1], 3), dtype=float)
#     fnis = np.zeros((decision_matrix.shape[1], 3), dtype=float)
#     for j in range(weighted_matrix.shape[1]):
#         if criteria_type[j] == 'max':
#             fpis[j] = np.max(weighted_matrix[:, j, :], axis=0)
#             fnis[j] = np.min(weighted_matrix[:, j, :], axis=0)
#         else:
#             fpis[j] = np.min(weighted_matrix[:, j, :], axis=0)
#             fnis[j] = np.max(weighted_matrix[:, j, :], axis=0)
#     distance_to_fpis = np.sqrt(np.sum(np.power(weighted_matrix - fpis, 2), axis=(1, 2)))
#     distance_to_fnis = np.sqrt(np.sum(np.power(weighted_matrix - fnis, 2), axis=(1, 2)))
#     with np.errstate(divide='ignore', invalid='ignore'):
#         similarity_to_ideal = distance_to_fnis / (distance_to_fpis + distance_to_fnis)
#         similarity_to_ideal[np.isnan(similarity_to_ideal)] = 0
#     return similarity_to_ideal
def fuzzy_topsis(decision_matrix, weights, criteria_types, delta=0.1):
    """
    Fuzzy Topsis w wersji ciągłej. 
    Przyjmuje:
    - objective_functions: lista funkcji celu
    - X: macierz m x d punktów (m alternatyw, d wymiar przestrzeni decyzyjnej)
    - weights: wagi kryteriów (tablica n-elementowa)
    - criteria_types: lista typu 'max' lub 'min' dla każdego kryterium
    - delta: parametr określający 'rozmycie' wartości (np. ±10%)

    Zwraca wektor similarity_to_ideal.
    """
    n = len(objective_functions)
    m = X.shape[0]
    # Obliczamy wartości funkcji celu dla każdego punktu
    crisp_values = np.zeros((m, n))
    for i in range(m):
        x = X[i]
        crisp_values[i, :] = [f(x) for f in objective_functions]

    # Tworzymy macierz rozmytą (m x n x 3)
    # Załóżmy prosty model: [val*(1-delta), val, val*(1+delta)] dla kryterium typu 'max'
    # i odwrotnie dla 'min'. Można też pozostawić [val, val, val], jeśli chcemy brak niepewności.
    decision_matrix = np.zeros((m, n, 3))
    for i in range(m):
        for j in range(n):
            v = crisp_values[i, j]
            if criteria_types[j] == 'max':
                decision_matrix[i, j, 0] = v * (1 - delta)
                decision_matrix[i, j, 1] = v
                decision_matrix[i, j, 2] = v * (1 + delta)
            else:
                # Dla minimalizacji odwrotnie. Można też użyć analogii:
                # Dla 'min', lepiej aby wartości 'lepsze' były niższe.
                # Załóżmy symetrię:
                decision_matrix[i, j, 0] = v * (1 - delta)
                decision_matrix[i, j, 1] = v
                decision_matrix[i, j, 2] = v * (1 + delta)

    # Teraz implementujemy fuzzy_topsis jak wcześniej
    # Normalizacja
    norm_matrix = np.zeros_like(decision_matrix, dtype=float)
    for j in range(n):
        if criteria_types[j] == 'max':
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
    for j in range(n):
        weighted_matrix[:, j, 0] = norm_matrix[:, j, 0] * weights[j]
        weighted_matrix[:, j, 1] = norm_matrix[:, j, 1] * weights[j]
        weighted_matrix[:, j, 2] = norm_matrix[:, j, 2] * weights[j]

    fpis = np.zeros((n, 3), dtype=float)
    fnis = np.zeros((n, 3), dtype=float)
    for j in range(n):
        if criteria_types[j] == 'max':
            fpis[j] = np.max(weighted_matrix[:, j, :], axis=0)
            fnis[j] = np.min(weighted_matrix[:, j, :], axis=0)
        else:
            fpis[j] = np.min(weighted_matrix[:, j, :], axis=0)
            fnis[j] = np.max(weighted_matrix[:, j, :], axis=0)

    distance_to_fpis = np.sqrt(np.sum((weighted_matrix - fpis)**2, axis=(1, 2)))
    distance_to_fnis = np.sqrt(np.sum((weighted_matrix - fnis)**2, axis=(1, 2)))

    with np.errstate(divide='ignore', invalid='ignore'):
        similarity_to_ideal = distance_to_fnis / (distance_to_fpis + distance_to_fnis)
        similarity_to_ideal[np.isnan(similarity_to_ideal)] = 0

    return similarity_to_ideal


def topsis(decision_matrix, weights, criteria):
    decision_matrix = np.array(decision_matrix)
    norm_matrix = decision_matrix / np.sqrt((decision_matrix ** 2).sum(axis=0))
    weighted_matrix = norm_matrix * weights
    ideal_best = np.max(weighted_matrix, axis=0) * (criteria == 1) + np.min(weighted_matrix, axis=0) * (criteria == -1)
    ideal_worst = np.min(weighted_matrix, axis=0) * (criteria == 1) + np.max(weighted_matrix, axis=0) * (criteria == -1)
    dist_to_ideal = np.sqrt(((weighted_matrix - ideal_best) ** 2).sum(axis=1))
    dist_to_worst = np.sqrt(((weighted_matrix - ideal_worst) ** 2).sum(axis=1))
    score = dist_to_worst / (dist_to_ideal + dist_to_worst)
    ranking = np.argsort(score)[::-1]
    return ranking, score


# def UTA_function_discrete(values, reference_points, lambda_param=1):
#     utility_values = []
#     for val in values:
#         alternative_utility = []
#         for i in range(len(val)):
#             ref_values = [ref[i] for ref in reference_points]
#             interp_func = interp1d(ref_values, [0, 1, 0], kind='linear', fill_value="extrapolate")
#             utility_value = interp_func(val[i]) * lambda_param
#             alternative_utility.append(utility_value)
#         utility_values.append(alternative_utility)
#     return np.array(utility_values)


# def UTAstar_discrete(values, reference_points, lambda_param=1):
#     utility_values = UTA_function_discrete(values, reference_points, lambda_param)
#     summed_utility = np.sum(utility_values, axis=1)
#     best_solution_idx = np.argmax(summed_utility)
#     return best_solution_idx, utility_values


# def UTA_function_continuous(values, reference_points, lambda_param=1, use_polynomial=False):
#     utility_values = []
#     for i in range(len(reference_points[0])):
#         ref_values = [ref[i] for ref in reference_points]
#         if use_polynomial:
#             coef = np.polyfit(ref_values, [0, 1, 0], 2)
#             poly_func = np.poly1d(coef)
#             utility_values.append(poly_func(values[:, i]) * lambda_param)
#         else:
#             interp_func = interp1d(ref_values, [0, 1, 0], kind='linear', fill_value="extrapolate")
#             utility_values.append(interp_func(values[:, i]) * lambda_param)
#     return np.array(utility_values).T


# def UTAstar_continuous(values, reference_points, lambda_param=1, use_polynomial=False):
#     utility_values = UTA_function_continuous(values, reference_points, lambda_param, use_polynomial)
#     summed_utility = np.sum(utility_values, axis=1)
#     best_solution = np.argmax(summed_utility)
#     return best_solution, utility_values

def UTA_function_discrete(values, reference_points, lambda_param=1):
    """
    Funkcja oblicza wartości użyteczności dla dyskretnych alternatyw na podstawie punktów odniesienia.
    Zakładamy:
    - values: macierz m x n, m - liczba alternatyw, n - liczba kryteriów
    - reference_points: macierz k x n, k - liczba punktów odniesienia na każde kryterium
    - Punkty odniesienia definiują k wartości dla danego kryterium. Użyteczność jest interpolowana
      liniowo od 0 do 1 między min i max wartością punktów odniesienia.
    """
    values = np.array(values)
    reference_points = np.array(reference_points)
    m, n = values.shape
    # utility_values będzie miało rozmiar m x n
    utility_values = np.zeros((m, n))

    for i in range(n):
        ref_vals = reference_points[:, i]
        # Sortujemy punkty odniesienia według wartości w danym kryterium
        sorted_idx = np.argsort(ref_vals)
        ref_vals_sorted = ref_vals[sorted_idx]

        # Generujemy wartości użyteczności w sposób liniowy od 0 do 1
        # dla k punktów odniesienia k: [0, 1/(k-1), 2/(k-1), ..., 1]
        k = len(ref_vals_sorted)
        y_values = np.linspace(0, 1, k)

        # Tworzymy interpolację liniową
        interp_func = interp1d(ref_vals_sorted, y_values, kind='linear', fill_value="extrapolate")

        # Stosujemy do wszystkich alternatyw dla tego kryterium
        utility_values[:, i] = interp_func(values[:, i]) * lambda_param

    return utility_values


def UTAstar_discrete(values, reference_points, lambda_param=1):
    """
    Wersja UTAstar dla problemu dyskretnego.
    """
    utility_values = UTA_function_discrete(values, reference_points, lambda_param)
    summed_utility = np.sum(utility_values, axis=1)
    best_solution_idx = np.argmax(summed_utility)
    return best_solution_idx, utility_values


# def UTA_function_continuous(values, reference_points, lambda_param=1, use_polynomial=False):
#     """
#     Funkcja oblicza wartości użyteczności dla ciągłego problemu.
#     Zakładamy:
#     - values: macierz m x n (m punktów w przestrzeni decyzyjnej, n - kryteriów)
#     - reference_points: macierz k x n (k punktów odniesienia na każde kryterium)
#     - Jeśli use_polynomial=True, dopasowujemy wielomian do punktów odniesienia i wartości użyteczności.
#       W przeciwnym razie interpolacja liniowa.

#     Użyteczność jest definiowana jako rosnąca funkcja pomiędzy min a max z reference_points.
#     """
#     values = np.array(values)
#     reference_points = np.array(reference_points)
#     m, n = values.shape
#     # utility_values będzie macierzą m x n
#     utility_values = np.zeros((m, n))

#     for i in range(n):
#         ref_vals = reference_points[:, i]
#         # Sortujemy punkty odniesienia
#         sorted_idx = np.argsort(ref_vals)
#         ref_vals_sorted = ref_vals[sorted_idx]

#         # Generujemy wartości od 0 do 1 dla k punktów
#         k = len(ref_vals_sorted)
#         y_values = np.linspace(0, 1, k)

#         if use_polynomial:
#             # Dopasowujemy wielomian 2-go stopnia do punktów (ref_vals_sorted, y_values)
#             # Możemy zmienić stopień wielomianu, jeśli chcemy bardziej skomplikowaną krzywą.
#             coef = np.polyfit(ref_vals_sorted, y_values, 2)
#             poly_func = np.poly1d(coef)
#             utility_values[:, i] = poly_func(values[:, i]) * lambda_param
#         else:
#             # Interpolacja liniowa
#             interp_func = interp1d(ref_vals_sorted, y_values, kind='linear', fill_value="extrapolate")
#             utility_values[:, i] = interp_func(values[:, i]) * lambda_param

#     return utility_values

def UTAstar_continuous(objective_functions, X, reference_points, lambda_param=1, use_polynomial=False):
    """
    UTAstar dla problemu ciągłego bazujący na funkcjach celu i zestawie punktów X.
    - objective_functions: lista funkcji celu f_i: R^d -> R
    - X: macierz m x d, m punktów w przestrzeni decyzyjnej, d wymiar
    - reference_points: macierz k x n, gdzie n to liczba kryteriów (funkcji celu)
    - lambda_param, use_polynomial: parametry UTA_function_continuous
    
    Zwraca indeks najlepszego rozwiązania oraz macierz wartości użyteczności.
    """
    # Obliczamy wartości funkcji celu dla każdego punktu X
    # n = liczba kryteriów
    # m = liczba punktów
    n = len(objective_functions)
    m = X.shape[0]
    values = np.zeros((m, n))
    for i in range(m):
        x = X[i]
        values[i, :] = [f(x) for f in objective_functions]

    utility_values = UTA_function_continuous(values, reference_points, lambda_param, use_polynomial)
    summed_utility = np.sum(utility_values, axis=1)
    best_solution = np.argmax(summed_utility)
    return best_solution, utility_values


def UTAstar_continuous(values, reference_points, lambda_param=1, use_polynomial=False):
    """
    Wersja UTAstar dla problemu ciągłego.
    """
    utility_values = UTA_function_continuous(values, reference_points, lambda_param, use_polynomial)
    summed_utility = np.sum(utility_values, axis=1)
    best_solution = np.argmax(summed_utility)
    return best_solution, utility_values