import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interp1d


def continuous_reference_set_method(alternatives, directions, a, b, n_extra=200):
    rng = np.random.default_rng()
    M = alternatives.shape[1]
    # Generujemy dodatkowe punkty w przestrzeni [a,b]
    extra_points = np.zeros((n_extra, M))
    for i in range(M):
        low = min(a[i], b[i])
        high = max(a[i], b[i])
        extra_points[:, i] = rng.uniform(low, high, n_extra)
    # Łączymy oryginalne alternatywy z nowo wygenerowanymi punktami
    all_points = np.vstack([alternatives, extra_points])
    # Funkcja normalizująca z uwzględnieniem kierunku optymalizacji
    def normalize(points, a, b, directions):
        norm_alt = np.zeros_like(points, dtype=float)
        for i in range(len(directions)):
            if directions[i] == -1:  # Minimalizacja
                norm_alt[:, i] = (points[:, i] - a[i]) / (b[i] - a[i])
            else:  # Maksymalizacja
                norm_alt[:, i] = (b[i] - points[:, i]) / (b[i] - a[i])
        return norm_alt
    # Funkcja scoringowa
    def scoring_function(points, a, b, directions):
        norm_alt = normalize(points, a, b, directions)
        distance_to_a = np.linalg.norm(norm_alt - np.zeros((len(points), len(a))), axis=1)
        distance_to_b = np.linalg.norm(norm_alt - np.ones((len(points), len(a))), axis=1)
        C = distance_to_b / (distance_to_a + distance_to_b)
        return C
    scores = scoring_function(all_points, a, b, directions)
    ranking = np.argsort(-scores)  # sortujemy malejąco po scores
    return ranking, scores, all_points



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


def fuzzy_topsis(decision_matrix, weights, criteria_type):
    decision_matrix = np.array(decision_matrix)
    # Zakładamy, że decision_matrix jest w formacie m x n x 3
    norm_matrix = np.zeros_like(decision_matrix, dtype=float)
    for j in range(decision_matrix.shape[1]):
        if criteria_type[j] == 'max':
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
        if criteria_type[j] == 'max':
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


#  Funkcja do wyznaczania funkcji użyteczności (ciągła wersja z preferencjami)
def UTA_function_continuous(values, reference_points, lambda_param=1, use_polynomial=False):
    """
    Aproksymuje funkcje użyteczności w przestrzeni ciągłej dla wielu kryteriów.
    Uwzględnia preferencje decydenta przez współczynnik λ i punkty referencyjne.
    
    :param values: Lista wartości kryteriów dla alternatyw
    :param reference_points: Lista punktów referencyjnych (np. [najlepszy, najgorszy, neutralny])
    :param lambda_param: Współczynnik λ (preferencje decydenta)
    :param use_polynomial: Jeśli True, stosuje aproksymację wielomianową, w przeciwnym razie interpolację
    :return: Funkcje użyteczności
    """
    utility_values = []
    for i in range(len(reference_points[0])):  # Iterujemy po kryteriach
        ref_values = [ref[i] for ref in reference_points]  # Zbiór punktów referencyjnych dla kryterium i
        if use_polynomial:
            # Przy użyciu funkcji aproksymacji wielomianowej
            coef = np.polyfit(ref_values, [0, 1, 0], 2)  # Przykład: funkcja kwadratowa
            poly_func = np.poly1d(coef)
            utility_values.append(poly_func(values[:, i]) * lambda_param)  # Użycie λ
        else:
            # Interpolacja liniowa
            interp_func = interp1d(ref_values, [0, 1, 0], kind='linear', fill_value="extrapolate")
            utility_values.append(interp_func(values[:, i]) * lambda_param)  # Użycie λ
    return np.array(utility_values).T  # Zwracamy funkcje użyteczności dla wszystkich alternatyw

# Funkcja do porównania alternatyw (ciągła wersja z uwzględnieniem preferencji)
def UTAstar_continuous(values, reference_points, lambda_param=1, use_polynomial=False):
    """
    Metoda UTA Star dla wersji ciągłej (wielokryterialnej) uwzględniająca preferencje decydenta.
    
    :param values: Lista wartości alternatyw w przestrzeni kryteriów
    :param reference_points: Lista punktów referencyjnych dla każdego kryterium
    :param lambda_param: Współczynnik λ (preferencje decydenta)
    :param use_polynomial: Jeśli True, stosuje aproksymację wielomianową, w przeciwnym razie interpolację
    :return: Wybrana alternatywa
    """
    # Wyznaczanie funkcji użyteczności
    utility_values = UTA_function_continuous(values, reference_points, lambda_param, use_polynomial)
    
    # Wybór rozwiązania kompromisowego - wybieramy alternatywę z maksymalną sumą użyteczności
    summed_utility = np.sum(utility_values, axis=1)
    best_solution = np.argmax(summed_utility)  # Wybieramy alternatywę z największą sumą funkcji użyteczności
    return best_solution, utility_values

import numpy as np
from scipy.interpolate import interp1d

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
            interp_func = interp1d(ref_values, [0, 1, 0], kind='linear', fill_value="extrapolate")
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
    summed_utility = np.sum(utility_values, axis=1)
    best_solution_idx = np.argmax(summed_utility)  # Wybieramy alternatywę z największą sumą funkcji użyteczności
    return best_solution_idx, utility_values
