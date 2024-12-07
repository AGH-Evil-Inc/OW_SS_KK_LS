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


def UTA_function_discrete(values, reference_points, lambda_param=1):
    utility_values = []
    for val in values:
        alternative_utility = []
        for i in range(len(val)):
            ref_values = [ref[i] for ref in reference_points]
            interp_func = interp1d(ref_values, [0, 1, 0], kind='linear', fill_value="extrapolate")
            utility_value = interp_func(val[i]) * lambda_param
            alternative_utility.append(utility_value)
        utility_values.append(alternative_utility)
    return np.array(utility_values)


def UTAstar_discrete(values, reference_points, lambda_param=1):
    utility_values = UTA_function_discrete(values, reference_points, lambda_param)
    summed_utility = np.sum(utility_values, axis=1)
    best_solution_idx = np.argmax(summed_utility)
    return best_solution_idx, utility_values


def UTA_function_continuous(values, reference_points, lambda_param=1, use_polynomial=False):
    utility_values = []
    for i in range(len(reference_points[0])):
        ref_values = [ref[i] for ref in reference_points]
        if use_polynomial:
            coef = np.polyfit(ref_values, [0, 1, 0], 2)
            poly_func = np.poly1d(coef)
            utility_values.append(poly_func(values[:, i]) * lambda_param)
        else:
            interp_func = interp1d(ref_values, [0, 1, 0], kind='linear', fill_value="extrapolate")
            utility_values.append(interp_func(values[:, i]) * lambda_param)
    return np.array(utility_values).T


def UTAstar_continuous(values, reference_points, lambda_param=1, use_polynomial=False):
    utility_values = UTA_function_continuous(values, reference_points, lambda_param, use_polynomial)
    summed_utility = np.sum(utility_values, axis=1)
    best_solution = np.argmax(summed_utility)
    return best_solution, utility_values
