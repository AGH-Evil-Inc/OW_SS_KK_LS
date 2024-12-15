import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
from methods import (
    continuous_reference_set_method,
    discrete_reference_set_method,
    fuzzy_topsis,
    topsis,
    UTAstar_discrete,
    UTAstar_continuous
)

def load_file():
    file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
    if file_path:
        try:
            data = pd.read_excel(file_path)
            messagebox.showinfo("Success", "File loaded successfully!")
            return data
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {e}")
    return None

def load_data():
    global data
    data = load_file()

def execute_method():
    global data

    if data is None:
        messagebox.showwarning("Warning", "Please load the data file first.")
        return

    method = method_var.get()

    try:
        if method == "Continuous Reference Set Method":
            a = np.min(data.values, axis=0)
            b = np.max(data.values, axis=0)
            directions = np.array([1] * data.shape[1])
            ranking, scores, _ = continuous_reference_set_method(data.values, directions, a, b)

        elif method == "Discrete Reference Set Method":
            a = np.min(data.values, axis=0)
            b = np.max(data.values, axis=0)
            directions = np.array([1] * data.shape[1])
            ranking, scores = discrete_reference_set_method(data.values, directions, a, b)

        elif method == "TOPSIS":
            weights = np.ones(data.shape[1]) / data.shape[1]
            criteria = np.array([1] * data.shape[1])
            ranking, scores = topsis(data.values, weights, criteria)

        elif method == "Fuzzy TOPSIS":
            weights = np.ones(data.shape[1]) / data.shape[1]
            criteria_type = ["max"] * data.shape[1]
            decision_matrix = np.stack([data.values, data.values, data.values], axis=2)
            scores = fuzzy_topsis(decision_matrix, weights, criteria_type)
            ranking = np.argsort(-scores)

        elif method == "UTA*: Discrete":
            reference_points = np.array([np.min(data.values, axis=0), np.max(data.values, axis=0)])
            if reference_points.shape[0] != 2 or reference_points.shape[1] != data.shape[1]:
                raise ValueError("Reference points must match the number of criteria and contain at least two points for interpolation.")
            lambda_param = 1
            best_solution, utility_values = UTAstar_discrete(data.values, reference_points, lambda_param)
            scores = utility_values.sum(axis=1)
            ranking = np.argsort(-scores)

        elif method == "UTA*: Continuous":
            reference_points = np.array([np.min(data.values, axis=0), np.max(data.values, axis=0)])
            if reference_points.shape[0] != 2 or reference_points.shape[1] != data.shape[1]:
                raise ValueError("Reference points must match the number of criteria and contain at least two points for interpolation.")
            lambda_param = 1
            best_solution, utility_values = UTAstar_continuous(data.values, reference_points, lambda_param)
            scores = utility_values.sum(axis=1)
            ranking = np.argsort(-scores)

        else:
            messagebox.showerror("Error", "Invalid method selected.")
            return

        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, "Ranking:\n")
        for i, rank in enumerate(ranking):
            result_text.insert(tk.END, f"{i + 1}. Point {rank + 1} (Score: {scores[rank]:.4f})\n")

    except ValueError as ve:
        messagebox.showerror("Input Error", f"{ve}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to execute method: {e}")

data = None

# Tworzenie głównego okna
root = tk.Tk()
root.title("Multi-Criteria Optimization GUI")
root.geometry("600x400")

# Przycisk do wczytywania pliku
load_button = tk.Button(root, text="Load Data File", command=load_data)
load_button.pack(pady=10)

# Wybór metody
method_var = tk.StringVar(value="Continuous Reference Set Method")
methods = [
    "Continuous Reference Set Method",
    "Discrete Reference Set Method",
    "TOPSIS",
    "Fuzzy TOPSIS",
    "UTA*: Discrete",
    "UTA*: Continuous"
]
method_menu = tk.OptionMenu(root, method_var, *methods)
method_menu.pack(pady=10)

# Przycisk do uruchamiania metody
execute_button = tk.Button(root, text="Execute Method", command=execute_method)
execute_button.pack(pady=10)

# Pole tekstowe do wyświetlania wyników
result_text = tk.Text(root, height=15, width=70)
result_text.pack(pady=10)

# Uruchamianie pętli głównej
root.mainloop()
