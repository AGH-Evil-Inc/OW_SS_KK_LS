import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from metody import (continuous_reference_set_method, discrete_reference_set_method,
                    fuzzy_topsis, topsis, UTAstar_continuous, UTAstar_discrete)

class OptimizationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Optymalizacja wielokryterialna")

        # Domyślnie plik danych jest pusty
        self.data_file = None
        self.sheet1_data = None
        self.sheet2_data = None

        # GUI Elements
        self.create_widgets()

    def create_widgets(self):
        # Button for loading data
        tk.Label(self.root, text="Załaduj dane:").grid(row=0, column=0, padx=10, pady=5, sticky='w')
        self.load_data_button = tk.Button(self.root, text="Wczytaj dane", command=self.load_data)
        self.load_data_button.grid(row=0, column=1, padx=10, pady=5)

        # Dropdown for method selection
        tk.Label(self.root, text="Wybierz metodę:").grid(row=1, column=0, padx=10, pady=5, sticky='w')
        self.method_var = tk.StringVar()
        self.method_dropdown = ttk.Combobox(self.root, textvariable=self.method_var, state='readonly')
        self.method_dropdown['values'] = [
            "Metoda ciągłego zbioru odniesienia",
            "Metoda dyskretnego zbioru odniesienia",
            "Fuzzy TOPSIS",
            "TOPSIS",
            "UTA Continuous",
            "UTA Discrete"
        ]
        self.method_dropdown.grid(row=1, column=1, padx=10, pady=5)

        # Button for running optimization
        self.run_button = tk.Button(self.root, text="Uruchom", command=self.run_optimization)
        self.run_button.grid(row=2, column=0, columnspan=2, pady=10)

        # Table for results
        self.result_table = ttk.Treeview(self.root, columns=("Rank", "Alternative", "Score"), show="headings")
        self.result_table.heading("Rank", text="Ranga")
        self.result_table.heading("Alternative", text="Alternatywa")
        self.result_table.heading("Score", text="Wynik")
        self.result_table.grid(row=3, column=0, columnspan=2, pady=10, padx=10, sticky='nsew')

        # Visualization button
        self.plot_button = tk.Button(self.root, text="Pokaż wykres", command=self.show_plot)
        self.plot_button.grid(row=4, column=0, columnspan=2, pady=10)

    def load_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
        if file_path:
            try:
                self.data_file = file_path
                self.sheet1_data = pd.read_excel(self.data_file, sheet_name='Sheet1')
                self.sheet2_data = pd.read_excel(self.data_file, sheet_name='Sheet2')
                messagebox.showinfo("Sukces", "Dane zostały wczytane poprawnie.")
            except Exception as e:
                messagebox.showerror("Błąd", f"Nie udało się wczytać danych: {e}")

    def run_optimization(self):
        if self.sheet1_data is None or self.sheet2_data is None:
            messagebox.showwarning("Błąd", "Najpierw wczytaj dane.")
            return

        method = self.method_var.get()

        if not method:
            messagebox.showwarning("Błąd", "Wybierz metodę optymalizacji.")
            return

        try:
            if method == "Metoda ciągłego zbioru odniesienia":
                ranking, scores, _ = continuous_reference_set_method(
                    self.sheet1_data.iloc[:, :-1].values, [1, -1],
                    self.sheet1_data.iloc[:, :-1].min().values, self.sheet1_data.iloc[:, :-1].max().values
                )
            elif method == "Metoda dyskretnego zbioru odniesienia":
                ranking, scores = discrete_reference_set_method(
                    self.sheet1_data.iloc[:, :-1].values, [1, -1],
                    self.sheet1_data.iloc[:, :-1].min().values, self.sheet1_data.iloc[:, :-1].max().values
                )
            elif method == "Fuzzy TOPSIS":
                weights = np.ones(self.sheet1_data.shape[1] - 1) / (self.sheet1_data.shape[1] - 1)
                ranking = fuzzy_topsis(
                    self.sheet1_data.iloc[:, :-1].values, weights, ['max', 'min']
                )
                scores = ranking  # Similarity to ideal
            elif method == "TOPSIS":
                weights = np.ones(self.sheet1_data.shape[1] - 1) / (self.sheet1_data.shape[1] - 1)
                ranking, scores = topsis(
                    self.sheet1_data.iloc[:, :-1].values, weights, [1, -1]
                )
            elif method == "UTA Continuous":
                best, utility = UTAstar_continuous(
                    self.sheet2_data.iloc[:, :-1].values, self.sheet2_data.iloc[:3, :-1].values
                )
                ranking = np.argsort(-utility.sum(axis=1))
                scores = utility.sum(axis=1)
            elif method == "UTA Discrete":
                best, utility = UTAstar_discrete(
                    self.sheet2_data.iloc[:, :-1].values, self.sheet2_data.iloc[:3, :-1].values
                )
                ranking = np.argsort(-utility.sum(axis=1))
                scores = utility.sum(axis=1)
            else:
                messagebox.showerror("Błąd", "Nieznana metoda optymalizacji.")
                return

            self.update_table(ranking, scores)
        except ValueError as e:
            messagebox.showerror("Błąd", f"Wystąpił problem z danymi: {e}")

    def update_table(self, ranking, scores):
        for row in self.result_table.get_children():
            self.result_table.delete(row)
        for idx, rank in enumerate(ranking):
            self.result_table.insert("", "end", values=(idx + 1, rank + 1, scores[rank]))

    def show_plot(self):
        if self.sheet1_data is None:
            messagebox.showwarning("Błąd", "Najpierw wczytaj dane.")
            return

        dimensions = self.sheet1_data.shape[1] - 1

        if dimensions == 2:
            plt.scatter(self.sheet1_data.iloc[:, 0], self.sheet1_data.iloc[:, 1])
            plt.xlabel("Kryterium 1")
            plt.ylabel("Kryterium 2")
            plt.title("Wykres 2D")
        elif dimensions == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(
                self.sheet1_data.iloc[:, 0],
                self.sheet1_data.iloc[:, 1],
                self.sheet1_data.iloc[:, 2]
            )
            ax.set_xlabel("Kryterium 1")
            ax.set_ylabel("Kryterium 2")
            ax.set_zlabel("Kryterium 3")
            plt.title("Wykres 3D")
        else:
            messagebox.showinfo("Informacja", "Wizualizacja dostępna tylko dla problemów 2D i 3D.")
            return

        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = OptimizationApp(root)
    root.mainloop()
