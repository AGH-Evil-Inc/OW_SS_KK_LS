import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import pandas as pd
import numpy as np
from methods import (topsis, discrete_reference_set_method, UTAstar_discrete,
                     continuous_reference_set_method, UTAstar_continuous, fuzzy_topsis)

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

class App:
    def __init__(self, master):
        self.master = master
        master.title("Optymalizacja Wielokryterialna")

        self.method_var = tk.StringVar()
        self.methods = [
            "Fuzzy Topsis-ciągły",
            "RSM-ciągły",
            "UTAstar-ciągły",
            "Topsis-dyskretny",
            "RSM-dyskretny",
            "UTAstar-dyskretny"
        ]
        
        # Ramka wyboru metody
        method_frame = tk.Frame(master)
        method_frame.pack(pady=10)
        
        tk.Label(method_frame, text="Wybierz metodę:").pack(side=tk.LEFT, padx=5)
        self.method_combobox = ttk.Combobox(method_frame, values=self.methods, textvariable=self.method_var, state='readonly')
        self.method_combobox.set(self.methods[0])
        self.method_combobox.pack(side=tk.LEFT)

        # Ramka ładowania danych
        data_frame = tk.Frame(master)
        data_frame.pack(pady=10)

        self.alternatives_file = None
        self.reference_file = None

        tk.Button(data_frame, text="Wczytaj alternatywy (alternatywy.xlsx)", command=self.load_alternatives).pack(side=tk.LEFT, padx=5)
        tk.Button(data_frame, text="Wczytaj punkty odniesienia (klasy.xlsx)", command=self.load_reference).pack(side=tk.LEFT, padx=5)
        
        # Ramka na tabele
        tables_frame = tk.Frame(master)
        tables_frame.pack(pady=10)
        
        # Tabela alternatyw
        self.alternatives_tree = ttk.Treeview(tables_frame, show='headings')
        self.alternatives_tree.pack(side=tk.LEFT, padx=10)
        
        # Tabela wyników
        self.results_tree = ttk.Treeview(tables_frame, show='headings')
        self.results_tree.pack(side=tk.LEFT, padx=10)

        # Ramka na przyciski operacji
        action_frame = tk.Frame(master)
        action_frame.pack(pady=10)
        tk.Button(action_frame, text="Uruchom metodę", command=self.run_method).pack(side=tk.LEFT, padx=10)
        
        # Ramka na wykres
        self.plot_frame = tk.Frame(master)
        self.plot_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        
        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.alternatives_df = None
        self.reference_df = None

    def load_alternatives(self):
        file_path = filedialog.askopenfilename(title="Wybierz plik z alternatywami", filetypes=[("Excel files","*.xlsx")])
        if file_path:
            self.alternatives_file = file_path
            self.alternatives_df = pd.read_excel(file_path)
            self.show_data_in_tree(self.alternatives_tree, self.alternatives_df)

    def load_reference(self):
        file_path = filedialog.askopenfilename(title="Wybierz plik z punktami odniesienia", filetypes=[("Excel files","*.xlsx")])
        if file_path:
            self.reference_file = file_path
            self.reference_df = pd.read_excel(file_path)

    def show_data_in_tree(self, tree, df):
        # Czyścimy starą tabelę
        tree.delete(*tree.get_children())
        tree["columns"] = list(df.columns)
        for col in df.columns:
            tree.heading(col, text=col)
            tree.column(col, anchor=tk.CENTER, width=100)
        
        for index, row in df.iterrows():
            tree.insert("", "end", values=list(row))

    def run_method(self):
        if self.alternatives_df is None:
            messagebox.showerror("Błąd", "Brak danych alternatyw.")
            return
        method_name = self.method_var.get()
        
        # Przygotowujemy dane
        criteria_columns = [c for c in self.alternatives_df.columns if c not in ["Nr alternatywy", "Nazwa alternatywy"]]
        if len(criteria_columns) < 2:
            messagebox.showerror("Błąd", "Wymagane co najmniej 2 kryteria.")
            return

        alternatives_data = self.alternatives_df[criteria_columns].to_numpy(dtype=float)
        num_criteria = alternatives_data.shape[1]

        # Sprawdzamy czy mamy plik z punktami odniesienia
        if self.reference_df is None:
            messagebox.showerror("Błąd", "Brak pliku z punktami odniesienia (klasy.xlsx).")
            return

        # Upewniamy się, że kolumny kryterialne występują również w reference_df
        for col in criteria_columns:
            if col not in self.reference_df.columns:
                messagebox.showerror("Błąd", f"Kolumna {col} nie znajduje się w pliku punktów odniesienia.")
                return

        # Pobieramy punkty odniesienia
        reference_points = self.reference_df[criteria_columns].to_numpy()
        # Definiujemy wektory a i b jako min i max z referencji (dla metod RSM)
        a = reference_points.min(axis=0)
        b = reference_points.max(axis=0)

        directions = np.ones(num_criteria)  # Zakładamy maksymalizację wszystkich kryteriów
        weights = np.ones(num_criteria) / num_criteria
        criteria_types = ["max"]*num_criteria

        # Wywołanie metody
        if method_name == "Fuzzy Topsis-ciągły":
            fuzzy_data = np.stack([alternatives_data, alternatives_data, alternatives_data], axis=-1)
            scores = fuzzy_topsis(fuzzy_data, weights, criteria_types)
            ranking = np.argsort(scores)[::-1]
            results_df = pd.DataFrame({
                "Nr alternatywy": self.alternatives_df["Nr alternatywy"],
                "Nazwa alternatywy": self.alternatives_df["Nazwa alternatywy"],
                "Wynik": scores
            })
            results_df = results_df.iloc[ranking].reset_index(drop=True)

        elif method_name == "RSM-ciągły":
            # W kodzie przykładowym używamy RSM-ciągły jakby dyskretnego zestawu punktów,
            # bo pełna implementacja continuous_reference_set_method wymagałaby funkcji celu.
            # Tutaj tylko demo scoringu:
            def scoring_function(alt):
                norm_alt = np.zeros(len(alt))
                for i in range(len(alt)):
                    if directions[i] == -1:
                        norm_alt[i] = (alt[i]-a[i])/(b[i]-a[i])
                    else:
                        norm_alt[i] = (b[i]-alt[i])/(b[i]-a[i])
                dist_a = np.linalg.norm(norm_alt - 0)
                dist_b = np.linalg.norm(norm_alt - 1)
                C = dist_b/(dist_a+dist_b)
                return C
            scores = np.array([scoring_function(x) for x in alternatives_data])
            ranking = np.argsort(scores)[::-1]
            results_df = pd.DataFrame({
                "Nr alternatywy": self.alternatives_df["Nr alternatywy"].iloc[ranking],
                "Nazwa alternatywy": self.alternatives_df["Nazwa alternatywy"].iloc[ranking],
                "Wynik": scores[ranking]
            }).reset_index(drop=True)

        elif method_name == "UTAstar-ciągły":
            # Korzystamy bezpośrednio z reference_points
            best_idx, utility_values = UTAstar_continuous(alternatives_data, reference_points)
            scores = utility_values.sum(axis=1)
            ranking = np.argsort(scores)[::-1]
            results_df = pd.DataFrame({
                "Nr alternatywy": self.alternatives_df["Nr alternatywy"].iloc[ranking],
                "Nazwa alternatywy": self.alternatives_df["Nazwa alternatywy"].iloc[ranking],
                "Wynik": scores[ranking]
            }).reset_index(drop=True)

        elif method_name == "Topsis-dyskretny":
            ranking, score = topsis(alternatives_data, weights, directions)
            results_df = pd.DataFrame({
                "Nr alternatywy": self.alternatives_df["Nr alternatywy"].iloc[ranking],
                "Nazwa alternatywy": self.alternatives_df["Nazwa alternatywy"].iloc[ranking],
                "Wynik": score[ranking]
            }).reset_index(drop=True)

        elif method_name == "RSM-dyskretny":
            ranking, scores = discrete_reference_set_method(alternatives_data, directions, a, b)
            results_df = pd.DataFrame({
                "Nr alternatywy": self.alternatives_df["Nr alternatywy"].iloc[ranking],
                "Nazwa alternatywy": self.alternatives_df["Nazwa alternatywy"].iloc[ranking],
                "Wynik": scores[ranking]
            }).reset_index(drop=True)

        elif method_name == "UTAstar-dyskretny":
            # Dla metody UTAstar_discrete również używamy reference_points
            best_idx, utility_values = UTAstar_discrete(alternatives_data, reference_points)
            scores = utility_values.sum(axis=1)
            ranking = np.argsort(scores)[::-1]
            results_df = pd.DataFrame({
                "Nr alternatywy": self.alternatives_df["Nr alternatywy"].iloc[ranking],
                "Nazwa alternatywy": self.alternatives_df["Nazwa alternatywy"].iloc[ranking],
                "Wynik": scores[ranking]
            }).reset_index(drop=True)

        else:
            messagebox.showerror("Błąd", "Nieznana metoda.")
            return

        # Wyświetlamy wyniki w tabeli
        self.show_data_in_tree(self.results_tree, results_df)

        # Tworzymy wykres
        self.draw_plot(alternatives_data, results_df, criteria_columns)

    def draw_plot(self, alternatives_data, results_df, criteria_columns):
        self.figure.clear()
        num_criteria = alternatives_data.shape[1]
        # Najlepszy punkt to pierwszy wiersz w results_df
        best_nr = results_df["Nr alternatywy"].iloc[0]
        best_row = self.alternatives_df[self.alternatives_df["Nr alternatywy"] == best_nr]
        best_values = best_row[criteria_columns].to_numpy().flatten()

        if num_criteria == 2:
            ax = self.figure.add_subplot(111)
            x = alternatives_data[:,0]
            y = alternatives_data[:,1]
            ax.scatter(x, y, c='blue', label='Alternatywy')
            ax.scatter(best_values[0], best_values[1], c='red', marker='*', s=200, label='Najlepsza alternatywa')
            ax.set_xlabel(criteria_columns[0])
            ax.set_ylabel(criteria_columns[1])
            ax.legend()
        else:
            ax = self.figure.add_subplot(111, projection='3d')
            x = alternatives_data[:,0]
            y = alternatives_data[:,1]
            z = alternatives_data[:,2] if num_criteria > 2 else np.zeros_like(x)
            ax.scatter(x, y, z, c='blue', label='Alternatywy')
            bx = best_values[0]
            by = best_values[1]
            bz = best_values[2] if num_criteria > 2 else 0
            ax.scatter(bx, by, bz, c='red', marker='*', s=200, label='Najlepsza alternatywa')
            ax.set_xlabel(criteria_columns[0])
            ax.set_ylabel(criteria_columns[1])
            if num_criteria > 2:
                ax.set_zlabel(criteria_columns[2])
            ax.legend()

        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
