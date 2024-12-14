import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import pandas as pd
import numpy as np
from methods import (topsis, discrete_reference_set_method, UTAstar_discrete,
                     continuous_reference_set_method, UTAstar_continuous, fuzzy_topsis)

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

# Definicje funkcji celu dla wersji ciągłej
def f1(x):
    return x[0]**2 + (x[1]+0.2)**2
def f2(x):
    return (x[0]-1)**2 + (x[1]+1)**2
def f3(x):
    return (x[0]-3)**2 + (x[1]+3)**2
def f4(x):
    return (x[0]-2)**2 + (x[1]-1)**2

objective_functions = [f1, f2, f3, f4]

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
        
        method_frame = tk.Frame(master)
        method_frame.pack(pady=5)
        
        tk.Label(method_frame, text="Wybierz metodę:").pack(side=tk.LEFT, padx=5)
        self.method_combobox = ttk.Combobox(method_frame, values=self.methods, textvariable=self.method_var, state='readonly')
        self.method_combobox.set(self.methods[0])
        self.method_combobox.pack(side=tk.LEFT, padx=5)

        data_frame = tk.Frame(master)
        data_frame.pack(pady=5)

        # Przyciski do wczytania danych ciągłych i dyskretnych
        tk.Button(data_frame, text="Wczytaj dane ciągłe (dane_ciagle.xlsx)", command=self.load_continuous_data).pack(side=tk.LEFT, padx=5)
        tk.Button(data_frame, text="Wczytaj dane dyskretne (dane_dyskretne.xlsx)", command=self.load_discrete_data).pack(side=tk.LEFT, padx=5)

        tables_frame = tk.Frame(master)
        tables_frame.pack(pady=5)

        # Tabela alternatyw
        alt_frame = tk.Frame(tables_frame)
        alt_frame.pack(side=tk.LEFT, padx=5)

        alt_scrollbar_y = ttk.Scrollbar(alt_frame, orient="vertical")
        alt_scrollbar_x = ttk.Scrollbar(alt_frame, orient="horizontal")
        self.alternatives_tree = ttk.Treeview(alt_frame, show='headings', 
                                              yscrollcommand=alt_scrollbar_y.set,
                                              xscrollcommand=alt_scrollbar_x.set)
        alt_scrollbar_y.config(command=self.alternatives_tree.yview)
        alt_scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        alt_scrollbar_x.config(command=self.alternatives_tree.xview)
        alt_scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.alternatives_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Tabela punktów odniesienia
        ref_frame = tk.Frame(tables_frame)
        ref_frame.pack(side=tk.LEFT, padx=5)

        ref_scrollbar_y = ttk.Scrollbar(ref_frame, orient="vertical")
        ref_scrollbar_x = ttk.Scrollbar(ref_frame, orient="horizontal")
        self.reference_tree = ttk.Treeview(ref_frame, show='headings',
                                            yscrollcommand=ref_scrollbar_y.set,
                                            xscrollcommand=ref_scrollbar_x.set)
        ref_scrollbar_y.config(command=self.reference_tree.yview)
        ref_scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        ref_scrollbar_x.config(command=self.reference_tree.xview)
        ref_scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.reference_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Tabela wyników
        res_frame = tk.Frame(tables_frame)
        res_frame.pack(side=tk.LEFT, padx=5)

        res_scrollbar_y = ttk.Scrollbar(res_frame, orient="vertical")
        res_scrollbar_x = ttk.Scrollbar(res_frame, orient="horizontal")
        self.results_tree = ttk.Treeview(res_frame, show='headings',
                                         yscrollcommand=res_scrollbar_y.set,
                                         xscrollcommand=res_scrollbar_x.set)
        res_scrollbar_y.config(command=self.results_tree.yview)
        res_scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        res_scrollbar_x.config(command=self.results_tree.xview)
        res_scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        action_frame = tk.Frame(master)
        action_frame.pack(pady=5)
        tk.Button(action_frame, text="Uruchom metodę", command=self.run_method).pack(side=tk.LEFT, padx=5)
        
        self.plot_frame = tk.Frame(master)
        self.plot_frame.pack(pady=5, fill=tk.BOTH, expand=True)
        
        self.figure = Figure(figsize=(5, 3), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Zmienne do przechowywania danych
        self.alternatives_df_cont = None
        self.reference_df_cont = None
        self.alternatives_df_disc = None
        self.reference_df_disc = None

    def load_continuous_data(self):
        file_path = filedialog.askopenfilename(title="Wybierz plik dane_ciagle.xlsx", filetypes=[("Excel files","*.xlsx")])
        if file_path:
            self.alternatives_df_cont = pd.read_excel(file_path, sheet_name='Alternatywy')
            self.reference_df_cont = pd.read_excel(file_path, sheet_name='ReferencePoints')
            self.show_data_in_tree(self.alternatives_tree, self.alternatives_df_cont)
            self.show_data_in_tree(self.reference_tree, self.reference_df_cont)

    def load_discrete_data(self):
        file_path = filedialog.askopenfilename(title="Wybierz plik dane_dyskretne.xlsx", filetypes=[("Excel files","*.xlsx")])
        if file_path:
            self.alternatives_df_disc = pd.read_excel(file_path, sheet_name='Alternatywy')
            self.reference_df_disc = pd.read_excel(file_path, sheet_name='ReferencePoints')
            self.show_data_in_tree(self.alternatives_tree, self.alternatives_df_disc)
            self.show_data_in_tree(self.reference_tree, self.reference_df_disc)

    def show_data_in_tree(self, tree, df):
        tree.delete(*tree.get_children())
        tree["columns"] = list(df.columns)
        for col in df.columns:
            tree.heading(col, text=col)
            tree.column(col, anchor=tk.CENTER, width=60)
        
        for index, row in df.iterrows():
            tree.insert("", "end", values=list(row))

    def run_method(self):
        method_name = self.method_var.get()

        # Sprawdzamy czy metoda jest ciągła czy dyskretna
        if "ciągły" in method_name: 
            # Metoda ciągła - korzystamy z danych ciągłych
            if self.alternatives_df_cont is None or self.reference_df_cont is None:
                messagebox.showerror("Błąd", "Nie wczytano danych ciągłych.")
                return
            # values to punkty decyzyjne (x1, x2, ...)
            decision_vars = self.alternatives_df_cont.to_numpy(dtype=float)
            # Reference points to wartości kryteriów (Kryterium 1, Kryterium 2, ...)
            # Zakładamy, że reference_points to k x n
            criteria_cols_cont = [c for c in self.reference_df_cont.columns if "Kryterium" in c]
            reference_points = self.reference_df_cont[criteria_cols_cont].to_numpy()

            # Obliczamy wartości funkcji celu dla decision_vars
            # decision_vars: m x d
            m = decision_vars.shape[0]
            n = len(objective_functions)
            values = np.zeros((m, n))
            for i in range(m):
                x = decision_vars[i]
                values[i,:] = [f(x) for f in objective_functions]

            # kierunki, wagi, itp.
            directions = np.ones(n) # max
            weights = np.ones(n)/n
            criteria_types = ["max"]*n

            if method_name == "Fuzzy Topsis-ciągły":
                # Przykładowo brak rozmycia: [v,v,v]
                # Można dodać delta jeśli potrzeba
                decision_matrix = np.stack([values, values, values], axis=-1)
                scores = fuzzy_topsis(decision_matrix, weights, criteria_types)
                ranking = np.argsort(scores)[::-1]
                results_df = pd.DataFrame({
                    "x1": decision_vars[:,0],
                    "x2": decision_vars[:,1],
                    "Wynik": scores
                }).iloc[ranking].reset_index(drop=True)

            elif method_name == "RSM-ciągły":
                # continuous_reference_set_method wymaga a, b itp.
                a = reference_points.min(axis=0)
                b = reference_points.max(axis=0)
                # bounds - zakładamy jakieś, np. x1 i x2 w przedziale [-5,5]
                bounds = [(-5,5), (-5,5)] # do dopasowania
                result, optimal_f_values, optimal_score = continuous_reference_set_method(
                    objective_functions, directions, a, b, bounds
                )
                # Zwracamy jedno optymalne rozwiązanie
                results_df = pd.DataFrame({
                    "x1": [result.x[0]],
                    "x2": [result.x[1]],
                    "Wynik": [optimal_score]
                })

            elif method_name == "UTAstar-ciągły":
                # UTAstar_continuous(values, reference_points)
                best_idx, utility_values = UTAstar_continuous(values, reference_points)
                scores = utility_values.sum(axis=1)
                ranking = np.argsort(scores)[::-1]
                results_df = pd.DataFrame({
                    "x1": decision_vars[:,0],
                    "x2": decision_vars[:,1],
                    "Wynik": scores
                }).iloc[ranking].reset_index(drop=True)

            else:
                messagebox.showerror("Błąd", "Nieznana metoda ciągła.")
                return

        else:
            # Metody dyskretne
            if self.alternatives_df_disc is None or self.reference_df_disc is None:
                messagebox.showerror("Błąd", "Nie wczytano danych dyskretnych.")
                return

            crit_cols_disc = [c for c in self.alternatives_df_disc.columns if c not in ["Nr alternatywy", "Nazwa alternatywy"]]
            alternatives_data = self.alternatives_df_disc[crit_cols_disc].to_numpy(dtype=float)
            a = self.reference_df_disc[crit_cols_disc].min(axis=0).to_numpy()
            b = self.reference_df_disc[crit_cols_disc].max(axis=0).to_numpy()
            directions = np.ones(len(crit_cols_disc))
            weights = np.ones(len(crit_cols_disc))/len(crit_cols_disc)
            criteria_types = ["max"]*len(crit_cols_disc)

            if method_name == "Topsis-dyskretny":
                ranking, score = topsis(alternatives_data, weights, directions)
                results_df = pd.DataFrame({
                    "Nr alternatywy": self.alternatives_df_disc["Nr alternatywy"].iloc[ranking],
                    "Nazwa alternatywy": self.alternatives_df_disc["Nazwa alternatywy"].iloc[ranking],
                    "Wynik": score[ranking]
                }).reset_index(drop=True)

            elif method_name == "RSM-dyskretny":
                ranking, scores = discrete_reference_set_method(alternatives_data, directions, a, b)
                results_df = pd.DataFrame({
                    "Nr alternatywy": self.alternatives_df_disc["Nr alternatywy"].iloc[ranking],
                    "Nazwa alternatywy": self.alternatives_df_disc["Nazwa alternatywy"].iloc[ranking],
                    "Wynik": scores[ranking]
                }).reset_index(drop=True)

            elif method_name == "UTAstar-dyskretny":
                reference_points = self.reference_df_disc[crit_cols_disc].to_numpy()
                best_idx, utility_values = UTAstar_discrete(alternatives_data, reference_points)
                scores = utility_values.sum(axis=1)
                ranking = np.argsort(scores)[::-1]
                results_df = pd.DataFrame({
                    "Nr alternatywy": self.alternatives_df_disc["Nr alternatywy"].iloc[ranking],
                    "Nazwa alternatywy": self.alternatives_df_disc["Nazwa alternatywy"].iloc[ranking],
                    "Wynik": scores[ranking]
                }).reset_index(drop=True)

            else:
                messagebox.showerror("Błąd", "Nieznana metoda dyskretna.")
                return

        self.show_data_in_tree(self.results_tree, results_df)
        self.draw_plot(method_name, results_df)

    def draw_plot(self, method_name, results_df):
        self.figure.clear()
        # W przypadku ciągłych metod mamy x1, x2
        # W przypadku dyskretnych może być wiele kryteriów; tutaj pokazujemy przykładowy wykres dla 2D lub 3D
        # Dla uproszczenia: jeżeli mamy x1, x2 - zróbmy wykres 2D
        # Jeżeli dyskretne, a kryteriów 2 lub 3 - też możemy przedstawić
        
        if "ciągły" in method_name:
            # Ciągły: mamy x1, x2
            if "x1" in results_df.columns and "x2" in results_df.columns:
                x = results_df["x1"].to_numpy()
                y = results_df["x2"].to_numpy()
                ax = self.figure.add_subplot(111)
                ax.scatter(x, y, c='blue', label='Punkty')
                # Najlepszy punkt to pierwszy
                ax.scatter(x[0], y[0], c='red', marker='*', s=200, label='Najlepszy')
                ax.set_xlabel("x1")
                ax.set_ylabel("x2")
                ax.legend()
        else:
            # Dyskretny
            # Sprawdźmy ile kryteriów było
            # Tu już zależy od założeń. Jeśli w dyskretnych nie chcemy rysować, pominiemy.
            pass

        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
