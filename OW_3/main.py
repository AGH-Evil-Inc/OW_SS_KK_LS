import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

from methods import (topsis, discrete_reference_set_method, UTAstar_discrete, 
                     continuous_reference_set_method, UTAstar_continuous, fuzzy_topsis)

class App:
    def __init__(self, master):
        self.master = master
        self.master.title("GUI_OW")
        
        self.alternatives_df = None
        self.classes_df = None
        
        # Ramka z przyciskami
        frame_top = tk.Frame(master)
        frame_top.pack(pady=10)
        
        self.btn_load_data = tk.Button(frame_top, text="Wczytaj dane z pliku", command=self.load_data)
        self.btn_load_data.pack(side=tk.LEFT, padx=5)
        
        # Combobox do wyboru metody
        self.method_var = tk.StringVar()
        self.method_box = ttk.Combobox(frame_top, textvariable=self.method_var, 
                                       values=["TOPSIS", "CRSM (dyskr.)", "CRSM (ciąg.)", "UTA (dyskretna)", "UTA (ciągła)", "Fuzzy TOPSIS"])
        self.method_box.set("Wybierz metodę...")
        self.method_box.pack(side=tk.LEFT, padx=5)
        
        self.btn_create_ranking = tk.Button(frame_top, text="Stwórz ranking", command=self.create_ranking)
        self.btn_create_ranking.pack(side=tk.LEFT, padx=5)
        
        # Ramka na tabelki
        frame_tables = tk.Frame(master)
        frame_tables.pack(pady=10, expand=True, fill='both')
        
        # Sekcja Alternatywy
        self.frame_alt = tk.LabelFrame(frame_tables, text="Alternatywy z kryteriami")
        self.frame_alt.pack(side=tk.LEFT, padx=10, expand=True, fill='both')
        
        self.frame_alt.config(width=400, height=200)
        self.frame_alt.pack_propagate(False)
        
        self.tree_alt = None
        
        # Sekcja Klasy
        self.frame_cls = tk.LabelFrame(frame_tables, text="Klasy")
        self.frame_cls.pack(side=tk.LEFT, padx=10, expand=True, fill='both')
        
        self.frame_cls.config(width=400, height=200)
        self.frame_cls.pack_propagate(False)

        self.tree_cls = ttk.Treeview(self.frame_cls, columns=["Nr","x","y","z"], show="headings", height=8)
        self.tree_cls.heading("Nr", text="Nr klasy")
        self.tree_cls.heading("x", text="x")
        self.tree_cls.heading("y", text="y")
        self.tree_cls.heading("z", text="z")

        self.tree_cls.column("Nr", width=60, anchor='center')  
        self.tree_cls.column("x", width=80, anchor='center')
        self.tree_cls.column("y", width=80, anchor='center')
        self.tree_cls.column("z", width=80, anchor='center')

        self.tree_cls.pack(padx=5, pady=5, expand=True, fill='both')
        
        # Tabela rankingu
        frame_rank = tk.LabelFrame(master, text="Stworzony ranking")
        frame_rank.pack(pady=10)
        
        self.tree_rank = ttk.Treeview(frame_rank, columns=["Nr","Score"], show="headings", height=8)
        self.tree_rank.heading("Nr", text="Nr Alternatywy")
        self.tree_rank.heading("Score", text="Wynik")
        self.tree_rank.pack(padx=5, pady=5)
        
        # Ramka na wykres
        self.frame_plot = tk.Frame(master)
        self.frame_plot.pack(pady=10)
        
        self.figure = Figure(figsize=(5,4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.frame_plot)
        self.plot_widget = self.canvas.get_tk_widget()
        self.plot_widget.pack()
        
    def load_data(self):
        alt_file = filedialog.askopenfilename(title="Wybierz plik XLSX z alternatywami", filetypes=[("Excel files", "*.xlsx")])
        if not alt_file:
            return
        cls_file = filedialog.askopenfilename(title="Wybierz plik XLSX z klasami", filetypes=[("Excel files", "*.xlsx")])
        if not cls_file:
            return
        
        self.alternatives_df = pd.read_excel(alt_file, engine='openpyxl')
        self.classes_df = pd.read_excel(cls_file, engine='openpyxl')

        if self.tree_alt is not None:
            self.tree_alt.destroy()
        
        # Dynamiczna konfiguracja tree_alt
        all_columns = self.alternatives_df.columns.tolist()
        self.tree_alt = ttk.Treeview(self.frame_alt, columns=all_columns, show='headings', height=8)

        for col in all_columns:
            self.tree_alt.heading(col, text=col)

        self.tree_alt.column(all_columns[0], width=60, anchor='center')
        self.tree_alt.column(all_columns[1], width=100, anchor='w')
        for c in all_columns[2:]:
            self.tree_alt.column(c, width=70, anchor='center')

        for i, row in self.alternatives_df.iterrows():
            values = row.tolist()
            self.tree_alt.insert("", "end", values=values)

        self.tree_alt.pack(padx=5, pady=5, expand=True, fill='both')
        
        # Wyświetlenie klas
        for i in self.tree_cls.get_children():
            self.tree_cls.delete(i)
        for i, row in self.classes_df.iterrows():
            self.tree_cls.insert("", "end", values=[row["Nr klasy"], row["x"], row["y"], row["z"]])
        
        messagebox.showinfo("Informacja", "Pobrano dane")
        
    def create_ranking(self):
        method = self.method_var.get()
        if method == "Wybierz metodę...":
            messagebox.showerror("Błąd", "Wybierz metodę!")
            return
        
        # Jeśli metoda nie jest ciągła, potrzebujemy alternatyw i klas
        if method != "CRSM (ciąg.)":
            if self.alternatives_df is None or self.classes_df is None:
                messagebox.showerror("Błąd", "Najpierw wczytaj dane!")
                return
            
        if method == "CRSM (ciąg.)":
            # Wczytujemy plik z danymi ciągłymi
            cont_file = filedialog.askopenfilename(title="Wybierz plik XLSX z danymi ciągłymi", filetypes=[("Excel files", "*.xlsx")])
            if not cont_file:
                return
            cont_df = pd.read_excel(cont_file, engine='openpyxl', header=None)
            # Zakładamy, że w pierwszej kolumnie mamy nazwy parametrów, w drugiej wartości
            cont_df.columns = ["Parametr","Wartosc"]
            cont_data = dict(zip(cont_df["Parametr"], cont_df["Wartosc"]))

            # Wyciągamy dane
            x1_min = float(cont_data["Bounds_x1_min"])
            x1_max = float(cont_data["Bounds_x1_max"])
            x2_min = float(cont_data["Bounds_x2_min"])
            x2_max = float(cont_data["Bounds_x2_max"])

            a11 = float(cont_data["a11"])
            a12 = float(cont_data["a12"])
            a13 = float(cont_data["a13"])
            a21 = float(cont_data["a21"])
            a22 = float(cont_data["a22"])
            a23 = float(cont_data["a23"])

            directions_str = str(cont_data["directions"])
            directions = np.array([int(d.strip()) for d in directions_str.split(',')])


            a_points_str = cont_data["a_points"]
            a_points = np.array([float(x.strip()) for x in a_points_str.split(',')])

            b_points_str = cont_data["b_points"]
            b_points = np.array([float(x.strip()) for x in b_points_str.split(',')])

            # Definiujemy funkcje celu
            def f1(x):
                x1, x2 = x
                return a11*x1**2 + a12*x2 + a13

            def f2(x):
                x1, x2 = x
                return a21*x1 + a22*x2**2 + a23

            objective_functions = [f1, f2]
            bounds = [(x1_min, x1_max), (x2_min, x2_max)]

            # Wywołanie CRSM ciągłego
            result, optimal_f_values, optimal_score = continuous_reference_set_method(
                objective_functions, directions, a_points, b_points, bounds
            )

            # Wyświetlamy wynik w rankingu - w przypadku problemu ciągłego nie mamy alternatyw dyskretnych,
            # ale możemy wyświetlić znalezione optimum jako jedną "alternatywę"
            for i in self.tree_rank.get_children():
                self.tree_rank.delete(i)

            self.tree_rank.insert("", "end", values=["Optimum", round(float(optimal_score),4)])
            
            # Wizualizacja 2D (jeśli 2 funkcje celu)
            # Uwaga: mamy tu problem, bo continuous_reference_set_method nie daje nam zestawu punktów,
            # a jedynie optimum. Możemy jedynie zaznaczyć optimum na wykresie jako punkt.
            self.figure.clear()
            if len(objective_functions) == 2:
                # Narysujmy siatkę punktów w dziedzinie, by zobrazować funkcje celu lub wynik
                # Ponieważ mamy 2 f-ce, możemy zarysować mapę punktów. Jednak to skomplikowane.
                # Prostszy wariant: pokażmy tylko punkt optimum na pustym wykresie.

                ax = self.figure.add_subplot(111)
                # Punkty docelowe a, b i opt
                ax.scatter(a_points[0], a_points[1], s=100, facecolors='none', edgecolors='g', linewidth=2, label='Aspiracja (a)')
                ax.scatter(b_points[0], b_points[1], s=100, facecolors='none', edgecolors='b', linewidth=2, label='Status Quo (b)')
                ax.scatter(optimal_f_values[0], optimal_f_values[1], s=200, facecolors='none', edgecolors='r', linewidth=2, label='Optimum')
                ax.set_xlabel("f1")
                ax.set_ylabel("f2")
                ax.legend()
            else:
                # Więcej niż 2 funkcje - brak wizualizacji (lub inny pomysł)
                ax = self.figure.add_subplot(111)
                ax.text(0.5, 0.5, "Optimum znalezione:\n f* = {}".format(optimal_f_values), ha='center')
            
            self.canvas.draw()

            messagebox.showinfo("Informacja", "Optimum znalezione dla problemu ciągłego!")
            return

        # Pozostałe metody dyskretne
        # Konwersja do float
        criteria_cols = self.alternatives_df.columns[2:]
        for c in criteria_cols:
            self.alternatives_df[c] = self.alternatives_df[c].replace(',', '.', regex=True).astype(float)

        for c in ["x","y","z"]:
            self.classes_df[c] = self.classes_df[c].replace(',', '.', regex=True).astype(float)
        
        decision_matrix = self.alternatives_df[criteria_cols].values
        directions = np.ones(len(criteria_cols))
        weights = np.ones(len(criteria_cols)) / len(criteria_cols)
        
        if len(self.classes_df) >= 2:
            a = self.classes_df.iloc[0][["x","y","z"]].values
            b = self.classes_df.iloc[1][["x","y","z"]].values
        else:
            a = np.min(decision_matrix, axis=0)
            b = np.max(decision_matrix, axis=0)
            if len(criteria_cols) > 3:
                # Jeśli więcej kryteriów niż 3, a i b zdefiniujmy jako min/max
                a = np.min(decision_matrix, axis=0)
                b = np.max(decision_matrix, axis=0)
        
        if method == "TOPSIS":
            ranking, score = topsis(decision_matrix, weights, directions)
        elif method == "CRSM (dyskr.)":
            ranking, score = discrete_reference_set_method(decision_matrix, directions, a, b)
        elif method == "UTA (dyskretna)":
            ref_point_neutral = (a+b)/2
            reference_points = np.array([a, b, ref_point_neutral])
            best_idx, utility_values = UTAstar_discrete(decision_matrix, reference_points, lambda_param=1)
            ranking = np.argsort(-np.sum(utility_values, axis=1))
            score = np.sum(utility_values, axis=1)
        elif method == "UTA (ciągła)":
            messagebox.showinfo("Informacja", "UTA ciągła wymaga problemu ciągłego.")
            return
        elif method == "Fuzzy TOPSIS":
            messagebox.showinfo("Informacja", "Fuzzy TOPSIS wymaga danych rozmytych.")
            return
        else:
            messagebox.showerror("Błąd", "Metoda nieobsługiwana.")
            return

        # Wyświetlenie rankingu
        for i in self.tree_rank.get_children():
            self.tree_rank.delete(i)
        
        for r, s in zip(ranking, score[ranking]):
            alt_nr = self.alternatives_df.iloc[r]["Nr alternatywy"]
            self.tree_rank.insert("", "end", values=[alt_nr, round(float(s),4)])
        
        # Rysowanie wykresu
        self.figure.clear()
        num_criteria = decision_matrix.shape[1]
        best_idx = ranking[0]

        if num_criteria == 2:
            ax = self.figure.add_subplot(111)
            x = decision_matrix[:,0]
            y = decision_matrix[:,1]
            scatter = ax.scatter(x, y, c=score, cmap='viridis', s=50)
            ax.scatter(x[best_idx], y[best_idx], s=200, facecolors='none', edgecolors='r', linewidth=2)
            ax.set_xlabel(criteria_cols[0])
            ax.set_ylabel(criteria_cols[1])
            self.figure.colorbar(scatter, ax=ax, label="Wynik")
        elif num_criteria == 3:
            ax = self.figure.add_subplot(111, projection='3d')
            x = decision_matrix[:,0]
            y = decision_matrix[:,1]
            z = decision_matrix[:,2]
            scatter = ax.scatter(x, y, z, c=score, cmap='viridis', s=50)
            ax.scatter(x[best_idx], y[best_idx], z[best_idx], s=200, facecolors='none', edgecolors='r', linewidth=2)
            ax.set_xlabel(criteria_cols[0])
            ax.set_ylabel(criteria_cols[1])
            ax.set_zlabel(criteria_cols[2])
            self.figure.colorbar(scatter, ax=ax, label="Wynik")
        else:
            ax = self.figure.add_subplot(111)
            x = decision_matrix[:,0]
            y = decision_matrix[:,1]
            scatter = ax.scatter(x, y, c=score, cmap='viridis', s=50)
            ax.scatter(x[best_idx], y[best_idx], s=200, facecolors='none', edgecolors='r', linewidth=2)
            ax.set_xlabel(criteria_cols[0])
            ax.set_ylabel(criteria_cols[1])
            self.figure.colorbar(scatter, ax=ax, label="Wynik")
        
        self.canvas.draw()
        
        messagebox.showinfo("Informacja", "Ranking utworzony!")


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
