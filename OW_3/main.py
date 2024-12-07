import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

# Importujemy wszystkie potrzebne metody z methods.py
from methods import (topsis, discrete_reference_set_method, UTAstar_discrete, 
                     continuous_reference_set_method, UTAstar_continuous, fuzzy_topsis)

class App:
    def __init__(self, master):
        self.master = master
        self.master.title("GUI_OW")
        
        # Zmienne do przechowywania danych
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
        frame_tables.pack(pady=10)
        
        # Tabela alternatyw
        frame_alt = tk.LabelFrame(frame_tables, text="Alternatywy z kryteriami")
        frame_alt.pack(side=tk.LEFT, padx=10)
        
        self.tree_alt = ttk.Treeview(frame_alt, columns=["Nr","Nazwa","C1","C2","C3"], show="headings", height=8)
        self.tree_alt.heading("Nr", text="Nr")
        self.tree_alt.heading("Nazwa", text="Nazwa")
        self.tree_alt.heading("C1", text="Kryterium 1")
        self.tree_alt.heading("C2", text="Kryterium 2")
        self.tree_alt.heading("C3", text="Kryterium 3")
        self.tree_alt.pack(padx=5, pady=5)
        
        # Tabela klas
        frame_cls = tk.LabelFrame(frame_tables, text="Klasy")
        frame_cls.pack(side=tk.LEFT, padx=10)
        
        self.tree_cls = ttk.Treeview(frame_cls, columns=["Nr","x","y","z"], show="headings", height=8)
        self.tree_cls.heading("Nr", text="Nr klasy")
        self.tree_cls.heading("x", text="x")
        self.tree_cls.heading("y", text="y")
        self.tree_cls.heading("z", text="z")
        self.tree_cls.pack(padx=5, pady=5)
        
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
        
        # Wyświetlenie w tabelkach
        for i in self.tree_alt.get_children():
            self.tree_alt.delete(i)
        for i, row in self.alternatives_df.iterrows():
            self.tree_alt.insert("", "end", values=[row["Nr alternatywy"], row["Nazwa alternatywy"], row["Kryterium 1"], row["Kryterium 2"], row["Kryterium 3"]])
        
        for i in self.tree_cls.get_children():
            self.tree_cls.delete(i)
        for i, row in self.classes_df.iterrows():
            self.tree_cls.insert("", "end", values=[row["Nr klasy"], row["x"], row["y"], row["z"]])
        
        messagebox.showinfo("Informacja", "Pobrano dane")
        
    def create_ranking(self):
        if self.alternatives_df is None or self.classes_df is None:
            messagebox.showerror("Błąd", "Najpierw wczytaj dane!")
            return
        
        method = self.method_var.get()
        if method == "Wybierz metodę...":
            messagebox.showerror("Błąd", "Wybierz metodę!")
            return
        
        # Konwersja przecinków na kropki jeśli istnieją
        self.alternatives_df[["Kryterium 1","Kryterium 2","Kryterium 3"]] = self.alternatives_df[["Kryterium 1","Kryterium 2","Kryterium 3"]].replace(',', '.', regex=True).astype(float)
        self.classes_df[["x","y","z"]] = self.classes_df[["x","y","z"]].replace(',', '.', regex=True).astype(float)
        
        decision_matrix = self.alternatives_df[["Kryterium 1","Kryterium 2","Kryterium 3"]].values
        directions = np.array([1,1,1])  # Maksymalizacja wszystkich
        weights = np.array([1/3, 1/3, 1/3])
        
        # Punkty referencyjne a i b
        if len(self.classes_df) >= 2:
            a = self.classes_df.iloc[0][["x","y","z"]].values
            b = self.classes_df.iloc[1][["x","y","z"]].values
        else:
            # Jeśli jest tylko 1 klasa, tworzymy sztuczne a i b
            a = np.min(decision_matrix, axis=0)
            b = np.max(decision_matrix, axis=0)
        
        if method == "TOPSIS":
            ranking, score = topsis(decision_matrix, weights, directions)
        elif method == "CRSM (dyskr.)":
            ranking, score = discrete_reference_set_method(decision_matrix, directions, a, b)
        elif method == "CRSM (ciąg.)":
            messagebox.showinfo("Informacja", "CRSM ciągła wymaga zdefiniowania problemu ciągłego. Brak implementacji przykładu.")
            return
        elif method == "UTA (dyskretna)":
            ref_point_neutral = (a+b)/2
            reference_points = np.array([a, b, ref_point_neutral])
            best_idx, utility_values = UTAstar_discrete(decision_matrix, reference_points, lambda_param=1)
            ranking = np.argsort(-np.sum(utility_values, axis=1))
            score = np.sum(utility_values, axis=1)
        elif method == "UTA (ciągła)":
            messagebox.showinfo("Informacja", "UTA ciągła wymaga problemu ciągłego. Brak implementacji przykładu.")
            return
        elif method == "Fuzzy TOPSIS":
            messagebox.showinfo("Informacja", "Fuzzy TOPSIS wymaga danych rozmytych. Brak implementacji przykładu.")
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
        if num_criteria == 2:
            ax = self.figure.add_subplot(111)
            x = decision_matrix[:,0]
            y = decision_matrix[:,1]
            scatter = ax.scatter(x, y, c=score, cmap='viridis', s=50)
            ax.set_xlabel("Kryterium 1")
            ax.set_ylabel("Kryterium 2")
            self.figure.colorbar(scatter, ax=ax, label="Wynik")
        elif num_criteria == 3:
            ax = self.figure.add_subplot(111, projection='3d')
            x = decision_matrix[:,0]
            y = decision_matrix[:,1]
            z = decision_matrix[:,2]
            scatter = ax.scatter(x, y, z, c=score, cmap='viridis', s=50)
            ax.set_xlabel("Kryterium 1")
            ax.set_ylabel("Kryterium 2")
            ax.set_zlabel("Kryterium 3")
            self.figure.colorbar(scatter, ax=ax, label="Wynik")
        else:
            ax = self.figure.add_subplot(111)
            x = decision_matrix[:,0]
            y = decision_matrix[:,1]
            scatter = ax.scatter(x, y, c=score, cmap='viridis', s=50)
            ax.set_xlabel("Kryterium 1")
            ax.set_ylabel("Kryterium 2")
            self.figure.colorbar(scatter, ax=ax, label="Wynik")
        
        self.canvas.draw()
        
        messagebox.showinfo("Informacja", "Ranking utworzony!")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
