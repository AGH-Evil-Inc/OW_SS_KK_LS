import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # jeśli potrzebujemy wykres 3D
import os

# Import z tego samego folderu
from methods import topsis  # na przykład importujemy jedną z metod
# W razie potrzeby importuj inne metody z methods:
# from methods import continuous_reference_set_method, discrete_reference_set_method, UTAstar_discrete, UTAstar_continuous, fuzzy_topsis

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
        self.method_box = ttk.Combobox(frame_top, textvariable=self.method_var, values=["TOPSIS", "CRSM", "UTA", "Fuzzy TOPSIS"])
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
        # Wczytujemy dane z dwóch plików: alternatyw i klas (można zmodyfikować wg potrzeb)
        alt_file = filedialog.askopenfilename(title="Wybierz plik XLSX z alternatywami", filetypes=[("Excel files", "*.xlsx")])
        if not alt_file:
            return
        cls_file = filedialog.askopenfilename(title="Wybierz plik XLSX z klasami", filetypes=[("Excel files", "*.xlsx")])
        if not cls_file:
            return
        
        self.alternatives_df = pd.read_excel(alt_file, engine='openpyxl')
        self.classes_df = pd.read_excel(cls_file, engine='openpyxl')
        
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
        
        decision_matrix = self.alternatives_df[["Kryterium 1","Kryterium 2","Kryterium 3"]].values
        # Przykład: zakładamy wszystkie kryteria do maksymalizacji
        directions = np.array([1,1,1])  
        # Równe wagi
        weights = np.array([1/3, 1/3, 1/3])
        
        if method == "TOPSIS":
            ranking, score = topsis(decision_matrix, weights, directions)
        else:
            # W tym przykładzie obsługujemy tylko TOPSIS. 
            # Można rozszerzyć o if method == "CRSM": ... itd.
            messagebox.showwarning("Uwaga", "Inne metody nie są jeszcze zaimplementowane w GUI.")
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
            # Jeśli jest inna liczba kryteriów, np. wyświetlamy tylko 2D z pierwszych 2 kryteriów
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
