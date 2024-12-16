import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import metody  # Import your methods module


class ModernGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Modern Optimization GUI")
        self.file_path = None
        self.sheet1_data = None
        self.sheet2_data = None
        self.canvas = None  # Canvas for the visualization

        self.create_widgets()

    def create_widgets(self):
        # Main container
        main_frame = tk.Frame(self.root, padx=10, pady=10)
        main_frame.pack(fill="both", expand=True)

        # Left frame for data and results
        left_frame = tk.Frame(main_frame)
        left_frame.pack(side="left", fill="both", expand=True)

        # Right frame for visualization
        self.visualization_frame = tk.Frame(main_frame)
        self.visualization_frame.pack(side="right", fill="both", expand=True)

        # File selection
        file_frame = tk.Frame(left_frame, padx=10, pady=10)
        file_frame.pack(fill="x")

        tk.Button(file_frame, text="Select Data File", command=self.load_file).pack(side="left")
        self.file_label = tk.Label(file_frame, text="No file selected")
        self.file_label.pack(side="left", padx=10)

        # Method selection
        method_frame = tk.Frame(left_frame, padx=10, pady=10)
        method_frame.pack(fill="x")

        tk.Label(method_frame, text="Select Method:").pack(side="left")
        self.method_var = tk.StringVar()
        self.method_var.set("RSM - dyskretna")
        methods = ["RSM - dyskretna", "RSM - ciągła", "UTA - dyskretna", "UTA - ciągła", "Topsis", "Fuzzy Topsis"]
        self.method_menu = tk.OptionMenu(method_frame, self.method_var, *methods)
        self.method_menu.pack(side="left", padx=10)

        # Data display
        data_frame = tk.Frame(left_frame, padx=10, pady=10)
        data_frame.pack(fill="both", expand=True)

        tk.Label(data_frame, text="Sheet1 Data:").pack()
        self.sheet1_text = tk.Text(data_frame, height=10, width=80, wrap="none")
        self.sheet1_text.pack()

        tk.Label(data_frame, text="Sheet2 Data:").pack()
        self.sheet2_text = tk.Text(data_frame, height=5, width=80, wrap="none")
        self.sheet2_text.pack()

        # Tabela rankingu
        frame_rank = tk.LabelFrame(left_frame, text="Stworzony ranking", padx=10, pady=10)
        frame_rank.pack(fill="both", expand=True)

        # Treeview dla wyników
        self.tree_rank = ttk.Treeview(frame_rank, columns=["Nr","Nr Punkt", "Punkt", "Wynik"], show="headings", height=8)
        self.tree_rank.heading("Nr", text="Ranking")
        self.tree_rank.heading("Nr Punkt", text="Nr Punktu")
        self.tree_rank.heading("Punkt", text="Punkt Alternatywy")
        self.tree_rank.heading("Wynik", text="Wynik")
        self.tree_rank.pack(fill="both", expand=True, padx=5, pady=5)

        # Przyciski sterujące
        button_frame = tk.Frame(left_frame, padx=10, pady=10)
        button_frame.pack(fill="x")

        tk.Button(button_frame, text="Run Optimization", command=self.run_optimization).pack(side="left", padx=5)

    def load_file(self):
        self.file_path = filedialog.askopenfilename(title="Select Excel File", filetypes=[("Excel files", "*.xlsx")])
        if not self.file_path:
            return

        self.file_label.config(text=self.file_path)

        # Load the data
        try:
            self.sheet1_data = pd.read_excel(self.file_path, sheet_name="Sheet1")
            self.sheet2_data = pd.read_excel(self.file_path, sheet_name="Sheet2")

            # Display data
            self.sheet1_text.delete("1.0", tk.END)
            self.sheet1_text.insert(tk.END, self.sheet1_data.to_string())

            self.sheet2_text.delete("1.0", tk.END)
            self.sheet2_text.insert(tk.END, self.sheet2_data.to_string())

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {e}")

    def run_optimization(self):
        if self.sheet1_data is None or self.sheet2_data is None or self.sheet1_data.empty or self.sheet2_data.empty:
            messagebox.showerror("Error", "No data loaded!")
            return

        method = self.method_var.get()


        alternatives = np.array(self.sheet1_data.iloc[:, 1:].values)
        typ_row = self.sheet2_data.loc[self.sheet2_data['Unnamed: 0'] == 'typ'].iloc[0, 1:]  # Pomijamy kolumnę 'Unnamed: 0'
        waga_row = self.sheet2_data.loc[self.sheet2_data['Unnamed: 0'] == 'waga'].iloc[0, 1:]  # Pomijamy kolumnę 'Unnamed: 0'
        criteria_types = np.where(typ_row == 'max', 1, -1)
        a = []
        b = []
        for col in typ_row.index:
            if typ_row[col] == 'min':
                a.append(self.sheet1_data[col].min())  # Minimalna wartość w kolumnie
                b.append(self.sheet1_data[col].max())  # Maksymalna wartość w kolumnie
            elif typ_row[col] == 'max':
                a.append(self.sheet1_data[col].max())  # Maksymalna wartość w kolumnie
                b.append(self.sheet1_data[col].min())  # Minimalna wartość w kolumnie
        a = np.array(a)
        b = np.array(b)

        try:
            if method == "RSM - ciągła":
                ranking, scores, all_points = metody.continuous_reference_set_method(
                    alternatives, criteria_types, a, b)
                alternatives = all_points

            elif method == "RSM - dyskretna":
                ranking, scores = metody.discrete_reference_set_method(
                    alternatives, criteria_types, a, b)

            elif method == "Topsis":
                decision_matrix = alternatives
                weights = list(map(float, waga_row.values))
                ranking, scores = metody.topsis(
                    alternatives, weights, criteria_types)

            elif method == "Fuzzy Topsis":
                def prepare_matrix_for_fuzzy_topsis(alternatives_vector, lambda_param=1):
                    n, m = alternatives_vector.shape
                    fuzzy_matrix = np.zeros((n, m, 3))
                    for i in range(n):
                        for j in range(m):
                            fuzzy_value = (alternatives_vector[i, j] - lambda_param, alternatives_vector[i, j], alternatives_vector[i, j] + lambda_param)
                            fuzzy_matrix[i, j] = fuzzy_value
                    return fuzzy_matrix
                weights = list(map(float, waga_row.values))
                temp_alternatives = prepare_matrix_for_fuzzy_topsis(alternatives, 1.5)
                scores, ranking = metody.fuzzy_topsis(temp_alternatives, weights, criteria_types)

            elif method == "UTA - dyskretna":
                values = alternatives
                median = np.array(self.sheet1_data.iloc[:, 1:].median().values)
                reference_points = np.array([b, median, a])
                _, _, scores = metody.UTAstar_discrete(
                    values, reference_points)
                ranking = np.argsort(scores)[::-1]
                alternatives = values
            
            elif method == "UTA - ciągła":
                values = alternatives
                median = np.array(self.sheet1_data.iloc[:, 1:].median().values)
                reference_points = np.array([b, median, a])
                best_sol, best_sol_score, scores, _ = metody.UTAstar_continuous(
                    values, reference_points)
                alternatives = np.append(alternatives, [best_sol], axis=0)
                scores = np.append(-scores, best_sol_score)
                ranking = np.argsort(scores)[::-1]

            else:
                raise NotImplementedError("Method not implemented.")

            # Wyświetlanie wyników w tabeli
            self.display_results(ranking, scores, alternatives)

            # Wyświetlanie wykresu
            self.visualize(ranking, scores, alternatives)

        except Exception as e:
            messagebox.showerror("Error", f"Optimization failed: {e}")

    def display_results(self, ranking, scores, points):
        # Czyszczenie poprzednich wyników
        for i in self.tree_rank.get_children():
            self.tree_rank.delete(i)

        # Dodawanie wyników do tabeli
        for idx, (r, s, point) in enumerate(zip(ranking, scores, points)):
            punkt = ', '.join(map(str, point))
            nr = ', '.join(map(str, [r]))
            self.tree_rank.insert("", "end", values=[idx + 1, nr , points[r], scores[r]])

    def visualize(self, ranking, scores, alternatives):
        """
        Visualizes the alternatives and highlights the best solution.

        Parameters:
        - ranking: array-like, the ranking of the alternatives.
        - scores: array-like, the scores of the alternatives.
        - alternatives: ndarray, the alternative solutions (n x m).
        """
        # Remove any existing canvas
        if self.canvas:
            self.canvas.get_tk_widget().destroy()

        # Create a new figure for the plot
        fig = Figure(figsize=(6, 4))

        if alternatives.shape[1] == 2:  # 2D Visualization
            ax = fig.add_subplot(111)
            x = alternatives[:, 0]
            y = alternatives[:, 1]

            # Scatter plot for all alternatives
            ax.scatter(x, y, label="Alternatives", color="blue")

            # Highlight the best solution
            best_index = ranking[0]
            ax.scatter(x[best_index], y[best_index], color="red", label="Best Solution", s=100, edgecolors="black")

            ax.set_title("2D Visualization")
            ax.set_xlabel("Criterion 1")
            ax.set_ylabel("Criterion 2")
            ax.legend()

        elif alternatives.shape[1] >= 3:  # 3D Visualization
            ax = fig.add_subplot(111, projection='3d')
            x = alternatives[:, 0]
            y = alternatives[:, 1]
            z = alternatives[:, 2]

            # Scatter plot for all alternatives
            ax.scatter(x, y, z, label="Alternatives", color="blue")

            # Highlight the best solution
            best_index = ranking[0]
            ax.scatter(x[best_index], y[best_index], z[best_index], color="red", label="Best Solution", s=100, edgecolors="black")

            ax.set_title("3D Visualization")
            ax.set_xlabel("Criterion 1")
            ax.set_ylabel("Criterion 2")
            ax.set_zlabel("Criterion 3")
            ax.legend()

        # Create a canvas and display the plot
        self.canvas = FigureCanvasTkAgg(fig, master=self.visualization_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)


if __name__ == "__main__":
    root = tk.Tk()
    app = ModernGUI(root)
    root.mainloop()
