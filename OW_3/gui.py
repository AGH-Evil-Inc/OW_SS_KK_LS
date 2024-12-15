import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import metody  # Import your methods module

class ModernGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Modern Optimization GUI")
        self.file_path = None
        self.sheet1_data = None
        self.sheet2_data = None

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

        # Result display
        result_frame = tk.Frame(left_frame, padx=10, pady=10)
        result_frame.pack(fill="both", expand=True)

        tk.Button(result_frame, text="Run Optimization", command=self.run_optimization).pack()

        tk.Label(result_frame, text="Results:").pack()
        self.result_text = tk.Text(result_frame, height=10, width=80, wrap="none")
        self.result_text.pack()

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

        if method == "Fuzzy Topsis":
            messagebox.showinfo("Info", "Fuzzy Topsis method is not yet implemented.")
            return

        try:
            if method == "RSM - ciągła":
                # Run RSM continuous optimization (placeholder)
                ranking, scores, _ = metody.continuous_reference_set_method(
                    self.sheet1_data.values, [1] * self.sheet1_data.shape[1],
                    self.sheet1_data.min().values, self.sheet1_data.max().values
                )

            elif method == "RSM - dyskretna":
                # Run RSM discrete optimization (placeholder)
                ranking, scores = metody.discrete_reference_set_method(
                    self.sheet1_data.values, [1] * self.sheet1_data.shape[1],
                    self.sheet1_data.min().values, self.sheet1_data.max().values
                )

            elif method == "Topsis":
                # Run TOPSIS optimization (placeholder)
                ranking, scores = metody.topsis(
                    self.sheet1_data.values, [1] * self.sheet1_data.shape[1], [1] * self.sheet1_data.shape[1]
                )

            elif method == "UTA - dyskretna":
                # Generate reference points and run UTA discrete (placeholder)
                ref_points = np.vstack([
                    self.sheet1_data.min().values,
                    self.sheet1_data.median().values,
                    self.sheet1_data.max().values
                ])

                best_idx, _ = metody.UTAstar_discrete(
                    self.sheet1_data.values, ref_points
                )
                ranking = [best_idx]

            elif method == "UTA - ciągła":
                # Generate reference points and run UTA continuous (placeholder)
                ref_points = np.vstack([
                    self.sheet1_data.min().values,
                    self.sheet1_data.median().values,
                    self.sheet1_data.max().values
                ])

                best_solution, _, _ = metody.UTAstar_continuous(
                    self.sheet1_data.values, ref_points
                )
                ranking = [best_solution]

            else:
                raise NotImplementedError("Method not implemented.")

            # Display results
            self.result_text.delete("1.0", tk.END)
            result_df = pd.DataFrame({
                "miejsce": list(range(1, len(ranking) + 1)),
                "Id": ranking,
                "punkt": [self.sheet1_data.iloc[rank].tolist() for rank in ranking],
                "wynik": [scores[rank] for rank in ranking]
            })
            from pandastable import Table
            table_frame = tk.Frame(self.visualization_frame)
            table_frame.pack(side="left", fill="both", expand=True)
            pt = Table(table_frame, dataframe=result_df, showstatusbar=True)
            pt.show()

            # Generate visualization
            self.visualize(ranking)

        except Exception as e:
            messagebox.showerror("Error", f"Optimization failed: {e}")

    def visualize(self, ranking):
        for widget in self.visualization_frame.winfo_children():
            widget.destroy()

        fig = Figure(figsize=(6, 4))

        if self.sheet1_data.shape[1] == 2:  # 2D Visualization
            ax = fig.add_subplot(111)
            ax.scatter(self.sheet1_data.iloc[:, 0], self.sheet1_data.iloc[:, 1], label="Alternatives")
            best = self.sheet1_data.iloc[ranking[0]]
            ax.scatter(best[0], best[1], color="red", label="Best Solution")
            ax.set_title("2D Visualization")
            ax.legend()

        elif self.sheet1_data.shape[1] == 3:  # 3D Visualization
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(self.sheet1_data.iloc[:, 0], self.sheet1_data.iloc[:, 1], self.sheet1_data.iloc[:, 2], label="Alternatives")
            best = self.sheet1_data.iloc[ranking[0]]
            ax.scatter(best.iloc[0], best.iloc[1], best.iloc[2], color="red", label="Best Solution")
            ax.set_title("3D Visualization")
            ax.legend()

        else:  # Higher dimensions
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(self.sheet1_data.iloc[:, 0], self.sheet1_data.iloc[:, 1], self.sheet1_data.iloc[:, 2], label="Alternatives")
            best = self.sheet1_data.iloc[ranking[0]]
            ax.scatter(best[0], best[1], best[2], color="red", label="Best Solution")
            ax.set_title("3D Visualization (First 3 Criteria)")
            ax.legend()

        canvas = FigureCanvasTkAgg(fig, master=self.visualization_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side="right", fill="both", expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = ModernGUI(root)
    root.mainloop()
