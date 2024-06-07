import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import math
import csv
import pandas as pd
import numpy as np
from tkinter import Radiobutton, Checkbutton, Label, IntVar, StringVar, Frame
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from fitter import Fitter
import matplotlib.pyplot as plt
import scipy.stats as stats
from smote import SMOTE
import anderson_library  # Import the custom Anderson-Darling library

class GridDataEntryApp:
    def __init__(self, master, default_rows=100, default_columns=1):
        self.master = master
        self.default_rows = default_rows
        self.default_columns = default_columns

        self.menu_bar = tk.Menu(self.master)
        self.master.config(menu=self.menu_bar)
        self.create_menu()

        self.create_data_section()

        self.notebook = ttk.Notebook(self.master)
        self.notebook.pack(side="left", fill="both", expand=True)

        self.tabs = {}

        self.create_main_tab("Main")

    def create_menu(self):
        self.menu_bar.add_cascade(label="New", command=self.new_project)
        self.menu_bar.add_cascade(label="Open", command=self.open_project)
        self.menu_bar.add_cascade(label="Save", command=self.save_project)
        self.menu_bar.add_cascade(label="Run", command=self.run_project)
        self.menu_bar.add_cascade(label="Exit", command=self.master.quit)
        self.menu_bar.add_cascade(label="Help", command=self.help_project)

    def create_data_section(self):
        self.data_frame = Frame(self.master, borderwidth=4, relief="groove")
        self.data_frame.pack(side="left", fill="y", padx=20, pady=20)

        self.data_type_var = StringVar()
        self.dist_vars = []

        Label(self.data_frame, text="Data Type:").grid(row=0, column=0, sticky="w")
        Radiobutton(self.data_frame, text="Discrete", variable=self.data_type_var, value="Discrete",
                    command=self.update_distribution).grid(row=1, column=0, sticky="w")
        Radiobutton(self.data_frame, text="Continuous", variable=self.data_type_var, value="Continuous",
                    command=self.update_distribution).grid(row=2, column=0, sticky="w")

        Label(self.data_frame, text="Data Processing Method:").grid(row=3, column=0, sticky="w")
        self.data_processing_var = StringVar(value="Original Data")
        self.data_processing_dropdown = ttk.Combobox(self.data_frame, textvariable=self.data_processing_var,
                                                     values=["Original Data", "Bootstrap", "SMOTE"])
        self.data_processing_dropdown.grid(row=4, column=0, sticky="w")
        self.data_processing_var.trace("w", self.update_data_processing)

        self.additional_option_label = Label(self.data_frame, text="Number of Samples:")
        self.additional_option_entry = tk.Entry(self.data_frame)
        self.additional_option_label.grid(row=5, column=0, sticky="w")
        self.additional_option_entry.grid(row=6, column=0, sticky="w")
        self.additional_option_label.grid_remove()
        self.additional_option_entry.grid_remove()

        self.k_neighbors_label = Label(self.data_frame, text="Number of k-neighbors:")
        self.k_neighbors_entry = tk.Entry(self.data_frame)
        self.k_neighbors_label.grid(row=7, column=0, sticky="w")
        self.k_neighbors_entry.grid(row=8, column=0, sticky="w")
        self.k_neighbors_label.grid_remove()
        self.k_neighbors_entry.grid_remove()

        self.dist_label = Label(self.data_frame, text="Distributions:")
        self.dist_label.grid(row=9, column=0, sticky="w")

        self.select_all_var = IntVar()
        self.select_all_button = Checkbutton(self.data_frame, text="Select All", variable=self.select_all_var, command=self.toggle_select_all)
        self.select_all_button.grid(row=10, column=0, sticky="w")

        self.data_type_var.set("Discrete")
        self.update_distribution()

    def toggle_select_all(self):
        """
        Toggle selection of all distributions.
        """
        new_value = 1 if self.select_all_var.get() == 1 else 0
        for dist in self.dist_vars:
            dist["var"].set(new_value)

    def update_distribution(self):
        for var in self.dist_vars:
            var['widget'].destroy()

        self.dist_vars.clear()

        if self.data_type_var.get() == "Discrete":
            options = ["Poisson", "Binomial", "Bernoulli"]
        else:
            options = ["norm", "uniform", "expon", "gamma", "beta", "lognorm", "weibull_min", "weibull_max"]

        row_offset = 11  # Adjusted to account for the "Select All" button
        for idx, option in enumerate(options):
            var = IntVar()
            chk = Checkbutton(self.data_frame, text=option, variable=var)
            chk.grid(row=row_offset + idx, column=0, sticky="w")
            self.dist_vars.append({"widget": chk, "var": var, "name": option})

    def update_data_processing(self, *args):
        method = self.data_processing_var.get()
        if method == "Bootstrap":
            self.additional_option_label.grid()
            self.additional_option_entry.grid()
            self.k_neighbors_label.grid_remove()
            self.k_neighbors_entry.grid_remove()
        elif method == "SMOTE":
            self.additional_option_label.grid()
            self.additional_option_entry.grid()
            self.k_neighbors_label.grid()
            self.k_neighbors_entry.grid()
        else:
            self.additional_option_label.grid_remove()
            self.additional_option_entry.grid_remove()
            self.k_neighbors_label.grid_remove()
            self.k_neighbors_entry.grid_remove()

    def create_grid(self, frame):
        self.entries = []
        for i in range(self.default_rows):
            for j in range(self.default_columns):
                entry = tk.Entry(frame, width=10)
                entry.grid(row=i, column=j)
                self.entries.append(entry)

    def close_all_tabs(self):
        """
        Close all tabs except the main tab.
        """
        for tab in list(self.tabs.keys()):
            if tab == 'Main':
                pass
            else:
                self.notebook.forget(self.tabs[tab])
        self.tabs.clear()

    def new_project(self):
        """
        Clear all data entries and close all tabs for a new project.
        """
        self.close_all_tabs()
        for entry in self.entries:
            entry.delete(0, tk.END)

    def open_project(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("CSV Files", "*.csv"), ("Excel Files", "*.xlsx"), ("All Files", "*.*")])
        if file_path:
            if file_path.endswith('.csv'):
                with open(file_path, 'r') as file:
                    self.data = pd.read_csv(file)
            elif file_path.endswith('.xlsx'):
                self.data = pd.read_excel(file_path)
            self.update_grid(self.data.values)

    def update_grid(self, data):
        for entry in self.entries:
            entry.destroy()

        self.entries.clear()

        for i, row in enumerate(data):
            for j, value in enumerate(row):
                entry = tk.Entry(self.frame, width=10)
                entry.grid(row=i, column=j)
                entry.insert(tk.END, value)
                self.entries.append(entry)

        self.frame.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def save_project(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".csv",
                                                 filetypes=[("CSV Files", "*.csv"), ("Excel Files", "*.xlsx"),
                                                            ("All Files", "*.*")])
        if file_path:
            data = []
            for row in range(self.default_rows):
                current_row = []
                for col in range(self.default_columns):
                    entry = self.entries[row * self.default_columns + col]
                    current_row.append(entry.get())
                data.append(current_row)
            if file_path.endswith('.csv'):
                with open(file_path, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerows(data)
            elif file_path.endswith('.xlsx'):
                df = pd.DataFrame(data)
                df.to_excel(file_path, index=False)

    def run_project(self):

        tab_name = f"Analysis {math.ceil(len(self.tabs)/2) }"

        self.create_new_tab(tab_name)

        # Get the processed data
        processed_data = self.get_processed_data()

        # Create statistics and plots using the processed data
        self.create_statistics(tab_name, processed_data)
        self.create_plots(tab_name, processed_data)

        best_fit_tab_name = f"Best_fit {math.ceil(len(self.tabs)/2)}"
        self.create_best_fit_tab(best_fit_tab_name, processed_data)

        # messagebox.showinfo("Run", f"Running the project! Check the '{tab_name}' tab for results.")

    def help_project(self):
        messagebox.showinfo("Help", "Contact support at support@example.com")

    def on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def on_frame_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def create_main_tab(self, tab_name):
        tab_frame = tk.Frame(self.notebook)
        self.notebook.add(tab_frame, text=tab_name)
        self.tabs[tab_name] = tab_frame

        plot_frame = tk.Frame(tab_frame)
        plot_frame.pack(side="left", fill="both", expand=True)

        self.canvas = tk.Canvas(plot_frame)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.frame = tk.Frame(self.canvas)

        self.vsb = tk.Scrollbar(plot_frame, orient="vertical", command=self.canvas.yview)
        self.hsb = tk.Scrollbar(plot_frame, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=self.vsb.set, xscrollcommand=self.hsb.set)
        self.vsb.pack(side="right", fill="y")
        self.hsb.pack(side="bottom", fill="x")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.create_window((0, 0), window=self.frame, anchor="nw")
        self.frame.bind("<Configure>", self.on_frame_configure)
        self.canvas.bind_all("<MouseWheel>", self.on_mousewheel)
        self.create_grid(self.frame)

    def create_new_tab(self, tab_name, main_tab=False):
        tab_frame = tk.Frame(self.notebook)
        self.notebook.add(tab_frame, text=tab_name)
        self.tabs[tab_name] = tab_frame

    def create_best_fit_tab(self, tab_name, data):
        tab_frame = tk.Frame(self.notebook)
        self.notebook.add(tab_frame, text=tab_name)
        self.tabs[tab_name] = tab_frame

        selected_distributions = [dist["name"] for dist in self.dist_vars if dist["var"].get() == 1]

        if not selected_distributions:
            messagebox.showwarning("No Distribution Selected", "Please select at least one distribution.")
            return

        data = data.select_dtypes(include=[np.number]).values.flatten()

        f = Fitter(data, distributions=selected_distributions)
        f.fit()
        best_distribution = f.get_best(method='sumsquare_error')
        best_fit_dist = list(best_distribution.keys())[0]

        right_frame = tk.Frame(tab_frame, width=600)
        right_frame.pack(side="left", fill="y")
        right_frame.pack_propagate(False)

        left_frame = tk.Frame(tab_frame)
        left_frame.pack(side="right", fill="both", expand=True)

        self.fig = Figure(figsize=(5, 5))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=left_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.parameters_frame = tk.Frame(left_frame)
        self.parameters_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

        def update_plot(dist_name):
            self.ax.clear()
            stats.probplot(data, dist=dist_name, sparams=f.fitted_param[dist_name], plot=self.ax)
            self.ax.set_title(f'Q-Q plot for {dist_name}')
            self.canvas.draw()
            update_parameters(dist_name, f.fitted_param[dist_name])

        distribution_parameters = {
            "norm": ["mean", "std"],
            "uniform": ["min", "max"],
            "expon": ["scale", "loc"],
            "gamma": ["a", "loc", "scale"],
            "beta": ["a", "b", "loc", "scale"],
            "lognorm": ["s", "loc", "scale"],
            "weibull_min": ["c", "loc", "scale"],
            "weibull_max": ["c", "loc", "scale"]
        }

        def update_parameters(dist_name, params):
            for widget in self.parameters_frame.winfo_children():
                widget.destroy()
            param_names = distribution_parameters.get(dist_name, [f"Param {i + 1}" for i in range(len(params))])
            tk.Label(self.parameters_frame, text=f"Parameters for {dist_name}:").pack(anchor="w")
            for param_name, param_value in zip(param_names, params):
                tk.Label(self.parameters_frame, text=f"{param_name}: {param_value:.4f}").pack(anchor="w")

        columns = ("Distribution", "KS Statistic", "p-value", "sumsquare_error", "Anderson-Darling Statistic", "Significance Level")
        tree = ttk.Treeview(right_frame, columns=columns, show="headings")
        tree.pack(fill=tk.BOTH, expand=True)

        for col in columns:
            tree.heading(col, text=col)

        def add_anderson_darling_result(dist_name, data):
            if dist_name in ["poisson", "binomial", "bernoulli"]:
                return "N/A", "N/A", "N/A"  # Skip AD test for discrete distributions
            ad_stat, ad_crit_vals, ad_sig_lvls = anderson_library.anderson_darling_test(data, dist=dist_name)
            return ad_stat, ad_crit_vals, ad_sig_lvls

        for dist_name in selected_distributions:
            ad_stat, ad_crit_vals, ad_sig_lvls = add_anderson_darling_result(dist_name, data)
            tree.insert("", "end",
                        values=(dist_name,f"{f._ks_stat[dist_name]:.6f}", f"{f._ks_pval[dist_name]:.6f}", f"{f._fitted_errors[dist_name]:.6f}", ad_stat, f'{ad_sig_lvls[2]:.0f}'))

        def on_tree_select(event):
            selected_item = tree.selection()[0]
            selected_dist = tree.item(selected_item, "values")[0]
            update_plot(selected_dist)

        tree.bind("<<TreeviewSelect>>", on_tree_select)

        def treeview_sort_column(tv, col, reverse):
            l = [(tv.set(k, col), k) for k in tv.get_children('')]
            l.sort(reverse=reverse)

            for index, (val, k) in enumerate(l):
                tv.move(k, '', index)

            tv.heading(col, command=lambda: treeview_sort_column(tv, col, not reverse))

        column_widths = {"Distribution": 60, "KS Statistic": 80, "p-value": 80, "sumsquare_error": 80, "Anderson-Darling Statistic": 80,  "Significance Level": 80}
        for col, width in column_widths.items():
            tree.heading(col, text=col, command=lambda c=col: treeview_sort_column(tree, c, False))
            tree.column(col, width=width)

        update_plot(best_fit_dist)

    def create_plots(self, tab_name, data):
        data = data.select_dtypes(include=[np.number])
        tab_frame = self.tabs[tab_name]

        fig = Figure(figsize=(10, 5))

        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        ax1.boxplot(data.values.flatten(), vert=True)
        ax1.set_title('Box Plot')

        ax2.hist(data.values.flatten(), bins=30, color='blue', alpha=0.7)
        ax2.set_title('Histogram')

        canvas = FigureCanvasTkAgg(fig, master=self.tabs[tab_name])
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def create_statistics(self, tab_name, data):
        data = data.select_dtypes(include=[np.number])
        tab_frame = self.tabs[tab_name]

        stats_frame = tk.Frame(tab_frame, width=2000)
        stats_frame.pack(side="left", fill="x", padx=10, pady=10)

        num_data_points = len(data)
        mean = data.mean().mean()
        median = data.median().median()
        variance = data.var().mean()
        std_dev = data.std().mean()
        min_value = data.min().min()
        max_value = data.max().max()

        stats = {
            "Number of Data Points": num_data_points,
            "Maximum": max_value,
            "Minimum": min_value,
            "Mean": mean,
            "Median": median,
            "Variance": variance,
            "Standard Deviation": std_dev
        }

        for stat_name, stat_value in stats.items():
            label = tk.Label(stats_frame, text=f"{stat_name}: {stat_value:.2f}")
            label.pack(anchor="w")

    def bootstrap(self, data, n_samples):
        # Randomly select n_samples with replacement from the data
        bootstrap_samples = data.sample(n=n_samples, replace=True)
        # Append the bootstrap samples to the original data
        new_data = pd.concat([data, bootstrap_samples], ignore_index=True)
        return new_data

    def smote_(self, data, n_samples):
        k_neighbors = int(self.k_neighbors_entry.get())
        sm = SMOTE(k_neighbors=k_neighbors, random_state=None)
        sm.fit(data)
        X_resampled = sm.sample(n_samples)
        return X_resampled

    def get_processed_data(self):
        method = self.data_processing_var.get()
        if method == "Original Data":
            return self.data
        elif method == "Bootstrap":
            n_samples = int(self.additional_option_entry.get())
            return self.bootstrap(self.data, n_samples)
        elif method == "SMOTE":
            n_samples = int(self.additional_option_entry.get())
            return self.smote_(self.data, n_samples)


def main():
    root = tk.Tk()
    root.title("Grid Data Entry Application")
    app = GridDataEntryApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
