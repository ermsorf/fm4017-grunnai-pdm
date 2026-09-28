from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


DATA_FOLDER = Path(__file__).parent.parent / "data" / "2024-06-01_2024-10-31"


def load_signal(file_path: Path) -> pd.Series:
	"""Read this project's CSV format: identifier, unit, then timestamp/value rows."""
	raw = pd.read_csv(file_path, header=None, skiprows=2, names=["time", "value"])
	raw["time"] = pd.to_datetime(raw["time"], errors="coerce")
	raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
	raw = raw.dropna(subset=["time", "value"]).set_index("time")
	return raw["value"].rename(file_path.stem)


class SignalViewer(tk.Tk):
	def __init__(self) -> None:
		super().__init__()
		self.title("CSV Signal Visualizer")
		self.geometry("1100x700")
		self.signals: dict[str, pd.Series] = {}

		top = ttk.Frame(self, padding=10)
		top.pack(fill="x")
		ttk.Label(top, text=f"Folder: {DATA_FOLDER.name}").pack(side="left")
		ttk.Button(top, text="Load signals", command=self.load_signals).pack(side="right")
		ttk.Button(top, text="Plot selected", command=self.plot_selected).pack(
			side="right", padx=8
		)

		body = ttk.Frame(self, padding=(10, 0, 10, 10))
		body.pack(fill="both", expand=True)
		self.listbox = tk.Listbox(body, selectmode=tk.MULTIPLE, width=55, exportselection=False)
		self.listbox.pack(side="left", fill="y")

		self.figure, self.axes = plt.subplots(figsize=(9, 5))
		self.canvas = FigureCanvasTkAgg(self.figure, master=body)
		self.canvas.get_tk_widget().pack(side="right", fill="both", expand=True, padx=(10, 0))

		self.load_signals()

	def load_signals(self) -> None:
		files = sorted(DATA_FOLDER.glob("*.csv"))
		if not files:
			messagebox.showerror("No CSV files", f"No CSV files found in:\n{DATA_FOLDER}")
			return

		self.signals = {}
		self.listbox.delete(0, tk.END)
		for file_path in files:
			try:
				signal = load_signal(file_path)
				self.signals[file_path.name] = signal
				self.listbox.insert(tk.END, file_path.name)
			except Exception as error:
				print(f"Skipped {file_path.name}: {error}")

		if self.signals:
			self.listbox.selection_set(0, min(2, len(self.signals) - 1))
			self.plot_selected()

	def plot_selected(self) -> None:
		selected = [self.listbox.get(i) for i in self.listbox.curselection()]
		if not selected:
			messagebox.showinfo("Select signals", "Select one or more CSV files first.")
			return

		self.axes.clear()
		for file_name in selected:
			signal = self.signals[file_name]
			self.axes.plot(signal.index, signal.values, linewidth=1, label=file_name)
		self.axes.set_title("Selected signals")
		self.axes.set_xlabel("Time")
		self.axes.set_ylabel("Value")
		self.axes.grid(True, alpha=0.3)
		self.axes.legend(fontsize="small")
		self.figure.autofmt_xdate()
		self.canvas.draw()


if __name__ == "__main__":
	SignalViewer().mainloop()
