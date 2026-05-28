from __future__ import annotations

import json
import math
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageDraw, ImageFont, ImageTk

from fingerprint_processor import (
    FingerprintAnalysis,
    analyze_fingerprint,
    binary_to_gray,
    from_pillow,
    gray_to_pillow,
    overlay_minutiae,
)


class FingerprintThinningApp:
    PANEL_TITLES = [
        ("source", "Oryginal"),
        ("enhanced", "Wzmocnienie"),
        ("binary", "Binaryzacja"),
        ("cleaned", "Po oczyszczaniu"),
        ("morph_raw", "Szkielet morfologiczny"),
        ("morph_overlay", "Minucje morfologiczne"),
        ("k3m_raw", "Szkielet K3M"),
        ("k3m_overlay", "Minucje K3M"),
        ("kmm_raw", "Szkielet KMM"),
        ("kmm_overlay", "Minucje KMM"),
    ]

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Biometria P3 - Porownanie algorytmow scieniania")
        self.root.geometry("1560x940")
        self.root.minsize(1280, 820)
        self.root.state("zoomed")

        self.current_path: Path | None = None
        self.current_image = None
        self.analysis: FingerprintAnalysis | None = None
        self.pending_result: queue.Queue = queue.Queue()
        self.worker_thread: threading.Thread | None = None
        self.is_processing = False

        self.threshold_bias_var = tk.IntVar(value=0)
        self.close_size_var = tk.IntVar(value=1)
        self.open_size_var = tk.IntVar(value=1)
        self.bridge_iterations_var = tk.IntVar(value=1)
        self.spur_iterations_var = tk.IntVar(value=1)
        self.spur_length_var = tk.IntVar(value=4)
        self.bridge_gap_var = tk.IntVar(value=7)
        self.bridge_links_var = tk.IntVar(value=1)
        self.min_distance_var = tk.IntVar(value=6)
        self.show_endings_var = tk.BooleanVar(value=True)
        self.show_bifurcations_var = tk.BooleanVar(value=True)
        self.binarization_method_var = tk.StringVar(value="auto")
        self.invert_var = tk.BooleanVar(value=False)
        self.sauvola_window_var = tk.IntVar(value=21)
        self.sauvola_k_var = tk.DoubleVar(value=0.2)
        self.sauvola_range_var = tk.DoubleVar(value=128.0)
        self.mask_enabled_var = tk.BooleanVar(value=False)
        self.mask_window_var = tk.IntVar(value=17)
        self.mask_min_std_var = tk.DoubleVar(value=12.0)
        self.mask_kernel_var = tk.IntVar(value=9)
        self.sample_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Wczytaj odcisk palca i uruchom porownanie algorytmow.")
        self.summary_var = tk.StringVar(value="Brak wynikow.")
        self.morph_var = tk.StringVar(value="Morfologia: brak wynikow.")
        self.k3m_var = tk.StringVar(value="K3M: brak wynikow.")
        self.kmm_var = tk.StringVar(value="KMM: brak wynikow.")

        self.preview_refs: dict[str, ImageTk.PhotoImage] = {}
        self.labels: dict[str, ttk.Label] = {}
        self.demo_files = self._collect_demo_files()
        self._build_ui()

    def _collect_demo_files(self) -> list[str]:
        root = Path(__file__).resolve().parent / "skaner_odciskow" / "en" / "Demo"
        files = sorted(root.glob("*/*.bmp"))
        return [str(path) for path in files]

    def _build_ui(self) -> None:
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        controls_container = ttk.Frame(self.root, padding=12)
        controls_container.grid(row=0, column=0, sticky="nsw")
        controls_container.columnconfigure(0, weight=1)
        controls_container.rowconfigure(0, weight=1)

        self.controls_canvas = tk.Canvas(controls_container, width=340, highlightthickness=0)
        self.controls_canvas.grid(row=0, column=0, sticky="nsw")
        controls_scrollbar = ttk.Scrollbar(controls_container, orient="vertical", command=self.controls_canvas.yview)
        controls_scrollbar.grid(row=0, column=1, sticky="ns")
        self.controls_canvas.configure(yscrollcommand=controls_scrollbar.set)

        controls = ttk.Frame(self.controls_canvas)
        controls.columnconfigure(0, weight=1)
        self.controls_window = self.controls_canvas.create_window((0, 0), window=controls, anchor="nw")
        controls.bind("<Configure>", self._on_controls_configure)
        self.controls_canvas.bind("<Configure>", self._on_canvas_configure)
        self.controls_canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        workspace = ttk.Frame(self.root, padding=(0, 12, 12, 12))
        workspace.grid(row=0, column=1, sticky="nsew")
        workspace.columnconfigure(0, weight=1)
        workspace.rowconfigure(0, weight=1)

        self._build_controls(controls)
        self._build_workspace(workspace)

    def _on_controls_configure(self, _event: tk.Event) -> None:
        self.controls_canvas.configure(scrollregion=self.controls_canvas.bbox("all"))

    def _on_canvas_configure(self, event: tk.Event) -> None:
        self.controls_canvas.itemconfigure(self.controls_window, width=event.width)

    def _on_mousewheel(self, event: tk.Event) -> None:
        if self.controls_canvas.winfo_exists():
            self.controls_canvas.yview_scroll(int(-event.delta / 120), "units")

    def _build_controls(self, parent: ttk.Frame) -> None:
        file_frame = ttk.LabelFrame(parent, text="Pliki", padding=10)
        file_frame.grid(row=0, column=0, sticky="ew")
        file_frame.columnconfigure(0, weight=1)
        ttk.Button(file_frame, text="Wczytaj obraz", command=self.open_image).grid(row=0, column=0, sticky="ew")
        ttk.Button(file_frame, text="Analizuj obraz", command=self.run_analysis).grid(row=1, column=0, sticky="ew", pady=(6, 0))
        ttk.Button(file_frame, text="Eksportuj dashboard", command=self.export_dashboard).grid(row=2, column=0, sticky="ew", pady=(6, 0))
        ttk.Button(file_frame, text="Eksportuj wyniki JSON", command=self.export_results_json).grid(
            row=3, column=0, sticky="ew", pady=(6, 0)
        )

        demo_frame = ttk.LabelFrame(parent, text="Szybki wybor demo", padding=10)
        demo_frame.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        demo_frame.columnconfigure(0, weight=1)
        sample_names = [Path(path).parent.name + "/" + Path(path).name for path in self.demo_files]
        self.sample_combo = ttk.Combobox(demo_frame, textvariable=self.sample_var, values=sample_names, state="readonly")
        self.sample_combo.grid(row=0, column=0, sticky="ew")
        ttk.Button(demo_frame, text="Zaladuj z demo", command=self.load_demo_image).grid(row=1, column=0, sticky="ew", pady=(6, 0))

        params_frame = ttk.LabelFrame(parent, text="Parametry", padding=10)
        params_frame.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        params_frame.columnconfigure(0, weight=1)
        self._add_combo(params_frame, "Binarizacja", self.binarization_method_var, ["auto", "otsu", "sauvola"], 0)
        ttk.Checkbutton(params_frame, text="Invert", variable=self.invert_var).grid(row=1, column=0, sticky="w")
        self._add_entry(params_frame, "Bias progu", self.threshold_bias_var, 2)
        self._add_entry(params_frame, "Sauvola okno", self.sauvola_window_var, 3)
        self._add_entry(params_frame, "Sauvola k", self.sauvola_k_var, 4)
        self._add_entry(params_frame, "Sauvola R", self.sauvola_range_var, 5)
        ttk.Checkbutton(params_frame, text="Maska odcisku", variable=self.mask_enabled_var).grid(row=6, column=0, sticky="w")
        self._add_entry(params_frame, "Maska okno", self.mask_window_var, 7)
        self._add_entry(params_frame, "Maska min std", self.mask_min_std_var, 8)
        self._add_entry(params_frame, "Maska kernel", self.mask_kernel_var, 9)
        self._add_combo(params_frame, "Domkniecie", self.close_size_var, [1, 3, 5], 10)
        self._add_combo(params_frame, "Otwarcie", self.open_size_var, [1, 3, 5], 11)
        self._add_combo(params_frame, "Laczenie przerw", self.bridge_iterations_var, [0, 1, 2], 12)
        self._add_entry(params_frame, "Max przerwa", self.bridge_gap_var, 13)
        self._add_entry(params_frame, "Linki na punkt", self.bridge_links_var, 14)
        self._add_entry(params_frame, "Pruning odnog", self.spur_iterations_var, 15)
        self._add_entry(params_frame, "Dlugosc odnogi", self.spur_length_var, 16)
        self._add_entry(params_frame, "Min dystans", self.min_distance_var, 17)
        ttk.Checkbutton(
            params_frame,
            text="Pokaz zakonczenia",
            variable=self.show_endings_var,
            command=self._on_minutiae_filter_change,
        ).grid(row=18, column=0, sticky="w")
        ttk.Checkbutton(
            params_frame,
            text="Pokaz bifurkacje",
            variable=self.show_bifurcations_var,
            command=self._on_minutiae_filter_change,
        ).grid(row=19, column=0, sticky="w")

        info_frame = ttk.LabelFrame(parent, text="Wyniki", padding=10)
        info_frame.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        info_frame.columnconfigure(0, weight=1)
        ttk.Label(info_frame, textvariable=self.summary_var, wraplength=290, justify="left").grid(row=0, column=0, sticky="ew")
        ttk.Label(info_frame, textvariable=self.morph_var, wraplength=290, justify="left").grid(row=1, column=0, sticky="ew", pady=(8, 0))
        ttk.Label(info_frame, textvariable=self.k3m_var, wraplength=290, justify="left").grid(row=2, column=0, sticky="ew", pady=(8, 0))
        ttk.Label(info_frame, textvariable=self.kmm_var, wraplength=290, justify="left").grid(row=3, column=0, sticky="ew", pady=(8, 0))

        ttk.Label(parent, textvariable=self.status_var, wraplength=290, justify="left").grid(row=4, column=0, sticky="ew", pady=(12, 0))
        self.progress = ttk.Progressbar(parent, mode="indeterminate")
        self.progress.grid(row=5, column=0, sticky="ew", pady=(8, 0))

    def _add_entry(self, parent: ttk.Frame, label: str, variable: tk.Variable, row: int) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        ttk.Entry(parent, textvariable=variable, width=12).grid(row=row, column=1, sticky="e", padx=(8, 0), pady=(0, 4))

    def _add_combo(self, parent: ttk.Frame, label: str, variable: tk.Variable, values: list, row: int) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        ttk.Combobox(parent, textvariable=variable, values=values, state="readonly", width=10).grid(
            row=row, column=1, sticky="e", padx=(8, 0), pady=(0, 4)
        )

    def _build_workspace(self, parent: ttk.Frame) -> None:
        canvas = tk.Canvas(parent, highlightthickness=0)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        canvas.configure(yscrollcommand=scrollbar.set)

        frame = ttk.Frame(canvas)
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(2, weight=1)
        frame.columnconfigure(3, weight=1)
        self.workspace_window = canvas.create_window((0, 0), window=frame, anchor="nw")

        def on_frame_configure(_event: tk.Event) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        def on_canvas_configure(event: tk.Event) -> None:
            canvas.itemconfigure(self.workspace_window, width=event.width)

        frame.bind("<Configure>", on_frame_configure)
        canvas.bind("<Configure>", on_canvas_configure)
        canvas.bind_all("<Shift-MouseWheel>", lambda event: canvas.yview_scroll(int(-event.delta / 120), "units"))

        for index, (key, title) in enumerate(self.PANEL_TITLES):
            row = index // 4
            column = index % 4
            holder = ttk.LabelFrame(frame, text=title, padding=8)
            holder.grid(row=row, column=column, sticky="nsew", padx=8, pady=8)
            holder.columnconfigure(0, weight=1)
            holder.rowconfigure(0, weight=1)
            label = ttk.Label(holder, text="Brak obrazu", anchor="center")
            label.grid(row=0, column=0, sticky="nsew")
            self.labels[key] = label

    def open_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Wybierz obraz odcisku",
            filetypes=[("Bitmap", "*.bmp"), ("Obrazy", "*.png;*.jpg;*.jpeg;*.bmp"), ("Wszystkie pliki", "*.*")],
        )
        if not path:
            return
        self._load_path(Path(path))

    def load_demo_image(self) -> None:
        selected = self.sample_var.get().strip()
        if not selected:
            messagebox.showwarning("Brak wyboru", "Wybierz najpierw obraz z listy demo.")
            return
        for path in self.demo_files:
            demo_label = Path(path).parent.name + "/" + Path(path).name
            if demo_label == selected:
                self._load_path(Path(path))
                return

    def _load_path(self, path: Path) -> None:
        try:
            image = from_pillow(Image.open(path))
        except Exception as error:
            messagebox.showerror("Blad odczytu", f"Nie udalo sie otworzyc pliku.\n\n{error}")
            return

        self.current_path = path
        self.current_image = image
        self.analysis = None
        self.summary_var.set("Brak wynikow.")
        self.morph_var.set("Morfologia: brak wynikow.")
        self.k3m_var.set("K3M: brak wynikow.")
        self.kmm_var.set("KMM: brak wynikow.")
        self.status_var.set(f"Wczytano obraz: {path.name}")
        self._set_preview("source", gray_to_pillow(image), (300, 360))

    def run_analysis(self) -> None:
        if self.current_image is None:
            messagebox.showwarning("Brak obrazu", "Najpierw wczytaj obraz odcisku.")
            return
        if self.is_processing:
            return

        params = {
            "threshold_bias": int(self.threshold_bias_var.get()),
            "close_size": int(self.close_size_var.get()),
            "open_size": int(self.open_size_var.get()),
            "bridge_iterations": int(self.bridge_iterations_var.get()),
            "spur_iterations": int(self.spur_iterations_var.get()),
            "spur_length": int(self.spur_length_var.get()),
            "bridge_gap": int(self.bridge_gap_var.get()),
            "bridge_links_per_endpoint": int(self.bridge_links_var.get()),
            "binarization_method": self.binarization_method_var.get(),
            "invert": bool(self.invert_var.get()),
            "sauvola_window_size": int(self.sauvola_window_var.get()),
            "sauvola_k": float(self.sauvola_k_var.get()),
            "sauvola_dynamic_range": float(self.sauvola_range_var.get()),
            "mask_enabled": bool(self.mask_enabled_var.get()),
            "mask_window_size": int(self.mask_window_var.get()),
            "mask_min_std": float(self.mask_min_std_var.get()),
            "mask_kernel_size": int(self.mask_kernel_var.get()),
            "minutiae_min_distance": int(self.min_distance_var.get()),
        }
        self.is_processing = True
        self.progress.start(10)
        self.status_var.set("Analiza odcisku w toku...")

        def worker() -> None:
            try:
                result = analyze_fingerprint(self.current_image.copy(), **params)
                self.pending_result.put(("success", result))
            except Exception as error:
                self.pending_result.put(("error", error))

        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()
        self.root.after(50, self._poll_worker)

    def _poll_worker(self) -> None:
        try:
            message = self.pending_result.get_nowait()
        except queue.Empty:
            if self.is_processing:
                self.root.after(50, self._poll_worker)
            return

        self.progress.stop()
        self.is_processing = False

        if message[0] == "error":
            messagebox.showerror("Blad analizy", str(message[1]))
            self.status_var.set("Analiza nie powiodla sie.")
            return

        self.analysis = message[1]
        self._apply_minutiae_filters()
        self._render_analysis(self.analysis)
        self.summary_var.set(self.analysis.comparison_summary)
        self.morph_var.set(self.analysis.morph.summary)
        self.k3m_var.set(self.analysis.k3m.summary)
        self.kmm_var.set(self.analysis.kmm.summary)
        self.status_var.set("Zakonczono porownanie algorytmow.")

    def _render_analysis(self, analysis: FingerprintAnalysis) -> None:
        self._set_preview("source", gray_to_pillow(analysis.source), (300, 360))
        self._set_preview("enhanced", gray_to_pillow(analysis.enhanced), (300, 360))
        self._set_preview("binary", gray_to_pillow(binary_to_gray(analysis.binary)), (300, 360))
        self._set_preview("cleaned", gray_to_pillow(binary_to_gray(analysis.cleaned)), (300, 360))
        self._set_preview("morph_raw", gray_to_pillow(binary_to_gray(analysis.morph.skeleton_raw)), (300, 360))
        self._set_preview("morph_overlay", analysis.morph.overlay, (300, 360))
        self._set_preview("k3m_raw", gray_to_pillow(binary_to_gray(analysis.k3m.skeleton_raw)), (300, 360))
        self._set_preview("k3m_overlay", analysis.k3m.overlay, (300, 360))
        self._set_preview("kmm_raw", gray_to_pillow(binary_to_gray(analysis.kmm.skeleton_raw)), (300, 360))
        self._set_preview("kmm_overlay", analysis.kmm.overlay, (300, 360))

    def _apply_minutiae_filters(self) -> None:
        if self.analysis is None:
            return
        show_endings = bool(self.show_endings_var.get())
        show_bifurcations = bool(self.show_bifurcations_var.get())
        self.analysis.morph.overlay = overlay_minutiae(
            self.analysis.enhanced,
            self.analysis.morph.skeleton_final,
            self.analysis.morph.minutiae,
            show_endings=show_endings,
            show_bifurcations=show_bifurcations,
        )
        self.analysis.k3m.overlay = overlay_minutiae(
            self.analysis.enhanced,
            self.analysis.k3m.skeleton_final,
            self.analysis.k3m.minutiae,
            show_endings=show_endings,
            show_bifurcations=show_bifurcations,
        )
        self.analysis.kmm.overlay = overlay_minutiae(
            self.analysis.enhanced,
            self.analysis.kmm.skeleton_final,
            self.analysis.kmm.minutiae,
            show_endings=show_endings,
            show_bifurcations=show_bifurcations,
        )

    def _on_minutiae_filter_change(self) -> None:
        if self.analysis is None:
            return
        self._apply_minutiae_filters()
        self._set_preview("morph_overlay", self.analysis.morph.overlay, (300, 360))
        self._set_preview("k3m_overlay", self.analysis.k3m.overlay, (300, 360))
        self._set_preview("kmm_overlay", self.analysis.kmm.overlay, (300, 360))

    def _set_preview(self, key: str, image: Image.Image, max_size: tuple[int, int]) -> None:
        label = self.labels[key]
        preview = image.copy()
        preview.thumbnail(max_size)
        photo = ImageTk.PhotoImage(preview)
        self.preview_refs[key] = photo
        label.configure(image=photo, text="")

    def export_dashboard(self) -> None:
        if self.analysis is None:
            messagebox.showwarning("Brak danych", "Najpierw wykonaj analize obrazu.")
            return
        self._apply_minutiae_filters()
        path = filedialog.asksaveasfilename(
            title="Zapisz dashboard",
            defaultextension=".png",
            filetypes=[("PNG", "*.png")],
        )
        if not path:
            return

        dashboard = self._compose_dashboard()
        try:
            dashboard.save(path)
        except Exception as error:
            messagebox.showerror("Blad zapisu", f"Nie udalo sie zapisac dashboardu.\n\n{error}")
            return
        self.status_var.set(f"Wyeksportowano dashboard do: {Path(path).name}")

    def export_results_json(self) -> None:
        if self.analysis is None:
            messagebox.showwarning("Brak danych", "Najpierw wykonaj analize obrazu.")
            return
        path = filedialog.asksaveasfilename(
            title="Zapisz wyniki JSON",
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
        )
        if not path:
            return

        payload = {
            "image": self.current_path.name if self.current_path else None,
            "threshold": self.analysis.threshold,
            "summary": self.analysis.comparison_summary,
            "morphology": {
                "summary": self.analysis.morph.summary,
                "minutiae": len(self.analysis.morph.minutiae),
            },
            "k3m": {
                "summary": self.analysis.k3m.summary,
                "minutiae": len(self.analysis.k3m.minutiae),
            },
            "kmm": {
                "summary": self.analysis.kmm.summary,
                "minutiae": len(self.analysis.kmm.minutiae),
            },
            "parameters": {
                "binarization_method": self.binarization_method_var.get(),
                "invert": bool(self.invert_var.get()),
                "threshold_bias": int(self.threshold_bias_var.get()),
                "sauvola_window_size": int(self.sauvola_window_var.get()),
                "sauvola_k": float(self.sauvola_k_var.get()),
                "sauvola_dynamic_range": float(self.sauvola_range_var.get()),
                "mask_enabled": bool(self.mask_enabled_var.get()),
                "mask_window_size": int(self.mask_window_var.get()),
                "mask_min_std": float(self.mask_min_std_var.get()),
                "mask_kernel_size": int(self.mask_kernel_var.get()),
                "close_size": int(self.close_size_var.get()),
                "open_size": int(self.open_size_var.get()),
                "bridge_iterations": int(self.bridge_iterations_var.get()),
                "bridge_gap": int(self.bridge_gap_var.get()),
                "bridge_links_per_endpoint": int(self.bridge_links_var.get()),
                "spur_iterations": int(self.spur_iterations_var.get()),
                "spur_length": int(self.spur_length_var.get()),
                "minutiae_min_distance": int(self.min_distance_var.get()),
            },
        }

        try:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
        except Exception as error:
            messagebox.showerror("Blad zapisu", f"Nie udalo sie zapisac wynikow.\n\n{error}")
            return
        self.status_var.set(f"Wyeksportowano wyniki JSON do: {Path(path).name}")

    def _compose_dashboard(self) -> Image.Image:
        assert self.analysis is not None
        panels = [
            ("Oryginal", gray_to_pillow(self.analysis.source)),
            ("Wzmocnienie", gray_to_pillow(self.analysis.enhanced)),
            ("Binaryzacja", gray_to_pillow(binary_to_gray(self.analysis.binary))),
            ("Po oczyszczaniu", gray_to_pillow(binary_to_gray(self.analysis.cleaned))),
            ("Szkielet morfologiczny", gray_to_pillow(binary_to_gray(self.analysis.morph.skeleton_raw))),
            ("Minucje morfologiczne", self.analysis.morph.overlay),
            ("Szkielet K3M", gray_to_pillow(binary_to_gray(self.analysis.k3m.skeleton_raw))),
            ("Minucje K3M", self.analysis.k3m.overlay),
            ("Szkielet KMM", gray_to_pillow(binary_to_gray(self.analysis.kmm.skeleton_raw))),
            ("Minucje KMM", self.analysis.kmm.overlay),
        ]

        margin = 24
        gap = 16
        panel_width = 320
        panel_height = 260
        header_height = 140
        columns = 4
        rows = max(1, math.ceil(len(panels) / columns))
        width = margin * 2 + columns * panel_width + (columns - 1) * gap
        height = margin * 2 + header_height + rows * panel_height + (rows - 1) * gap
        canvas = Image.new("RGB", (width, height), "#f2f5f9")
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default()

        draw.rounded_rectangle((12, 12, width - 12, height - 12), radius=18, fill="#ffffff", outline="#d6deea")
        title = f"Biometria P3 - {self.current_path.name if self.current_path else 'dashboard'}"
        draw.text((margin, margin), title, fill="#0f172a", font=font)
        draw.text((margin, margin + 24), self.summary_var.get(), fill="#334155", font=font)
        draw.text((margin, margin + 42), self.morph_var.get(), fill="#334155", font=font)
        draw.text((margin, margin + 60), self.k3m_var.get(), fill="#334155", font=font)
        draw.text((margin, margin + 78), self.kmm_var.get(), fill="#334155", font=font)

        grid_top = margin + header_height
        for index, (panel_title, panel_image) in enumerate(panels):
            row = index // columns
            column = index % columns
            x = margin + column * (panel_width + gap)
            y = grid_top + row * (panel_height + gap)
            panel = self._compose_panel(panel_title, panel_image, panel_width, panel_height)
            canvas.paste(panel, (x, y))

        return canvas

    def _compose_panel(self, title: str, image: Image.Image, width: int, height: int) -> Image.Image:
        panel = Image.new("RGB", (width, height), "#ffffff")
        draw = ImageDraw.Draw(panel)
        font = ImageFont.load_default()
        draw.rounded_rectangle((0, 0, width - 1, height - 1), radius=16, fill="#ffffff", outline="#cbd5e1")
        draw.text((16, 14), title, fill="#0f172a", font=font)
        image_box = (16, 40, width - 16, height - 16)
        preview = image.convert("RGB")
        preview.thumbnail((image_box[2] - image_box[0], image_box[3] - image_box[1]))
        paste_x = image_box[0] + ((image_box[2] - image_box[0]) - preview.width) // 2
        paste_y = image_box[1] + ((image_box[3] - image_box[1]) - preview.height) // 2
        draw.rounded_rectangle(image_box, radius=12, fill="#f8fafc", outline="#e2e8f0")
        panel.paste(preview, (paste_x, paste_y))
        return panel


def main() -> None:
    root = tk.Tk()
    style = ttk.Style(root)
    if "vista" in style.theme_names():
        style.theme_use("vista")
    FingerprintThinningApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
