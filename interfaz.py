"""
Módulo de Interfaz Gráfica de Usuario (GUI)
Aplicación para Procesamiento Digital de Imágenes con 3 Pestañas Especializadas:
  - Pestaña 1: Reducción Espacial de Imágenes (por Máscara y Stride sin imresize).
  - Pestaña 2: Mitigación de Ruido en el Dominio Espacial (Filtros: Media, Moda, Gaussiano).
  - Pestaña 3: Filtrado en el Dominio de la Frecuencia (Transformada de Fourier 2D - Pasa-Bajas, Pasa-Altas, Notch).

Diseño Minimalista Profesional, Fondo Blanco y Pizarra Claro (Light Theme).
"""

import os
import sys
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import List, Optional
import numpy as np
from PIL import Image, ImageTk

from logica import (
    cargar_imagen,
    reducir_imagen,
    guardar_imagen,
    calcular_metricas,
    calcular_metricas_ruido,
    generar_mascara,
    generar_mascara_fourier,
    aplicar_filtro_fourier
)

# Paleta de Colorimetría y Superficies en Modo Claro (Minimalista Light)
THEME = {
    # Superficies de Contraste
    "bg_root": "#F8FAFC",              # Fondo general pizarra ultra claro
    "bg_card": "#FFFFFF",              # Tarjetas y paneles en blanco puro
    "bg_card_inner": "#F1F5F9",        # Fondo interno para contenedores de imagen y visores
    "bg_input": "#FFFFFF",             # Fondo para inputs y combos
    "border": "#E2E8F0",               # Bordes y divisores sutiles
    "border_highlight": "#CBD5E1",     # Bordes activos y elementos destacados

    # Textos y Tipografía
    "text_primary": "#0F172A",         # Texto principal negro pizarra (alto contraste)
    "text_secondary": "#475569",       # Subtítulos y textos de soporte
    "text_muted": "#64748B",           # Metadatos, etiquetas secundarias y guías
    "text_accent": "#0284C7",          # Azul océano acento

    # Acentos de Acción y Pestañas
    "accent_primary": "#4F46E5",        # Índigo vibrante (Reducción / Cargar)
    "accent_primary_hover": "#4338CA",
    "accent_success": "#059669",        # Esmeralda (Guardar / Exportar)
    "accent_success_hover": "#047857",
    "accent_secondary": "#F1F5F9",
    "accent_secondary_hover": "#E2E8F0",

    # Pestaña 2: Mitigación de Ruido Espacial
    "ruido_accent": "#7C3AED",         # Púrpura violeta
    "ruido_accent_hover": "#6D28D9",
    "ruido_badge_bg": "#F5F3FF",
    "ruido_badge_border": "#DDD6FE",

    # Pestaña 3: Transformada de Fourier
    "fourier_accent": "#0284C7",        # Azul océano / Cian profundo
    "fourier_accent_hover": "#0369A1",
    "fourier_badge_bg": "#E0F2FE",
    "fourier_badge_border": "#BAE6FD",

    # Identidad Cromática de Badges
    "orig_accent": "#D97706",          # Ámbar Cálido
    "orig_badge_bg": "#FEF3C7",
    "orig_badge_border": "#FDE68A",

    "red_accent": "#4F46E5",
    "red_badge_bg": "#EEF2FF",
    "red_badge_border": "#C7D2FE",

    "font_main": "Segoe UI" if sys.platform == "win32" else "Helvetica",
}


def listar_imagenes_muestra(directorio: str) -> List[str]:
    """Lista los archivos de imagen disponibles en la carpeta original."""
    if not os.path.exists(directorio):
        return []
    extensiones = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
    return sorted([
        f for f in os.listdir(directorio)
        if os.path.splitext(f.lower())[1] in extensiones
    ])


class InterfazReduccion:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Sistema de Procesamiento Espacial y Frecuencial (Fourier 2D)")
        self.root.geometry("1400x900")
        self.root.minsize(1120, 760)
        self.root.configure(bg=THEME["bg_root"])

        # Estado interno compartido
        self.pestaña_activa = "reduccion"  # "reduccion", "ruido", "fourier"
        self.img_original: Optional[np.ndarray] = None
        self.img_reducida: Optional[np.ndarray] = None
        self.img_filtrada_ruido: Optional[np.ndarray] = None
        self.resultado_fourier: Optional[dict] = None
        self.ruta_imagen_actual: str = ""

        self.dir_base = os.path.dirname(os.path.abspath(__file__))
        self.dir_img_orig = os.path.join(self.dir_base, "img", "original")
        self.dir_img_red = os.path.join(self.dir_base, "img", "reducida")
        os.makedirs(self.dir_img_orig, exist_ok=True)
        os.makedirs(self.dir_img_red, exist_ok=True)

        # Referencias de imágenes Tkinter para evitar garbage collector
        self._tk_orig_red = None
        self._tk_salida_red = None
        self._tk_orig_ruido = None
        self._tk_salida_ruido = None
        self._tk_orig_four = None
        self._tk_salida_four = None
        self._tk_mask_four = None
        self._tk_esp_four = None

        # Configurar estilos ttk
        self._configurar_estilos()

        # Construir UI
        self._construir_layout()

        # Cargar muestra inicial si existe
        self.root.after(100, self._cargar_muestra_defecto)

    def _configurar_estilos(self):
        style = ttk.Style()
        style.theme_use("clam")

        self.root.option_add("*TCombobox*Listbox.background", THEME["bg_card"])
        self.root.option_add("*TCombobox*Listbox.foreground", THEME["text_primary"])
        self.root.option_add("*TCombobox*Listbox.selectBackground", THEME["accent_primary"])
        self.root.option_add("*TCombobox*Listbox.selectForeground", "#FFFFFF")

        style.configure(
            "TCombobox",
            background=THEME["bg_card"],
            foreground=THEME["text_primary"],
            fieldbackground=THEME["bg_card"],
            bordercolor=THEME["border_highlight"],
            darkcolor=THEME["border"],
            lightcolor=THEME["border"],
            arrowcolor=THEME["text_secondary"],
            padding=4
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", THEME["bg_card"])],
            selectbackground=[("readonly", THEME["bg_card_inner"])],
            selectforeground=[("readonly", THEME["text_primary"])],
        )

    def _construir_layout(self):
        # 1. Cabecera (Header) en Blanco Puro con borde inferior sutil
        header_frame = tk.Frame(self.root, bg=THEME["bg_card"], height=68, padx=24)
        header_frame.pack(fill=tk.X, side=tk.TOP)
        header_frame.pack_propagate(False)

        titulo_box = tk.Frame(header_frame, bg=THEME["bg_card"])
        titulo_box.pack(side=tk.LEFT, pady=10)

        strip_row = tk.Frame(titulo_box, bg=THEME["bg_card"])
        strip_row.pack(anchor="w", pady=(0, 2))
        tk.Label(strip_row, text="●", font=(THEME["font_main"], 8, "bold"), fg="#E11D48", bg=THEME["bg_card"]).pack(side=tk.LEFT, padx=(0, 2))
        tk.Label(strip_row, text="●", font=(THEME["font_main"], 8, "bold"), fg="#059669", bg=THEME["bg_card"]).pack(side=tk.LEFT, padx=(0, 2))
        tk.Label(strip_row, text="●", font=(THEME["font_main"], 8, "bold"), fg="#0284C7", bg=THEME["bg_card"]).pack(side=tk.LEFT, padx=(0, 6))
        tk.Label(
            strip_row,
            text="PROCESAMIENTO ESPACIAL Y FRECUENCIAL • TRANSFORMADA DE FOURIER 2D",
            font=(THEME["font_main"], 8, "bold"),
            fg=THEME["text_muted"],
            bg=THEME["bg_card"]
        ).pack(side=tk.LEFT)

        lbl_titulo = tk.Label(
            titulo_box,
            text="Reducción, Mitigación de Ruido y Filtrado en Frecuencia (Fourier)",
            font=(THEME["font_main"], 13, "bold"),
            fg=THEME["text_primary"],
            bg=THEME["bg_card"]
        )
        lbl_titulo.pack(anchor="w")

        # Controles y botones en cabecera
        btn_box = tk.Frame(header_frame, bg=THEME["bg_card"])
        btn_box.pack(side=tk.RIGHT, pady=14)

        lbl_muestras = tk.Label(
            btn_box,
            text="Muestra:",
            font=(THEME["font_main"], 9),
            bg=THEME["bg_card"],
            fg=THEME["text_secondary"],
        )
        lbl_muestras.pack(side=tk.LEFT, padx=(0, 6))

        muestras = listar_imagenes_muestra(self.dir_img_orig)
        valores_combo = ["-- Muestras img/original --"] + muestras
        self.combo_muestras = ttk.Combobox(
            btn_box,
            values=valores_combo,
            state="readonly",
            width=22,
            font=(THEME["font_main"], 9),
        )
        self.combo_muestras.set(muestras[0] if muestras else "-- Muestras img/original --")
        self.combo_muestras.bind("<<ComboboxSelected>>", self._on_muestra_seleccionada)
        self.combo_muestras.pack(side=tk.LEFT, padx=(0, 10))

        self.btn_cargar = tk.Button(
            btn_box,
            text="📂 Cargar Imagen...",
            font=(THEME["font_main"], 9, "bold"),
            bg=THEME["accent_primary"],
            fg="#FFFFFF",
            activebackground=THEME["accent_primary_hover"],
            activeforeground="#FFFFFF",
            relief=tk.FLAT,
            padx=12,
            pady=5,
            cursor="hand2",
            command=self._seleccionar_imagen
        )
        self.btn_cargar.pack(side=tk.LEFT, padx=(0, 8))

        self.btn_guardar = tk.Button(
            btn_box,
            text="💾 Exportar Resultado",
            font=(THEME["font_main"], 9, "bold"),
            bg=THEME["accent_success"],
            fg="#FFFFFF",
            activebackground=THEME["accent_success_hover"],
            activeforeground="#FFFFFF",
            relief=tk.FLAT,
            padx=12,
            pady=5,
            cursor="hand2",
            command=self._guardar_imagen_activa
        )
        self.btn_guardar.pack(side=tk.LEFT)

        # 2. Barra de Navegación por 3 Pestañas (Tabs)
        tab_nav_frame = tk.Frame(self.root, bg=THEME["bg_card"], height=44, padx=24)
        tab_nav_frame.pack(fill=tk.X, side=tk.TOP)
        tab_nav_frame.pack_propagate(False)

        self.btn_tab_red = tk.Button(
            tab_nav_frame,
            text="  📉 Pestaña 1: Reducción Espacial (Stride)  ",
            font=(THEME["font_main"], 9, "bold"),
            relief=tk.FLAT,
            cursor="hand2",
            command=lambda: self._cambiar_pestaña("reduccion")
        )
        self.btn_tab_red.pack(side=tk.LEFT, pady=4, padx=(0, 6))

        self.btn_tab_ruido = tk.Button(
            tab_nav_frame,
            text="  🛡️ Pestaña 2: Mitigación Espacial (Media / Moda / Gaussiano)  ",
            font=(THEME["font_main"], 9, "bold"),
            relief=tk.FLAT,
            cursor="hand2",
            command=lambda: self._cambiar_pestaña("ruido")
        )
        self.btn_tab_ruido.pack(side=tk.LEFT, pady=4, padx=(0, 6))

        self.btn_tab_fourier = tk.Button(
            tab_nav_frame,
            text="  🌊 Pestaña 3: Transformada de Fourier (Frecuencia)  ",
            font=(THEME["font_main"], 9, "bold"),
            relief=tk.FLAT,
            cursor="hand2",
            command=lambda: self._cambiar_pestaña("fourier")
        )
        self.btn_tab_fourier.pack(side=tk.LEFT, pady=4)

        # Divisor sutil
        sep = tk.Frame(self.root, bg=THEME["border"], height=1)
        sep.pack(fill=tk.X)

        # 3. Contenedor de Contenido Dinámico
        self.container_principal = tk.Frame(self.root, bg=THEME["bg_root"])
        self.container_principal.pack(fill=tk.BOTH, expand=True, padx=20, pady=12)

        # Construir las 3 pestañas
        self._construir_pestaña_reduccion()
        self._construir_pestaña_ruido()
        self._construir_pestaña_fourier()

        # Activar Pestaña 1 por defecto
        self._actualizar_estilo_botones_tab()
        self.frame_reduccion.pack(fill=tk.BOTH, expand=True)

        # 4. Footer / Barra de Estado
        footer_frame = tk.Frame(
            self.root,
            bg=THEME["bg_card"],
            padx=24,
            pady=8,
            highlightthickness=1,
            highlightbackground=THEME["border"]
        )
        footer_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.lbl_estado = tk.Label(
            footer_frame,
            text="Listo • Selecciona una muestra o carga un archivo para procesar",
            font=(THEME["font_main"], 9),
            bg=THEME["bg_card"],
            fg=THEME["text_secondary"]
        )
        self.lbl_estado.pack(side=tk.LEFT)

        self.lbl_metricas_footer = tk.Label(
            footer_frame,
            text="",
            font=(THEME["font_main"], 9),
            bg=THEME["bg_card"],
            fg=THEME["text_muted"]
        )
        self.lbl_metricas_footer.pack(side=tk.RIGHT)

    def _actualizar_estilo_botones_tab(self):
        tabs = [
            ("reduccion", self.btn_tab_red, THEME["accent_primary"], THEME["accent_primary_hover"]),
            ("ruido", self.btn_tab_ruido, THEME["ruido_accent"], THEME["ruido_accent_hover"]),
            ("fourier", self.btn_tab_fourier, THEME["fourier_accent"], THEME["fourier_accent_hover"]),
        ]
        for nombre, btn, color_bg, color_hover in tabs:
            if self.pestaña_activa == nombre:
                btn.config(bg=color_bg, fg="#FFFFFF", activebackground=color_hover, activeforeground="#FFFFFF")
            else:
                btn.config(bg=THEME["bg_card_inner"], fg=THEME["text_secondary"], activebackground=THEME["border"], activeforeground=THEME["text_primary"])

    def _cambiar_pestaña(self, nueva: str):
        if self.pestaña_activa == nueva:
            return
        self.pestaña_activa = nueva
        self._actualizar_estilo_botones_tab()

        self.frame_reduccion.pack_forget()
        self.frame_ruido.pack_forget()
        self.frame_fourier.pack_forget()

        if nueva == "reduccion":
            self.frame_reduccion.pack(fill=tk.BOTH, expand=True)
            self.procesar_reduccion()
        elif nueva == "ruido":
            self.frame_ruido.pack(fill=tk.BOTH, expand=True)
            self.procesar_mitigacion_ruido()
        else:
            self.frame_fourier.pack(fill=tk.BOTH, expand=True)
            self.procesar_fourier()

    # =========================================================================
    # PESTAÑA 1: REDUCCIÓN ESPACIAL (STRIDE Y MÁSCARA)
    # =========================================================================
    def _construir_pestaña_reduccion(self):
        self.frame_reduccion = tk.Frame(self.container_principal, bg=THEME["bg_root"])

        sidebar = tk.Frame(self.frame_reduccion, bg=THEME["bg_card"], width=330, padx=18, pady=14, highlightthickness=1, highlightbackground=THEME["border"])
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 14))
        sidebar.pack_propagate(False)

        lbl_panel = tk.Label(sidebar, text="⚙️ PARÁMETROS DE REDUCCIÓN", font=(THEME["font_main"], 10, "bold"), fg=THEME["text_accent"], bg=THEME["bg_card"])
        lbl_panel.pack(anchor="w", pady=(0, 12))

        # 1. Tamaño Máscara
        tk.Label(sidebar, text="1. Tamaño de Máscara (Kernel):", font=(THEME["font_main"], 9, "bold"), fg=THEME["text_primary"], bg=THEME["bg_card"]).pack(anchor="w", pady=(2, 3))
        self.var_mascara_red = tk.StringVar(value="3x3")
        self.combo_mascara_red = ttk.Combobox(sidebar, textvariable=self.var_mascara_red, values=["3x3", "5x5", "7x7", "9x9", "11x11", "13x13", "15x15"], state="readonly", font=(THEME["font_main"], 9))
        self.combo_mascara_red.pack(fill=tk.X, pady=(0, 10))
        self.combo_mascara_red.bind("<<ComboboxSelected>>", lambda e: self._actualizar_mascara_red_y_procesar())

        # 2. Salto Stride
        lbl_stride_box = tk.Frame(sidebar, bg=THEME["bg_card"])
        lbl_stride_box.pack(fill=tk.X, pady=(2, 2))
        tk.Label(lbl_stride_box, text="2. Salto de Muestreo (Stride):", font=(THEME["font_main"], 9, "bold"), fg=THEME["text_primary"], bg=THEME["bg_card"]).pack(side=tk.LEFT)
        self.lbl_stride_red_val = tk.Label(lbl_stride_box, text="2 px (1/2)", font=(THEME["font_main"], 9, "bold"), fg=THEME["text_accent"], bg=THEME["bg_card_inner"], padx=6, pady=1)
        self.lbl_stride_red_val.pack(side=tk.RIGHT)

        self.slider_stride_red = tk.Scale(sidebar, from_=2, to=8, orient=tk.HORIZONTAL, bg=THEME["bg_card"], fg=THEME["text_primary"], troughcolor=THEME["bg_card_inner"], activebackground=THEME["accent_primary"], highlightthickness=0, command=self._on_stride_red_change)
        self.slider_stride_red.set(2)
        self.slider_stride_red.pack(fill=tk.X, pady=(0, 10))

        # 3. Filtro Antialiasing
        tk.Label(sidebar, text="3. Filtro Antialiasing Previo:", font=(THEME["font_main"], 9, "bold"), fg=THEME["text_primary"], bg=THEME["bg_card"]).pack(anchor="w", pady=(2, 3))
        self.var_tipo_red = tk.StringVar(value="gaussiano")
        self.combo_tipo_red = ttk.Combobox(sidebar, textvariable=self.var_tipo_red, values=["gaussiano", "media", "ponderado", "personalizado"], state="readonly", font=(THEME["font_main"], 9))
        self.combo_tipo_red.pack(fill=tk.X, pady=(0, 10))
        self.combo_tipo_red.bind("<<ComboboxSelected>>", lambda e: self._actualizar_mascara_red_y_procesar())

        # 4. Sigma Gaussiano
        lbl_sigma_box_red = tk.Frame(sidebar, bg=THEME["bg_card"])
        lbl_sigma_box_red.pack(fill=tk.X, pady=(2, 2))
        self.lbl_sigma_red = tk.Label(lbl_sigma_box_red, text="4. Sigma (Filtro Gaussiano):", font=(THEME["font_main"], 9, "bold"), fg=THEME["text_primary"], bg=THEME["bg_card"])
        self.lbl_sigma_red.pack(side=tk.LEFT)
        self.lbl_sigma_red_val = tk.Label(lbl_sigma_box_red, text="1.0", font=(THEME["font_main"], 9, "bold"), fg=THEME["text_accent"], bg=THEME["bg_card_inner"], padx=6, pady=1)
        self.lbl_sigma_red_val.pack(side=tk.RIGHT)

        self.slider_sigma_red = tk.Scale(sidebar, from_=0.3, to=3.0, resolution=0.1, orient=tk.HORIZONTAL, bg=THEME["bg_card"], fg=THEME["text_primary"], troughcolor=THEME["bg_card_inner"], activebackground=THEME["accent_primary"], highlightthickness=0, command=self._on_sigma_red_change)
        self.slider_sigma_red.set(1.0)
        self.slider_sigma_red.pack(fill=tk.X, pady=(0, 4))

        self.lbl_sigma_red_hint = tk.Label(sidebar, text="Control activo para la campana de Gauss", font=(THEME["font_main"], 7), fg=THEME["text_muted"], bg=THEME["bg_card"])
        self.lbl_sigma_red_hint.pack(anchor="w", pady=(0, 10))

        # Botón Reducción
        self.btn_procesar_red = tk.Button(sidebar, text="⚡ APLICAR REDUCCIÓN", font=(THEME["font_main"], 10, "bold"), bg=THEME["accent_primary"], fg="#FFFFFF", activebackground=THEME["accent_primary_hover"], activeforeground="#FFFFFF", relief=tk.FLAT, pady=8, cursor="hand2", command=self.procesar_reduccion)
        self.btn_procesar_red.pack(fill=tk.X, pady=(2, 10))

        # Métricas
        tk.Label(sidebar, text="📊 Métricas de Compresión:", font=(THEME["font_main"], 9, "bold"), fg=THEME["text_secondary"], bg=THEME["bg_card"]).pack(anchor="w")
        self.lbl_metricas_red = tk.Label(sidebar, text="Original: - x -\nReducida: - x -\nFactor Escalar: -\nReducción de Píxeles: -\nTiempo: -", font=(THEME["font_main"], 8), fg=THEME["text_primary"], bg=THEME["bg_card_inner"], justify=tk.LEFT, padx=10, pady=8, relief=tk.FLAT)
        self.lbl_metricas_red.pack(fill=tk.X, pady=(4, 0))

        # Panel Central y Derecho
        visores_frame = tk.Frame(self.frame_reduccion, bg=THEME["bg_root"])
        visores_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        visores_frame.columnconfigure(0, weight=1, uniform="layout_cols_red")
        visores_frame.columnconfigure(1, weight=1, uniform="layout_cols_red")
        visores_frame.rowconfigure(0, weight=1)

        # Matriz Central
        self.panel_matriz_red = tk.Frame(visores_frame, bg=THEME["bg_card"], padx=14, pady=10, highlightthickness=1, highlightbackground=THEME["border"])
        self.panel_matriz_red.grid(row=0, column=0, sticky="nsew", padx=(0, 7))
        tk.Frame(self.panel_matriz_red, bg=THEME["accent_primary"], height=2).pack(fill=tk.X, pady=(0, 6))

        h_mat = tk.Frame(self.panel_matriz_red, bg=THEME["bg_card"])
        h_mat.pack(fill=tk.X, pady=(0, 4))
        badge_box = tk.Frame(h_mat, bg="#E0E7FF", padx=1, pady=1)
        badge_box.pack(side=tk.RIGHT)
        self.lbl_badge_tam_mascara_red = tk.Label(badge_box, text=" ● KERNEL 3x3 ", font=(THEME["font_main"], 8, "bold"), fg=THEME["accent_primary"], bg="#EEF2FF", padx=6, pady=1)
        self.lbl_badge_tam_mascara_red.pack()
        tk.Label(h_mat, text="MATRIZ DE LA MÁSCARA ACTIVA", font=(THEME["font_main"], 10, "bold"), fg=THEME["text_primary"], bg=THEME["bg_card"]).pack(side=tk.LEFT)

        self.lbl_subtitulo_matriz_red = tk.Label(self.panel_matriz_red, text="Coeficientes de convolución espacial para atenuación de aliasing", font=(THEME["font_main"], 8), fg=THEME["text_muted"], bg=THEME["bg_card"])
        self.lbl_subtitulo_matriz_red.pack(anchor="w", pady=(0, 6))

        txt_cont = tk.Frame(self.panel_matriz_red, bg=THEME["bg_card_inner"], highlightthickness=1, highlightbackground=THEME["border"])
        txt_cont.pack(fill=tk.BOTH, expand=True)
        sc_x = tk.Scrollbar(txt_cont, orient=tk.HORIZONTAL)
        sc_y = tk.Scrollbar(txt_cont, orient=tk.VERTICAL)
        self.txt_matriz_red = tk.Text(txt_cont, wrap=tk.NONE, bg=THEME["bg_card_inner"], fg=THEME["text_primary"], font=("Consolas", 10), relief=tk.FLAT, padx=12, pady=12, xscrollcommand=sc_x.set, yscrollcommand=sc_y.set)
        sc_x.config(command=self.txt_matriz_red.xview)
        sc_y.config(command=self.txt_matriz_red.yview)
        sc_y.pack(side=tk.RIGHT, fill=tk.Y)
        sc_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.txt_matriz_red.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        t_bar = tk.Frame(self.panel_matriz_red, bg=THEME["bg_card"], pady=6)
        t_bar.pack(fill=tk.X)
        self.btn_aplicar_matriz_red = tk.Button(t_bar, text="✨ Aplicar Esta Matriz", font=(THEME["font_main"], 9, "bold"), bg=THEME["accent_primary"], fg="#FFFFFF", relief=tk.FLAT, padx=10, pady=4, cursor="hand2", command=self._aplicar_matriz_editada_red)
        self.btn_aplicar_matriz_red.pack(side=tk.LEFT, padx=(0, 6))
        self.btn_norm_matriz_red = tk.Button(t_bar, text="⚖️ Normalizar (Suma=1)", font=(THEME["font_main"], 8), bg=THEME["bg_card_inner"], fg=THEME["text_primary"], relief=tk.FLAT, padx=8, pady=4, cursor="hand2", command=self._normalizar_matriz_actual_red)
        self.btn_norm_matriz_red.pack(side=tk.LEFT, padx=(0, 6))
        self.btn_rest_matriz_red = tk.Button(t_bar, text="↺ Restaurar", font=(THEME["font_main"], 8), bg=THEME["bg_card_inner"], fg=THEME["text_secondary"], relief=tk.FLAT, padx=8, pady=4, cursor="hand2", command=self._restaurar_matriz_red)
        self.btn_rest_matriz_red.pack(side=tk.LEFT)

        self.lbl_estado_matriz_red = tk.Label(self.panel_matriz_red, text="Solo lectura. Selecciona 'personalizado' para editar libremente.", font=(THEME["font_main"], 8), fg=THEME["text_muted"], bg=THEME["bg_card"])
        self.lbl_estado_matriz_red.pack(anchor="w", pady=(2, 0))

        # Canvases Derecha
        derecha_col = tk.Frame(visores_frame, bg=THEME["bg_root"])
        derecha_col.grid(row=0, column=1, sticky="nsew", padx=(7, 0))
        derecha_col.rowconfigure(0, weight=1)
        derecha_col.rowconfigure(1, weight=1)
        derecha_col.columnconfigure(0, weight=1)

        # Original
        card_orig = tk.Frame(derecha_col, bg=THEME["bg_card"], padx=10, pady=6, highlightthickness=1, highlightbackground=THEME["border"])
        card_orig.grid(row=0, column=0, sticky="nsew", pady=(0, 5))
        tk.Frame(card_orig, bg=THEME["orig_accent"], height=2).pack(fill=tk.X, pady=(0, 4))
        h_orig = tk.Frame(card_orig, bg=THEME["bg_card"])
        h_orig.pack(fill=tk.X)
        badge_o = tk.Frame(h_orig, bg=THEME["orig_badge_border"], padx=1, pady=1)
        badge_o.pack(side=tk.RIGHT)
        tk.Label(badge_o, text=" ● ENTRADA ", font=(THEME["font_main"], 8, "bold"), fg=THEME["orig_accent"], bg=THEME["orig_badge_bg"], padx=6).pack()
        tk.Label(h_orig, text="IMAGEN ORIGINAL", font=(THEME["font_main"], 9, "bold"), fg=THEME["text_primary"], bg=THEME["bg_card"]).pack(side=tk.LEFT)

        self.lbl_dim_orig_red = tk.Label(card_orig, text="0 x 0 px", font=(THEME["font_main"], 8), fg=THEME["text_muted"], bg=THEME["bg_card"])
        self.lbl_dim_orig_red.pack(anchor="w", pady=(0, 2))
        self.canvas_orig_red = tk.Canvas(card_orig, bg=THEME["bg_card_inner"], highlightthickness=0, bd=0)
        self.canvas_orig_red.pack(fill=tk.BOTH, expand=True)
        self.canvas_orig_red.bind("<Configure>", lambda e: self._redibujar_reduccion())

        # Reducida
        card_red = tk.Frame(derecha_col, bg=THEME["bg_card"], padx=10, pady=6, highlightthickness=1, highlightbackground=THEME["border"])
        card_red.grid(row=1, column=0, sticky="nsew", pady=(5, 0))
        tk.Frame(card_red, bg=THEME["red_accent"], height=2).pack(fill=tk.X, pady=(0, 4))
        h_red = tk.Frame(card_red, bg=THEME["bg_card"])
        h_red.pack(fill=tk.X)
        badge_r = tk.Frame(h_red, bg=THEME["red_badge_border"], padx=1, pady=1)
        badge_r.pack(side=tk.RIGHT)
        tk.Label(badge_r, text=" ● SALIDA ", font=(THEME["font_main"], 8, "bold"), fg=THEME["red_accent"], bg=THEME["red_badge_bg"], padx=6).pack()
        tk.Label(h_red, text="IMAGEN REDUCIDA", font=(THEME["font_main"], 9, "bold"), fg=THEME["text_primary"], bg=THEME["bg_card"]).pack(side=tk.LEFT)

        self.lbl_dim_salida_red = tk.Label(card_red, text="0 x 0 px", font=(THEME["font_main"], 8), fg=THEME["text_muted"], bg=THEME["bg_card"])
        self.lbl_dim_salida_red.pack(anchor="w", pady=(0, 2))
        self.canvas_salida_red = tk.Canvas(card_red, bg=THEME["bg_card_inner"], highlightthickness=0, bd=0)
        self.canvas_salida_red.pack(fill=tk.BOTH, expand=True)
        self.canvas_salida_red.bind("<Configure>", lambda e: self._redibujar_reduccion())

    # =========================================================================
    # PESTAÑA 2: MITIGACIÓN DE RUIDO ESPACIAL (MEDIA, MODA, GAUSSIANO)
    # =========================================================================
    def _construir_pestaña_ruido(self):
        self.frame_ruido = tk.Frame(self.container_principal, bg=THEME["bg_root"])

        sidebar = tk.Frame(self.frame_ruido, bg=THEME["bg_card"], width=330, padx=18, pady=14, highlightthickness=1, highlightbackground=THEME["border"])
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 14))
        sidebar.pack_propagate(False)

        lbl_panel = tk.Label(sidebar, text="🛡️ MITIGACIÓN DE RUIDO ESPACIAL", font=(THEME["font_main"], 10, "bold"), fg=THEME["ruido_accent"], bg=THEME["bg_card"])
        lbl_panel.pack(anchor="w", pady=(0, 12))

        # 1. Filtro Especializado
        tk.Label(sidebar, text="1. Filtro Especializado:", font=(THEME["font_main"], 9, "bold"), fg=THEME["text_primary"], bg=THEME["bg_card"]).pack(anchor="w", pady=(2, 3))
        self.var_tipo_ruido = tk.StringVar(value="media")
        self.combo_tipo_ruido = ttk.Combobox(sidebar, textvariable=self.var_tipo_ruido, values=["media", "moda", "gaussiano"], state="readonly", font=(THEME["font_main"], 9))
        self.combo_tipo_ruido.pack(fill=tk.X, pady=(0, 10))
        self.combo_tipo_ruido.bind("<<ComboboxSelected>>", lambda e: self._actualizar_info_filtro_ruido_y_procesar())

        # 2. Tamaño de Vecindad
        tk.Label(sidebar, text="2. Tamaño de Vecindad / Ventana:", font=(THEME["font_main"], 9, "bold"), fg=THEME["text_primary"], bg=THEME["bg_card"]).pack(anchor="w", pady=(2, 3))
        self.var_mascara_ruido = tk.StringVar(value="3x3")
        self.combo_mascara_ruido = ttk.Combobox(sidebar, textvariable=self.var_mascara_ruido, values=["3x3", "5x5", "7x7", "9x9", "11x11"], state="readonly", font=(THEME["font_main"], 9))
        self.combo_mascara_ruido.pack(fill=tk.X, pady=(0, 10))
        self.combo_mascara_ruido.bind("<<ComboboxSelected>>", lambda e: self._actualizar_info_filtro_ruido_y_procesar())

        # 3. Sigma Gaussiano
        lbl_sigma_box_ruido = tk.Frame(sidebar, bg=THEME["bg_card"])
        lbl_sigma_box_ruido.pack(fill=tk.X, pady=(2, 2))
        self.lbl_sigma_ruido = tk.Label(lbl_sigma_box_ruido, text="3. Sigma (Filtro Gaussiano):", font=(THEME["font_main"], 9, "bold"), fg=THEME["text_primary"], bg=THEME["bg_card"])
        self.lbl_sigma_ruido.pack(side=tk.LEFT)
        self.lbl_sigma_ruido_val = tk.Label(lbl_sigma_box_ruido, text="1.0", font=(THEME["font_main"], 9, "bold"), fg=THEME["ruido_accent"], bg=THEME["bg_card_inner"], padx=6, pady=1)
        self.lbl_sigma_ruido_val.pack(side=tk.RIGHT)

        self.slider_sigma_ruido = tk.Scale(sidebar, from_=0.3, to=3.0, resolution=0.1, orient=tk.HORIZONTAL, bg=THEME["bg_card"], fg=THEME["text_primary"], troughcolor=THEME["bg_card_inner"], activebackground=THEME["ruido_accent"], highlightthickness=0, command=self._on_sigma_ruido_change)
        self.slider_sigma_ruido.set(1.0)
        self.slider_sigma_ruido.pack(fill=tk.X, pady=(0, 4))
        self.lbl_sigma_ruido_hint = tk.Label(sidebar, text="⚠️ Inactivo: solo aplica para filtro Gaussiano", font=(THEME["font_main"], 7), fg="#D97706", bg=THEME["bg_card"])
        self.lbl_sigma_ruido_hint.pack(anchor="w", pady=(0, 10))

        # 4. Salto (Stride)
        lbl_stride_box = tk.Frame(sidebar, bg=THEME["bg_card"])
        lbl_stride_box.pack(fill=tk.X, pady=(2, 2))
        tk.Label(lbl_stride_box, text="4. Salto / Escala (Stride):", font=(THEME["font_main"], 9, "bold"), fg=THEME["text_primary"], bg=THEME["bg_card"]).pack(side=tk.LEFT)
        self.lbl_stride_ruido_val = tk.Label(lbl_stride_box, text="1 px (Sin reducción)", font=(THEME["font_main"], 8, "bold"), fg=THEME["ruido_accent"], bg=THEME["bg_card_inner"], padx=6, pady=1)
        self.lbl_stride_ruido_val.pack(side=tk.RIGHT)

        self.slider_stride_ruido = tk.Scale(sidebar, from_=1, to=4, orient=tk.HORIZONTAL, bg=THEME["bg_card"], fg=THEME["text_primary"], troughcolor=THEME["bg_card_inner"], activebackground=THEME["ruido_accent"], highlightthickness=0, command=self._on_stride_ruido_change)
        self.slider_stride_ruido.set(1)
        self.slider_stride_ruido.pack(fill=tk.X, pady=(0, 10))

        # Botón
        self.btn_procesar_ruido = tk.Button(sidebar, text="🛡️ APLICAR FILTRO DE RUIDO", font=(THEME["font_main"], 10, "bold"), bg=THEME["ruido_accent"], fg="#FFFFFF", activebackground=THEME["ruido_accent_hover"], activeforeground="#FFFFFF", relief=tk.FLAT, pady=8, cursor="hand2", command=self.procesar_mitigacion_ruido)
        self.btn_procesar_ruido.pack(fill=tk.X, pady=(2, 10))

        # Métricas
        tk.Label(sidebar, text="📊 Evaluación de Ruido y Limpieza:", font=(THEME["font_main"], 9, "bold"), fg=THEME["text_secondary"], bg=THEME["bg_card"]).pack(anchor="w")
        self.lbl_metricas_ruido = tk.Label(sidebar, text="Resolución: - x -\nCorrección Media (MAE): -\nCorrección Máxima: -\nPíxeles Procesados: -\nTiempo: -", font=(THEME["font_main"], 8), fg=THEME["text_primary"], bg=THEME["bg_card_inner"], justify=tk.LEFT, padx=10, pady=8, relief=tk.FLAT)
        self.lbl_metricas_ruido.pack(fill=tk.X, pady=(4, 0))

        # Panel Central y Derecho
        visores_frame = tk.Frame(self.frame_ruido, bg=THEME["bg_root"])
        visores_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        visores_frame.columnconfigure(0, weight=1, uniform="layout_cols_ruido")
        visores_frame.columnconfigure(1, weight=1, uniform="layout_cols_ruido")
        visores_frame.rowconfigure(0, weight=1)

        # Panel Diagnóstico
        self.panel_diagnostico_card = tk.Frame(visores_frame, bg=THEME["bg_card"], padx=14, pady=10, highlightthickness=1, highlightbackground=THEME["border"])
        self.panel_diagnostico_card.grid(row=0, column=0, sticky="nsew", padx=(0, 7))
        tk.Frame(self.panel_diagnostico_card, bg=THEME["ruido_accent"], height=2).pack(fill=tk.X, pady=(0, 6))

        h_diag = tk.Frame(self.panel_diagnostico_card, bg=THEME["bg_card"])
        h_diag.pack(fill=tk.X, pady=(0, 4))
        badge_ruido_box = tk.Frame(h_diag, bg=THEME["ruido_badge_border"], padx=1, pady=1)
        badge_ruido_box.pack(side=tk.RIGHT)
        self.lbl_badge_filtro_ruido = tk.Label(badge_ruido_box, text=" ● FILTRO MEDIA ", font=(THEME["font_main"], 8, "bold"), fg=THEME["ruido_accent"], bg=THEME["ruido_badge_bg"], padx=6, pady=1)
        self.lbl_badge_filtro_ruido.pack()
        tk.Label(h_diag, text="DIAGNÓSTICO Y ANÁLISIS DEL FILTRO", font=(THEME["font_main"], 10, "bold"), fg=THEME["text_primary"], bg=THEME["bg_card"]).pack(side=tk.LEFT)

        self.lbl_subtitulo_diag = tk.Label(self.panel_diagnostico_card, text="Comportamiento matemático frente a perturbaciones de ruido en la imagen", font=(THEME["font_main"], 8), fg=THEME["text_muted"], bg=THEME["bg_card"])
        self.lbl_subtitulo_diag.pack(anchor="w", pady=(0, 6))

        self.card_info_ruido = tk.Label(self.panel_diagnostico_card, text="Cargando información del filtro...", font=(THEME["font_main"], 8), fg=THEME["text_primary"], bg=THEME["bg_card_inner"], justify=tk.LEFT, padx=10, pady=8, relief=tk.FLAT)
        self.card_info_ruido.pack(fill=tk.X, pady=(0, 8))

        tk.Label(self.panel_diagnostico_card, text="Matriz de Vecindad de Mitigación:", font=(THEME["font_main"], 8, "bold"), fg=THEME["text_secondary"], bg=THEME["bg_card"]).pack(anchor="w", pady=(0, 3))
        txt_cont_ruido = tk.Frame(self.panel_diagnostico_card, bg=THEME["bg_card_inner"], highlightthickness=1, highlightbackground=THEME["border"])
        txt_cont_ruido.pack(fill=tk.BOTH, expand=True)

        sc_x_r = tk.Scrollbar(txt_cont_ruido, orient=tk.HORIZONTAL)
        sc_y_r = tk.Scrollbar(txt_cont_ruido, orient=tk.VERTICAL)
        self.txt_matriz_ruido = tk.Text(txt_cont_ruido, wrap=tk.NONE, bg=THEME["bg_card_inner"], fg=THEME["text_primary"], font=("Consolas", 10), relief=tk.FLAT, padx=12, pady=12, xscrollcommand=sc_x_r.set, yscrollcommand=sc_y_r.set)
        sc_x_r.config(command=self.txt_matriz_ruido.xview)
        sc_y_r.config(command=self.txt_matriz_ruido.yview)
        sc_y_r.pack(side=tk.RIGHT, fill=tk.Y)
        sc_x_r.pack(side=tk.BOTTOM, fill=tk.X)
        self.txt_matriz_ruido.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.lbl_estado_ruido_diag = tk.Label(self.panel_diagnostico_card, text="Listo para procesar la imagen con ruido.", font=(THEME["font_main"], 8), fg=THEME["text_muted"], bg=THEME["bg_card"])
        self.lbl_estado_ruido_diag.pack(anchor="w", pady=(4, 0))

        # Canvases Derecha
        derecha_col = tk.Frame(visores_frame, bg=THEME["bg_root"])
        derecha_col.grid(row=0, column=1, sticky="nsew", padx=(7, 0))
        derecha_col.rowconfigure(0, weight=1)
        derecha_col.rowconfigure(1, weight=1)
        derecha_col.columnconfigure(0, weight=1)

        card_orig = tk.Frame(derecha_col, bg=THEME["bg_card"], padx=10, pady=6, highlightthickness=1, highlightbackground=THEME["border"])
        card_orig.grid(row=0, column=0, sticky="nsew", pady=(0, 5))
        tk.Frame(card_orig, bg=THEME["orig_accent"], height=2).pack(fill=tk.X, pady=(0, 4))
        h_orig = tk.Frame(card_orig, bg=THEME["bg_card"])
        h_orig.pack(fill=tk.X)
        badge_o = tk.Frame(h_orig, bg=THEME["orig_badge_border"], padx=1, pady=1)
        badge_o.pack(side=tk.RIGHT)
        tk.Label(badge_o, text=" ● ENTRADA (CON RUIDO) ", font=(THEME["font_main"], 8, "bold"), fg=THEME["orig_accent"], bg=THEME["orig_badge_bg"], padx=6).pack()
        tk.Label(h_orig, text="IMAGEN DE ENTRADA", font=(THEME["font_main"], 9, "bold"), fg=THEME["text_primary"], bg=THEME["bg_card"]).pack(side=tk.LEFT)

        self.lbl_dim_orig_ruido = tk.Label(card_orig, text="0 x 0 px", font=(THEME["font_main"], 8), fg=THEME["text_muted"], bg=THEME["bg_card"])
        self.lbl_dim_orig_ruido.pack(anchor="w", pady=(0, 2))
        self.canvas_orig_ruido = tk.Canvas(card_orig, bg=THEME["bg_card_inner"], highlightthickness=0, bd=0)
        self.canvas_orig_ruido.pack(fill=tk.BOTH, expand=True)
        self.canvas_orig_ruido.bind("<Configure>", lambda e: self._redibujar_ruido())

        card_salida = tk.Frame(derecha_col, bg=THEME["bg_card"], padx=10, pady=6, highlightthickness=1, highlightbackground=THEME["border"])
        card_salida.grid(row=1, column=0, sticky="nsew", pady=(5, 0))
        tk.Frame(card_salida, bg=THEME["ruido_accent"], height=2).pack(fill=tk.X, pady=(0, 4))
        h_salida = tk.Frame(card_salida, bg=THEME["bg_card"])
        h_salida.pack(fill=tk.X)
        badge_s = tk.Frame(h_salida, bg=THEME["ruido_badge_border"], padx=1, pady=1)
        badge_s.pack(side=tk.RIGHT)
        tk.Label(badge_s, text=" ● SALIDA (FILTRADA) ", font=(THEME["font_main"], 8, "bold"), fg=THEME["ruido_accent"], bg=THEME["ruido_badge_bg"], padx=6).pack()
        tk.Label(h_salida, text="IMAGEN MITIGADA / RESTAURADA", font=(THEME["font_main"], 9, "bold"), fg=THEME["text_primary"], bg=THEME["bg_card"]).pack(side=tk.LEFT)

        self.lbl_dim_salida_ruido = tk.Label(card_salida, text="0 x 0 px", font=(THEME["font_main"], 8), fg=THEME["text_muted"], bg=THEME["bg_card"])
        self.lbl_dim_salida_ruido.pack(anchor="w", pady=(0, 2))
        self.canvas_salida_ruido = tk.Canvas(card_salida, bg=THEME["bg_card_inner"], highlightthickness=0, bd=0)
        self.canvas_salida_ruido.pack(fill=tk.BOTH, expand=True)
        self.canvas_salida_ruido.bind("<Configure>", lambda e: self._redibujar_ruido())

    # =========================================================================
    # PESTAÑA 3: TRANSFORMADA DE FOURIER 2D (DOMINIO DE LA FRECUENCIA)
    # =========================================================================
    def _construir_pestaña_fourier(self):
        self.frame_fourier = tk.Frame(self.container_principal, bg=THEME["bg_root"])

        sidebar = tk.Frame(self.frame_fourier, bg=THEME["bg_card"], width=330, padx=18, pady=14, highlightthickness=1, highlightbackground=THEME["border"])
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 14))
        sidebar.pack_propagate(False)

        lbl_panel = tk.Label(sidebar, text="🌊 FILTRADO EN FRECUENCIA (FFT)", font=(THEME["font_main"], 10, "bold"), fg=THEME["fourier_accent"], bg=THEME["bg_card"])
        lbl_panel.pack(anchor="w", pady=(0, 12))

        # 1. Tipo de Filtro Frecuencial
        tk.Label(sidebar, text="1. Tipo de Filtro Frecuencial:", font=(THEME["font_main"], 9, "bold"), fg=THEME["text_primary"], bg=THEME["bg_card"]).pack(anchor="w", pady=(2, 3))
        self.var_tipo_fourier = tk.StringVar(value="pasa_bajas")
        self.combo_tipo_fourier = ttk.Combobox(
            sidebar,
            textvariable=self.var_tipo_fourier,
            values=["pasa_bajas", "pasa_altas", "rechazo_banda", "pasa_banda"],
            state="readonly",
            font=(THEME["font_main"], 9)
        )
        self.combo_tipo_fourier.pack(fill=tk.X, pady=(0, 10))
        self.combo_tipo_fourier.bind("<<ComboboxSelected>>", lambda e: self._on_fourier_params_change())

        # 2. Perfil Matemático
        tk.Label(sidebar, text="2. Perfil Matemático:", font=(THEME["font_main"], 9, "bold"), fg=THEME["text_primary"], bg=THEME["bg_card"]).pack(anchor="w", pady=(2, 3))
        self.var_modelo_fourier = tk.StringVar(value="gaussiano")
        self.combo_modelo_fourier = ttk.Combobox(
            sidebar,
            textvariable=self.var_modelo_fourier,
            values=["gaussiano", "butterworth", "ideal"],
            state="readonly",
            font=(THEME["font_main"], 9)
        )
        self.combo_modelo_fourier.pack(fill=tk.X, pady=(0, 10))
        self.combo_modelo_fourier.bind("<<ComboboxSelected>>", lambda e: self._on_fourier_params_change())

        # 3. Frecuencia de Corte D0
        lbl_d0_box = tk.Frame(sidebar, bg=THEME["bg_card"])
        lbl_d0_box.pack(fill=tk.X, pady=(2, 2))
        tk.Label(lbl_d0_box, text="3. Frecuencia de Corte (D₀):", font=(THEME["font_main"], 9, "bold"), fg=THEME["text_primary"], bg=THEME["bg_card"]).pack(side=tk.LEFT)
        self.lbl_d0_val = tk.Label(lbl_d0_box, text="60 px", font=(THEME["font_main"], 9, "bold"), fg=THEME["fourier_accent"], bg=THEME["bg_card_inner"], padx=6, pady=1)
        self.lbl_d0_val.pack(side=tk.RIGHT)

        self.slider_d0 = tk.Scale(sidebar, from_=5, to=250, orient=tk.HORIZONTAL, bg=THEME["bg_card"], fg=THEME["text_primary"], troughcolor=THEME["bg_card_inner"], activebackground=THEME["fourier_accent"], highlightthickness=0, command=self._on_d0_change)
        self.slider_d0.set(60)
        self.slider_d0.pack(fill=tk.X, pady=(0, 6))

        # 4. Orden Butterworth
        lbl_orden_box = tk.Frame(sidebar, bg=THEME["bg_card"])
        lbl_orden_box.pack(fill=tk.X, pady=(2, 2))
        self.lbl_orden = tk.Label(lbl_orden_box, text="4. Orden Butterworth (n):", font=(THEME["font_main"], 9, "bold"), fg=THEME["text_muted"], bg=THEME["bg_card"])
        self.lbl_orden.pack(side=tk.LEFT)
        self.lbl_orden_val = tk.Label(lbl_orden_box, text="2", font=(THEME["font_main"], 9, "bold"), fg=THEME["text_muted"], bg=THEME["border"], padx=6, pady=1)
        self.lbl_orden_val.pack(side=tk.RIGHT)

        self.slider_orden = tk.Scale(sidebar, from_=1, to=5, orient=tk.HORIZONTAL, bg=THEME["bg_card"], fg=THEME["text_primary"], troughcolor=THEME["border"], activebackground=THEME["fourier_accent"], highlightthickness=0, command=self._on_orden_change)
        self.slider_orden.set(2)
        self.slider_orden.config(state=tk.DISABLED)
        self.slider_orden.pack(fill=tk.X, pady=(0, 6))

        # 5. Frecuencia Secundaria D1 (para bandas)
        lbl_d1_box = tk.Frame(sidebar, bg=THEME["bg_card"])
        lbl_d1_box.pack(fill=tk.X, pady=(2, 2))
        self.lbl_d1 = tk.Label(lbl_d1_box, text="5. Frecuencia Banda (D₁):", font=(THEME["font_main"], 9, "bold"), fg=THEME["text_muted"], bg=THEME["bg_card"])
        self.lbl_d1.pack(side=tk.LEFT)
        self.lbl_d1_val = tk.Label(lbl_d1_box, text="120 px", font=(THEME["font_main"], 9, "bold"), fg=THEME["text_muted"], bg=THEME["border"], padx=6, pady=1)
        self.lbl_d1_val.pack(side=tk.RIGHT)

        self.slider_d1 = tk.Scale(sidebar, from_=10, to=350, orient=tk.HORIZONTAL, bg=THEME["bg_card"], fg=THEME["text_primary"], troughcolor=THEME["border"], activebackground=THEME["fourier_accent"], highlightthickness=0, command=self._on_d1_change)
        self.slider_d1.set(120)
        self.slider_d1.config(state=tk.DISABLED)
        self.slider_d1.pack(fill=tk.X, pady=(0, 10))

        # Botón
        self.btn_procesar_fourier = tk.Button(sidebar, text="🌊 APLICAR FILTRO FOURIER", font=(THEME["font_main"], 10, "bold"), bg=THEME["fourier_accent"], fg="#FFFFFF", activebackground=THEME["fourier_accent_hover"], activeforeground="#FFFFFF", relief=tk.FLAT, pady=8, cursor="hand2", command=self.procesar_fourier)
        self.btn_procesar_fourier.pack(fill=tk.X, pady=(2, 10))

        # Métricas de Fourier
        tk.Label(sidebar, text="📊 Métricas del Espectro:", font=(THEME["font_main"], 9, "bold"), fg=THEME["text_secondary"], bg=THEME["bg_card"]).pack(anchor="w")
        self.lbl_metricas_fourier = tk.Label(sidebar, text="Resolución: - x -\nRadio D₀: -\nCorrección MAE: -\nTiempo FFT: -", font=(THEME["font_main"], 8), fg=THEME["text_primary"], bg=THEME["bg_card_inner"], justify=tk.LEFT, padx=10, pady=8, relief=tk.FLAT)
        self.lbl_metricas_fourier.pack(fill=tk.X, pady=(4, 0))

        # Visores Central y Derecho
        visores_frame = tk.Frame(self.frame_fourier, bg=THEME["bg_root"])
        visores_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        visores_frame.columnconfigure(0, weight=1, uniform="layout_cols_fourier")
        visores_frame.columnconfigure(1, weight=1, uniform="layout_cols_fourier")
        visores_frame.rowconfigure(0, weight=1)

        # Panel Central: Máscara H(u,v) y Espectro de Magnitud
        central_col = tk.Frame(visores_frame, bg=THEME["bg_root"])
        central_col.grid(row=0, column=0, sticky="nsew", padx=(0, 7))
        central_col.rowconfigure(0, weight=1)
        central_col.rowconfigure(1, weight=1)
        central_col.columnconfigure(0, weight=1)

        # Máscara H(u,v)
        card_mask = tk.Frame(central_col, bg=THEME["bg_card"], padx=10, pady=6, highlightthickness=1, highlightbackground=THEME["border"])
        card_mask.grid(row=0, column=0, sticky="nsew", pady=(0, 5))
        tk.Frame(card_mask, bg=THEME["fourier_accent"], height=2).pack(fill=tk.X, pady=(0, 4))
        h_mask = tk.Frame(card_mask, bg=THEME["bg_card"])
        h_mask.pack(fill=tk.X)
        badge_m = tk.Frame(h_mask, bg=THEME["fourier_badge_border"], padx=1, pady=1)
        badge_m.pack(side=tk.RIGHT)
        self.lbl_badge_fourier_mask = tk.Label(badge_m, text=" ● MÁSCARA H(u,v) ", font=(THEME["font_main"], 8, "bold"), fg=THEME["fourier_accent"], bg=THEME["fourier_badge_bg"], padx=6).pack()
        tk.Label(h_mask, text="MÁSCARA EN FRECUENCIA H(u,v)", font=(THEME["font_main"], 9, "bold"), fg=THEME["text_primary"], bg=THEME["bg_card"]).pack(side=tk.LEFT)

        self.lbl_desc_mask_fourier = tk.Label(card_mask, text="⚪ Puntos blancos: Frecuencias que pasan  •  ⚫ Puntos negros: Frecuencias bloqueadas", font=(THEME["font_main"], 7), fg=THEME["text_muted"], bg=THEME["bg_card"])
        self.lbl_desc_mask_fourier.pack(anchor="w", pady=(0, 2))
        self.canvas_mask_fourier = tk.Canvas(card_mask, bg="#000000", highlightthickness=0, bd=0)
        self.canvas_mask_fourier.pack(fill=tk.BOTH, expand=True)
        self.canvas_mask_fourier.bind("<Configure>", lambda e: self._redibujar_fourier())

        # Espectro de Magnitud
        card_esp = tk.Frame(central_col, bg=THEME["bg_card"], padx=10, pady=6, highlightthickness=1, highlightbackground=THEME["border"])
        card_esp.grid(row=1, column=0, sticky="nsew", pady=(5, 0))
        tk.Frame(card_esp, bg=THEME["fourier_accent"], height=2).pack(fill=tk.X, pady=(0, 4))
        h_esp = tk.Frame(card_esp, bg=THEME["bg_card"])
        h_esp.pack(fill=tk.X)
        badge_e = tk.Frame(h_esp, bg=THEME["fourier_badge_border"], padx=1, pady=1)
        badge_e.pack(side=tk.RIGHT)
        tk.Label(badge_e, text=" ● ESPECTRO FFT ", font=(THEME["font_main"], 8, "bold"), fg=THEME["fourier_accent"], bg=THEME["fourier_badge_bg"], padx=6).pack()
        tk.Label(h_esp, text="ESPECTRO DE MAGNITUD log(1 + |F|)", font=(THEME["font_main"], 9, "bold"), fg=THEME["text_primary"], bg=THEME["bg_card"]).pack(side=tk.LEFT)

        self.lbl_desc_esp_fourier = tk.Label(card_esp, text="Centro = Bajas frecuencias (estructura)  •  Periferia = Altas frecuencias (ruido y aristas)", font=(THEME["font_main"], 7), fg=THEME["text_muted"], bg=THEME["bg_card"])
        self.lbl_desc_esp_fourier.pack(anchor="w", pady=(0, 2))
        self.canvas_esp_fourier = tk.Canvas(card_esp, bg="#000000", highlightthickness=0, bd=0)
        self.canvas_esp_fourier.pack(fill=tk.BOTH, expand=True)
        self.canvas_esp_fourier.bind("<Configure>", lambda e: self._redibujar_fourier())

        # Panel Derecho: Entrada Espacial vs Salida Reconstruida (IFFT2)
        derecha_col = tk.Frame(visores_frame, bg=THEME["bg_root"])
        derecha_col.grid(row=0, column=1, sticky="nsew", padx=(7, 0))
        derecha_col.rowconfigure(0, weight=1)
        derecha_col.rowconfigure(1, weight=1)
        derecha_col.columnconfigure(0, weight=1)

        # Entrada
        card_orig = tk.Frame(derecha_col, bg=THEME["bg_card"], padx=10, pady=6, highlightthickness=1, highlightbackground=THEME["border"])
        card_orig.grid(row=0, column=0, sticky="nsew", pady=(0, 5))
        tk.Frame(card_orig, bg=THEME["orig_accent"], height=2).pack(fill=tk.X, pady=(0, 4))
        h_orig = tk.Frame(card_orig, bg=THEME["bg_card"])
        h_orig.pack(fill=tk.X)
        badge_o = tk.Frame(h_orig, bg=THEME["orig_badge_border"], padx=1, pady=1)
        badge_o.pack(side=tk.RIGHT)
        tk.Label(badge_o, text=" ● ENTRADA ", font=(THEME["font_main"], 8, "bold"), fg=THEME["orig_accent"], bg=THEME["orig_badge_bg"], padx=6).pack()
        tk.Label(h_orig, text="IMAGEN ENTRADA (ESPACIAL)", font=(THEME["font_main"], 9, "bold"), fg=THEME["text_primary"], bg=THEME["bg_card"]).pack(side=tk.LEFT)

        self.lbl_dim_orig_fourier = tk.Label(card_orig, text="0 x 0 px", font=(THEME["font_main"], 8), fg=THEME["text_muted"], bg=THEME["bg_card"])
        self.lbl_dim_orig_fourier.pack(anchor="w", pady=(0, 2))
        self.canvas_orig_fourier = tk.Canvas(card_orig, bg=THEME["bg_card_inner"], highlightthickness=0, bd=0)
        self.canvas_orig_fourier.pack(fill=tk.BOTH, expand=True)
        self.canvas_orig_fourier.bind("<Configure>", lambda e: self._redibujar_fourier())

        # Salida
        card_salida = tk.Frame(derecha_col, bg=THEME["bg_card"], padx=10, pady=6, highlightthickness=1, highlightbackground=THEME["border"])
        card_salida.grid(row=1, column=0, sticky="nsew", pady=(5, 0))
        tk.Frame(card_salida, bg=THEME["fourier_accent"], height=2).pack(fill=tk.X, pady=(0, 4))
        h_salida = tk.Frame(card_salida, bg=THEME["bg_card"])
        h_salida.pack(fill=tk.X)
        badge_s = tk.Frame(h_salida, bg=THEME["fourier_badge_border"], padx=1, pady=1)
        badge_s.pack(side=tk.RIGHT)
        tk.Label(badge_s, text=" ● SALIDA IFFT2 ", font=(THEME["font_main"], 8, "bold"), fg=THEME["fourier_accent"], bg=THEME["fourier_badge_bg"], padx=6).pack()
        tk.Label(h_salida, text="RECONSTRUCCIÓN FOURIER IFFT2", font=(THEME["font_main"], 9, "bold"), fg=THEME["text_primary"], bg=THEME["bg_card"]).pack(side=tk.LEFT)

        self.lbl_dim_salida_fourier = tk.Label(card_salida, text="0 x 0 px", font=(THEME["font_main"], 8), fg=THEME["text_muted"], bg=THEME["bg_card"])
        self.lbl_dim_salida_fourier.pack(anchor="w", pady=(0, 2))
        self.canvas_salida_fourier = tk.Canvas(card_salida, bg=THEME["bg_card_inner"], highlightthickness=0, bd=0)
        self.canvas_salida_fourier.pack(fill=tk.BOTH, expand=True)
        self.canvas_salida_fourier.bind("<Configure>", lambda e: self._redibujar_fourier())

    # =========================================================================
    # EVENTOS DE PESTAÑA 3: FOURIER
    # =========================================================================
    def _on_d0_change(self, val):
        self.lbl_d0_val.config(text=f"{int(val)} px")

    def _on_orden_change(self, val):
        self.lbl_orden_val.config(text=f"{int(val)}")

    def _on_d1_change(self, val):
        self.lbl_d1_val.config(text=f"{int(val)} px")

    def _on_fourier_params_change(self):
        tipo = self.var_tipo_fourier.get()
        modelo = self.var_modelo_fourier.get()

        es_butter = (modelo == "butterworth")
        es_banda = (tipo in ("rechazo_banda", "pasa_banda"))

        if es_butter:
            self.slider_orden.config(state=tk.NORMAL, fg=THEME["text_primary"], troughcolor=THEME["bg_card_inner"])
            self.lbl_orden.config(fg=THEME["text_primary"])
            self.lbl_orden_val.config(fg=THEME["fourier_accent"], bg=THEME["bg_card_inner"])
        else:
            self.slider_orden.config(state=tk.DISABLED, fg=THEME["border_highlight"], troughcolor=THEME["border"])
            self.lbl_orden.config(fg=THEME["text_muted"])
            self.lbl_orden_val.config(fg=THEME["text_muted"], bg=THEME["border"])

        if es_banda:
            self.slider_d1.config(state=tk.NORMAL, fg=THEME["text_primary"], troughcolor=THEME["bg_card_inner"])
            self.lbl_d1.config(fg=THEME["text_primary"])
            self.lbl_d1_val.config(fg=THEME["fourier_accent"], bg=THEME["bg_card_inner"])
        else:
            self.slider_d1.config(state=tk.DISABLED, fg=THEME["border_highlight"], troughcolor=THEME["border"])
            self.lbl_d1.config(fg=THEME["text_muted"])
            self.lbl_d1_val.config(fg=THEME["text_muted"], bg=THEME["border"])

        if tipo == "pasa_bajas":
            self.lbl_desc_mask_fourier.config(text="⚪ Centro blanco (pasan bajas frecuencias) • ⚫ Periferia negra (se elimina ruido de alta frecuencia)")
        elif tipo == "pasa_altas":
            self.lbl_desc_mask_fourier.config(text="⚫ Centro negro (bloquea bajas frecuencias) • ⚪ Periferia blanca (pasan altas frecuencias y bordes)")
        else:
            self.lbl_desc_mask_fourier.config(text="Anillo frecuencial: Filtra o aísla el rango de frecuencias comprendido entre D₀ y D₁")

        self.procesar_fourier()

    # =========================================================================
    # CONTROLADORES DE EVENTOS Y PROCESAMIENTO
    # =========================================================================
    def _on_stride_red_change(self, val):
        stride = int(val)
        self.lbl_stride_red_val.config(text=f"{stride} px (1/{stride})")

    def _on_sigma_red_change(self, val):
        sigma = float(val)
        self.lbl_sigma_red_val.config(text=f"{sigma:.1f}")
        self._actualizar_mascara_red_info()

    def _on_stride_ruido_change(self, val):
        stride = int(val)
        if stride == 1:
            self.lbl_stride_ruido_val.config(text="1 px (Sin reducción)")
        else:
            self.lbl_stride_ruido_val.config(text=f"{stride} px (1/{stride})")

    def _on_sigma_ruido_change(self, val):
        sigma = float(val)
        self.lbl_sigma_ruido_val.config(text=f"{sigma:.1f}")
        self._actualizar_info_filtro_ruido()

    def _actualizar_mascara_red_y_procesar(self):
        self._actualizar_mascara_red_info()
        self.procesar_reduccion()

    def _actualizar_info_filtro_ruido_y_procesar(self):
        self._actualizar_info_filtro_ruido()
        self.procesar_mitigacion_ruido()

    def _actualizar_mascara_red_info(self):
        tipo = self.var_tipo_red.get()
        es_gauss = (tipo == "gaussiano")
        es_personalizado = (tipo == "personalizado")

        if es_gauss:
            self.slider_sigma_red.config(state=tk.NORMAL, fg=THEME["text_primary"], troughcolor=THEME["bg_card_inner"])
            self.lbl_sigma_red.config(fg=THEME["text_primary"])
            self.lbl_sigma_red_val.config(text=f"{float(self.slider_sigma_red.get()):.1f}", fg=THEME["text_accent"], bg=THEME["bg_card_inner"])
            self.lbl_sigma_red_hint.config(text="Control activo para la campana de Gauss", fg=THEME["text_muted"])
        else:
            self.slider_sigma_red.config(state=tk.DISABLED, fg=THEME["border_highlight"], troughcolor=THEME["border"])
            self.lbl_sigma_red.config(fg=THEME["text_muted"])
            self.lbl_sigma_red_val.config(text="N/A", fg=THEME["text_muted"], bg=THEME["border"])
            self.lbl_sigma_red_hint.config(text="⚠️ Inactivo: solo aplica para filtro Gaussiano", fg="#D97706")

        if es_personalizado:
            self.txt_matriz_red.config(state=tk.NORMAL)
            self.btn_aplicar_matriz_red.config(state=tk.NORMAL, bg=THEME["accent_primary"], fg="#FFFFFF")
            self.btn_norm_matriz_red.config(state=tk.NORMAL, bg=THEME["bg_card_inner"], fg=THEME["text_primary"])
            self.lbl_estado_matriz_red.config(text="✏️ Modo Personalizado: Edita valores y pulsa 'Aplicar'.", fg=THEME["accent_primary"])
            return

        try:
            tam_str = self.var_mascara_red.get()
            self.lbl_badge_tam_mascara_red.config(text=f" ● KERNEL {tam_str} ")
            tam = int(tam_str.split("x")[0])
            sigma = float(self.slider_sigma_red.get())

            k = generar_mascara(tipo, tam, sigma)

            self.txt_matriz_red.config(state=tk.NORMAL)
            self.txt_matriz_red.delete("1.0", tk.END)

            lineas = []
            for fila in k:
                lineas.append("  ".join(f"{v:6.4f}" for v in fila))
            self.txt_matriz_red.insert(tk.END, "\n".join(lineas))

            self.txt_matriz_red.config(state=tk.DISABLED)
            self.btn_aplicar_matriz_red.config(state=tk.DISABLED, bg="#E2E8F0", disabledforeground="#64748B")
            self.btn_norm_matriz_red.config(state=tk.DISABLED, bg="#F1F5F9", disabledforeground="#94A3B8")
            self.lbl_estado_matriz_red.config(text="🔒 Solo lectura: Selecciona 'personalizado' para editar coeficientes.", fg=THEME["text_muted"])
        except Exception:
            pass

    def _actualizar_info_filtro_ruido(self):
        tipo = self.var_tipo_ruido.get()
        tam_str = self.var_mascara_ruido.get()
        tam = int(tam_str.split("x")[0])
        es_gauss = (tipo == "gaussiano")

        if es_gauss:
            self.slider_sigma_ruido.config(state=tk.NORMAL, fg=THEME["text_primary"], troughcolor=THEME["bg_card_inner"])
            self.lbl_sigma_ruido.config(fg=THEME["text_primary"])
            self.lbl_sigma_ruido_val.config(text=f"{float(self.slider_sigma_ruido.get()):.1f}", fg=THEME["ruido_accent"], bg=THEME["bg_card_inner"])
            self.lbl_sigma_ruido_hint.config(text="Control activo para calibrar el suavizado gaussiano", fg=THEME["text_muted"])
        else:
            self.slider_sigma_ruido.config(state=tk.DISABLED, fg=THEME["border_highlight"], troughcolor=THEME["border"])
            self.lbl_sigma_ruido.config(fg=THEME["text_muted"])
            self.lbl_sigma_ruido_val.config(text="N/A", fg=THEME["text_muted"], bg=THEME["border"])
            self.lbl_sigma_ruido_hint.config(text="⚠️ Inactivo: solo aplica para filtro Gaussiano", fg="#D97706")

        self.lbl_badge_filtro_ruido.config(text=f" ● FILTRO {tipo.upper()} {tam_str} ")

        if tipo == "media":
            desc = (
                "🔹 Filtro de la Media (Box Blur):\n"
                f"Promedia uniformemente los {tam * tam} píxeles de la ventana con pesos 1/{tam * tam}.\n"
                "• Eficacia: Reduce la varianza del ruido gaussiano homogéneo de media cero.\n"
                "• Efecto: Produce un suavizado uniforme en toda la imagen."
            )
            val = 1.0 / (tam * tam)
            lineas = ["  ".join(f"{val:6.4f}" for _ in range(tam)) for _ in range(tam)]
            self.lbl_estado_ruido_diag.config(text=f"Filtro Lineal Uniforme ({tam}x{tam}) • Suma de pesos = 1.000", fg=THEME["text_muted"])
        elif tipo == "moda":
            desc = (
                "🔹 Filtro de Moda (Estadístico de Frecuencia Local):\n"
                f"En cada ventana {tam}x{tam}, selecciona el valor de color que más se repite.\n"
                "• Eficacia: Suprime valores anómalos o atípicos generados por el ruido.\n"
                "• Efecto: Preserva zonas tonales dominantes y bordes con desempate inteligente."
            )
            lineas = ["  ".join(" 1.0 " for _ in range(tam)) for _ in range(tam)]
            self.lbl_estado_ruido_diag.config(text=f"Filtro No Lineal Estadístico ({tam}x{tam}) • Consenso por frecuencia", fg=THEME["ruido_accent"])
        else:
            sigma = float(self.slider_sigma_ruido.get())
            desc = (
                f"🔹 Filtro Gaussiano 2D (σ={sigma:.1f}):\n"
                f"Convoluciona con una campana normal 2D sobre la vecindad {tam}x{tam}.\n"
                "• Eficacia: Atenúa el ruido ponderando los píxeles según la distancia al centro.\n"
                "• Efecto: Mitiga el ruido preservando bordes principales con mayor nitidez que la media."
            )
            k = generar_mascara("gaussiano", tam, sigma)
            lineas = ["  ".join(f"{v:6.4f}" for v in fila) for fila in k]
            self.lbl_estado_ruido_diag.config(text=f"Filtro Lineal Ponderado ({tam}x{tam}, σ={sigma:.1f}) • Suma = 1.000", fg=THEME["text_muted"])

        self.card_info_ruido.config(text=desc)

        self.txt_matriz_ruido.config(state=tk.NORMAL)
        self.txt_matriz_ruido.delete("1.0", tk.END)
        self.txt_matriz_ruido.insert(tk.END, "\n".join(lineas))
        self.txt_matriz_ruido.config(state=tk.DISABLED)

    # =========================================================================
    # LÓGICA DE PROCESAMIENTO
    # =========================================================================
    def procesar_reduccion(self, kernel_override: Optional[np.ndarray] = None):
        if self.img_original is None:
            return

        try:
            stride = int(self.slider_stride_red.get())
            tipo_filtro = self.var_tipo_red.get()
            sigma = float(self.slider_sigma_red.get())

            if kernel_override is not None:
                kernel = kernel_override
                tam_mascara = kernel.shape[0]
            elif tipo_filtro == "personalizado":
                kernel = self._obtener_matriz_red_desde_texto()
                tam_mascara = kernel.shape[0]
            else:
                tam_str = self.var_mascara_red.get()
                tam_mascara = int(tam_str.split("x")[0])
                kernel = None

            t0 = time.time()
            self.img_reducida = reducir_imagen(
                self.img_original,
                tam_mascara=tam_mascara,
                tipo_filtro=tipo_filtro,
                stride=stride,
                sigma=sigma,
                kernel_personalizado=kernel
            )
            t_ms = (time.time() - t0) * 1000.0

            h_r, w_r = self.img_reducida.shape[:2]
            self.lbl_dim_salida_red.config(text=f"Resolución: {w_r} × {h_r} px  •  {w_r * h_r:,} píxeles")

            metricas = calcular_metricas(self.img_original, self.img_reducida, stride, tam_mascara)
            texto_metricas = (
                f"Original: {metricas['ancho_orig']}x{metricas['alto_orig']} ({metricas['pixeles_orig']:,} px)\n"
                f"Reducida: {metricas['ancho_red']}x{metricas['alto_red']} ({metricas['pixeles_red']:,} px)\n"
                f"Factor Escalar: {metricas['factor_reduccion']:.2f}x\n"
                f"Reducción: {metricas['porcentaje_reduccion']:.1f}%\n"
                f"Tiempo: {t_ms:.1f} ms"
            )
            self.lbl_metricas_red.config(text=texto_metricas)

            if self.pestaña_activa == "reduccion":
                self.lbl_metricas_footer.config(
                    text=f"Reducción {metricas['ancho_orig']}×{metricas['alto_orig']} → {w_r}×{h_r} px  •  Salto: {stride} px  •  {t_ms:.1f} ms"
                )

            self._redibujar_reduccion()

        except Exception as e:
            messagebox.showerror("Error en Reducción", f"Ocurrió un error al procesar:\n{str(e)}")

    def procesar_mitigacion_ruido(self):
        if self.img_original is None:
            return

        try:
            stride = int(self.slider_stride_ruido.get())
            tipo_filtro = self.var_tipo_ruido.get()
            sigma = float(self.slider_sigma_ruido.get())
            tam_str = self.var_mascara_ruido.get()
            tam_mascara = int(tam_str.split("x")[0])

            t0 = time.time()
            self.img_filtrada_ruido = reducir_imagen(
                self.img_original,
                tam_mascara=tam_mascara,
                tipo_filtro=tipo_filtro,
                stride=stride,
                sigma=sigma
            )
            t_ms = (time.time() - t0) * 1000.0

            h_f, w_f = self.img_filtrada_ruido.shape[:2]
            self.lbl_dim_salida_ruido.config(text=f"Resolución: {w_f} × {h_f} px  •  {w_f * h_f:,} píxeles")

            metricas_ruido = calcular_metricas_ruido(self.img_original, self.img_filtrada_ruido, stride, tam_mascara, tipo_filtro)

            txt_mae = f"{metricas_ruido['mae_correccion']:.2f} niveles" if stride == 1 else "N/A (hubo reducción)"
            txt_max = f"{metricas_ruido['max_correccion']:.1f} niveles" if stride == 1 else "N/A"

            texto_metricas = (
                f"Resolución: {w_f} × {h_f} px\n"
                f"Filtro: {tipo_filtro.capitalize()} ({tam_mascara}x{tam_mascara})\n"
                f"Corrección Media (MAE): {txt_mae}\n"
                f"Corrección Máxima: {txt_max}\n"
                f"Tiempo: {t_ms:.1f} ms"
            )
            self.lbl_metricas_ruido.config(text=texto_metricas)

            if self.pestaña_activa == "ruido":
                self.lbl_metricas_footer.config(
                    text=f"Ruido {tipo_filtro.capitalize()} {tam_mascara}x{tam_mascara}  •  Salida: {w_f}×{h_f} px  •  MAE: {txt_mae}  •  {t_ms:.1f} ms"
                )

            self._redibujar_ruido()

        except Exception as e:
            messagebox.showerror("Error en Mitigación de Ruido", f"Ocurrió un error al filtrar ruido:\n{str(e)}")

    def procesar_fourier(self):
        if self.img_original is None:
            return

        try:
            tipo_filtro = self.var_tipo_fourier.get()
            modelo = self.var_modelo_fourier.get()
            d0 = float(self.slider_d0.get())
            d1 = float(self.slider_d1.get())
            orden = int(self.slider_orden.get())

            self.resultado_fourier = aplicar_filtro_fourier(
                self.img_original,
                tipo_filtro=tipo_filtro,
                modelo=modelo,
                d0=d0,
                d1=d1,
                orden=orden
            )

            h, w = self.img_original.shape[:2]
            txt_metricas = (
                f"Resolución: {w} × {h} px\n"
                f"Filtro: {tipo_filtro.replace('_', ' ').capitalize()}\n"
                f"Perfil: {modelo.capitalize()}\n"
                f"Corte (D₀): {d0:.0f} px\n"
                f"Corrección MAE: {self.resultado_fourier['mae_correccion']:.2f}\n"
                f"Tiempo FFT: {self.resultado_fourier['tiempo_ms']:.1f} ms"
            )
            self.lbl_metricas_fourier.config(text=txt_metricas)

            if self.pestaña_activa == "fourier":
                self.lbl_metricas_footer.config(
                    text=f"Fourier {tipo_filtro} ({modelo}, D₀={d0:.0f}px)  •  MAE: {self.resultado_fourier['mae_correccion']:.2f}  •  {self.resultado_fourier['tiempo_ms']:.1f} ms"
                )

            self._redibujar_fourier()

        except Exception as e:
            messagebox.showerror("Error en Filtro Fourier", f"Ocurrió un error en el filtrado frecuencial:\n{str(e)}")

    # =========================================================================
    # RENDERIZADO Y DIBUJO EN CANVAS
    # =========================================================================
    def _redibujar_reduccion(self):
        if self.pestaña_activa != "reduccion":
            return
        if self.img_original is not None:
            self._mostrar_en_canvas(self.canvas_orig_red, self.img_original, "_tk_orig_red")
        if self.img_reducida is not None:
            self._mostrar_en_canvas(self.canvas_salida_red, self.img_reducida, "_tk_salida_red")

    def _redibujar_ruido(self):
        if self.pestaña_activa != "ruido":
            return
        if self.img_original is not None:
            self._mostrar_en_canvas(self.canvas_orig_ruido, self.img_original, "_tk_orig_ruido")
        if self.img_filtrada_ruido is not None:
            self._mostrar_en_canvas(self.canvas_salida_ruido, self.img_filtrada_ruido, "_tk_salida_ruido")

    def _redibujar_fourier(self):
        if self.pestaña_activa != "fourier":
            return
        if self.img_original is not None:
            self._mostrar_en_canvas(self.canvas_orig_fourier, self.img_original, "_tk_orig_four")
        if self.resultado_fourier is not None:
            self._mostrar_en_canvas(self.canvas_salida_fourier, self.resultado_fourier["imagen_salida"], "_tk_salida_four")
            self._mostrar_en_canvas(self.canvas_mask_fourier, self.resultado_fourier["mascara_visual"], "_tk_mask_four")
            self._mostrar_en_canvas(self.canvas_esp_fourier, self.resultado_fourier["espectro_orig_visual"], "_tk_esp_four")

    def _mostrar_en_canvas(self, canvas: tk.Canvas, arr: np.ndarray, attr_ref: str):
        canvas.update_idletasks()
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()

        if cw <= 1 or ch <= 1:
            cw, ch = 480, 400

        im_pil = Image.fromarray(arr)
        img_w, img_h = im_pil.size

        escala = min((cw - 16) / img_w, (ch - 16) / img_h, 1.0)
        nuevo_w = max(1, int(img_w * escala))
        nuevo_h = max(1, int(img_h * escala))

        im_resized = im_pil.resize((nuevo_w, nuevo_h), Image.Resampling.LANCZOS)
        tk_img = ImageTk.PhotoImage(im_resized)

        setattr(self, attr_ref, tk_img)
        canvas.delete("all")
        canvas.create_image(cw // 2, ch // 2, image=tk_img, anchor=tk.CENTER)

    # =========================================================================
    # CARGA Y EXPORTACIÓN DE ARCHIVOS
    # =========================================================================
    def _cargar_muestra_defecto(self):
        muestras = listar_imagenes_muestra(self.dir_img_orig)
        if muestras:
            self._abrir_archivo(os.path.join(self.dir_img_orig, muestras[0]))

    def _on_muestra_seleccionada(self, _):
        muestra = self.combo_muestras.get()
        if muestra and not muestra.startswith("--"):
            ruta = os.path.join(self.dir_img_orig, muestra)
            if os.path.exists(ruta):
                self._abrir_archivo(ruta)

    def _seleccionar_imagen(self):
        tipos = [
            ("Archivos de imagen", "*.jpg *.jpeg *.png *.bmp *.webp *.tiff"),
            ("Todos los archivos", "*.*")
        ]
        ruta = filedialog.askopenfilename(
            initialdir=self.dir_img_orig,
            title="Seleccionar Imagen",
            filetypes=tipos
        )
        if ruta:
            nombre = os.path.basename(ruta)
            if hasattr(self, "combo_muestras") and nombre in self.combo_muestras["values"]:
                self.combo_muestras.set(nombre)
            self._abrir_archivo(ruta)

    def _abrir_archivo(self, ruta: str):
        try:
            self.img_original = cargar_imagen(ruta)
            self.ruta_imagen_actual = ruta
            h, w = self.img_original.shape[:2]
            txt_dim = f"Resolución: {w} × {h} px  •  {w * h:,} píxeles"

            self.lbl_dim_orig_red.config(text=txt_dim)
            self.lbl_dim_orig_ruido.config(text=txt_dim)
            self.lbl_dim_orig_fourier.config(text=txt_dim)
            self.lbl_dim_salida_fourier.config(text=txt_dim)
            self.lbl_estado.config(text=f"Cargada: {os.path.basename(ruta)}")

            self._actualizar_mascara_red_info()
            self._actualizar_info_filtro_ruido()

            self.procesar_reduccion()
            self.procesar_mitigacion_ruido()
            self.procesar_fourier()

        except Exception as e:
            messagebox.showerror("Error al cargar imagen", f"No se pudo cargar la imagen:\n{str(e)}")

    def _guardar_imagen_activa(self):
        if self.pestaña_activa == "reduccion":
            img_guardar = self.img_reducida
            subfijo = f"reducida_s{self.slider_stride_red.get()}_m{self.var_mascara_red.get()}"
        elif self.pestaña_activa == "ruido":
            img_guardar = self.img_filtrada_ruido
            subfijo = f"filtrada_{self.var_tipo_ruido.get()}_m{self.var_mascara_ruido.get()}"
        else:
            if self.resultado_fourier is None:
                img_guardar = None
            else:
                img_guardar = self.resultado_fourier["imagen_salida"]
            tipo = self.var_tipo_fourier.get()
            mod = self.var_modelo_fourier.get()
            d0 = int(self.slider_d0.get())
            subfijo = f"fourier_{tipo}_{mod}_D{d0}"

        if img_guardar is None:
            messagebox.showwarning("Atención", "No hay imagen procesada en esta pestaña para guardar.")
            return

        base = "imagen"
        ext = ".png"
        if self.ruta_imagen_actual:
            base, ext = os.path.splitext(os.path.basename(self.ruta_imagen_actual))

        nombre_defecto = f"{base}_{subfijo}{ext}"

        ruta = filedialog.asksaveasfilename(
            title="Guardar Imagen Procesada",
            initialfile=nombre_defecto,
            initialdir=self.dir_img_red,
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("Todos los archivos", "*.*")]
        )

        if ruta:
            try:
                guardar_imagen(img_guardar, ruta)
                messagebox.showinfo("Éxito", f"Imagen guardada correctamente en:\n{ruta}")
                self.lbl_estado.config(text=f"Guardada: {ruta}")
            except Exception as e:
                messagebox.showerror("Error al guardar", f"No se pudo guardar:\n{str(e)}")

    # =========================================================================
    # MÉTODOS MATRICIALES (PESTAÑA 1)
    # =========================================================================
    def _obtener_matriz_red_desde_texto(self) -> Optional[np.ndarray]:
        contenido = self.txt_matriz_red.get("1.0", tk.END).strip()
        if not contenido:
            raise ValueError("El editor de matriz está vacío.")

        lineas = [l.strip() for l in contenido.splitlines() if l.strip()]
        if not lineas:
            raise ValueError("No se encontraron filas con valores numéricos.")

        filas = []
        for l in lineas:
            partes = l.replace(",", " ").split()
            if partes:
                filas.append([float(p) for p in partes])

        n_filas = len(filas)
        for idx, fila in enumerate(filas):
            if len(fila) != n_filas:
                raise ValueError(f"La matriz debe ser cuadrada. Fila {idx + 1} tiene {len(fila)} columnas, esperaba {n_filas}.")

        if n_filas % 2 == 0:
            raise ValueError(f"La dimensión debe ser impar (3, 5, 7...), actualmente es {n_filas}x{n_filas}.")

        return np.array(filas, dtype=np.float64)

    def _aplicar_matriz_editada_red(self):
        try:
            matriz = self._obtener_matriz_red_desde_texto()
            tam = matriz.shape[0]
            self.var_tipo_red.set("personalizado")
            tam_str = f"{tam}x{tam}"
            if tam_str in self.combo_mascara_red["values"]:
                self.var_mascara_red.set(tam_str)
            self._actualizar_mascara_red_info()
            self.lbl_badge_tam_mascara_red.config(text=f" ● PERSONALIZADO {tam_str} ")
            self.lbl_estado_matriz_red.config(text=f"✔ Matriz {tam_str} aplicada (Suma = {np.sum(matriz):.3f})", fg=THEME["accent_success"])
            self.procesar_reduccion(kernel_override=matriz)
        except Exception as e:
            messagebox.showerror("Error en Matriz", str(e))

    def _normalizar_matriz_actual_red(self):
        try:
            matriz = self._obtener_matriz_red_desde_texto()
            suma = np.sum(matriz)
            if abs(suma) < 1e-9:
                messagebox.showwarning("Atención", "La suma es cero; no se puede normalizar.")
                return
            matriz_norm = matriz / suma
            self.txt_matriz_red.delete("1.0", tk.END)
            self.txt_matriz_red.insert(tk.END, "\n".join("  ".join(f"{v:6.4f}" for v in f) for f in matriz_norm))
            self.lbl_estado_matriz_red.config(text="✔ Matriz normalizada (Suma total = 1.000).", fg=THEME["accent_primary"])
        except Exception as e:
            messagebox.showerror("Error al normalizar", str(e))

    def _restaurar_matriz_red(self):
        if self.var_tipo_red.get() == "personalizado":
            self.var_tipo_red.set("gaussiano")
        self._actualizar_mascara_red_y_procesar()


def iniciar_interfaz():
    root = tk.Tk()
    app = InterfazReduccion(root)
    root.mainloop()


if __name__ == "__main__":
    iniciar_interfaz()
