"""
Módulo de Lógica: Reducción de Imágenes por Filtrado Espacial (Máscara/Kernel) y Submuestreo (Stride).
NO utiliza imresize ni cv2.resize. Implementación matricial y algorítmica pura usando NumPy.
"""

import os
import time
from typing import Tuple, Optional
import numpy as np
from PIL import Image


def generar_mascara(tipo: str, tamano: int, sigma: float = 1.0) -> np.ndarray:
    """
    Genera una máscara/kernel normalizada de tamaño tamano x tamano (ej. 3x3, 5x5, 7x7).
    Tipos soportados:
      - 'media': Promedio aritmético uniforme (box blur) con pesos 1/(tamano^2).
      - 'gaussiano': Filtro gaussiano 2D simétrico con varianza sigma.
      - 'moda': Indicador simbólico de vecindad para filtro estadístico de frecuencia.
      - 'ponderado': Kernel de pesos piramidales/binomios.
      - 'personalizado': Matriz definida por el usuario.
    """
    if tamano % 2 == 0:
        raise ValueError("El tamaño de la máscara debe ser impar (ej. 3, 5, 7).")

    if tipo == "media":
        mascara = np.ones((tamano, tamano), dtype=np.float64) / (tamano * tamano)
        return mascara

    elif tipo == "gaussiano":
        radio = tamano // 2
        y, x = np.ogrid[-radio:radio + 1, -radio:radio + 1]
        # Función gaussiana 2D
        h = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        suma = h.sum()
        if suma != 0:
            h /= suma
        return h

    elif tipo == "ponderado":
        radio = tamano // 2
        y, x = np.ogrid[-radio:radio + 1, -radio:radio + 1]
        dist = np.abs(x) + np.abs(y)
        max_dist = 2 * radio
        pesos = (max_dist + 1) - dist
        pesos = pesos.astype(np.float64)
        return pesos / np.sum(pesos)

    elif tipo in ("moda", "mediana", "maximo", "minimo"):
        # Kernel unitario simbólico representativo de la vecindad de muestreo
        return np.ones((tamano, tamano), dtype=np.float64)

    else:
        # Por defecto media
        return np.ones((tamano, tamano), dtype=np.float64) / (tamano * tamano)


def _calcular_moda_vectorizada(ventanas: np.ndarray) -> np.ndarray:
    """
    Calcula la moda estadística para cada ventana espacial a lo largo de los ejes (-2, -1).
    Utiliza scipy.stats.mode con desempate hacia la mediana cuando no existen valores repetidos,
    evitando oscurecimientos artificiales y preservando la luminosidad natural del vecindario.
    """
    try:
        from scipy.stats import mode as sp_mode
        res = sp_mode(ventanas, axis=(-2, -1), keepdims=False)
        m = res.mode
        c = res.count
        # Si la frecuencia máxima es mayor a 1, existe una moda genuina.
        # Si todos los elementos son distintos (c == 1), se desempata con la mediana
        med = np.median(ventanas, axis=(-2, -1))
        salida = np.where(c > 1, m, med)
        return salida.astype(np.float32)
    except Exception:
        # Fallback sin scipy usando mediana como estimador central robusto
        return np.median(ventanas, axis=(-2, -1)).astype(np.float32)


def reducir_imagen(
    img: np.ndarray,
    tam_mascara: int = 3,
    tipo_filtro: str = "media",
    stride: int = 2,
    sigma: float = 1.0,
    modo_borde: str = "reflect",
    kernel_personalizado: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Reduce las dimensiones de una imagen aplicando una máscara de vecindad
    y un paso de salto (stride) sin usar funciones de redimensionamiento automáticas.

    Parámetros:
      - img: Imagen en formato NumPy uint8 (H, W) o (H, W, C).
      - tam_mascara: Tamaño de la vecindad de píxeles (3, 5, 7, etc.).
      - tipo_filtro: 'media', 'gaussiano', 'mediana', 'maximo', 'minimo', 'ponderado', 'personalizado'.
      - stride: Cuántos píxeles salta en cada paso para muestrear la imagen reducida (ej. 2, 3, 4).
      - sigma: Desviación estándar en caso de filtro gaussiano.
      - modo_borde: 'reflect', 'edge', 'constant'.
      - kernel_personalizado: Matriz NumPy 2D con los pesos editados por el usuario.

    Retorna:
      - np.ndarray de tipo uint8 con la imagen reducida preservando calidad visual.
    """
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    if kernel_personalizado is not None:
        kernel = kernel_personalizado.astype(np.float64)
        tam_mascara = kernel.shape[0]
        # Normalizar si la suma es diferente de cero para evitar saturación
        suma = np.sum(kernel)
        if abs(suma) > 1e-7:
            kernel = kernel / suma
        es_lineal = True
    else:
        if tam_mascara % 2 == 0:
            raise ValueError("El tamaño de la máscara debe ser un número impar (3, 5, 7...).")
        es_lineal = tipo_filtro in ("media", "gaussiano", "ponderado")
        kernel = generar_mascara(tipo_filtro, tam_mascara, sigma) if es_lineal else None

    es_color = (img.ndim == 3 and img.shape[2] in (3, 4))
    alto_orig, ancho_orig = img.shape[:2]
    radio = tam_mascara // 2

    # Padding para manejar los bordes sin pérdida de límites
    if es_color:
        canales = img.shape[2]
        pad_width = ((radio, radio), (radio, radio), (0, 0))
    else:
        canales = 1
        pad_width = ((radio, radio), (radio, radio))

    if modo_borde == "reflect":
        img_pad = np.pad(img, pad_width, mode="reflect")
    elif modo_borde == "edge":
        img_pad = np.pad(img, pad_width, mode="edge")
    else:
        img_pad = np.pad(img, pad_width, mode="constant", constant_values=0)

    # Coordenadas de muestreo con salto (stride) en el espacio original
    # Usar vistas con salto directo sobre la imagen con padding
    alto_salida = (alto_orig + stride - 1) // stride
    ancho_salida = (ancho_orig + stride - 1) // stride

    try:
        from numpy.lib.stride_tricks import sliding_window_view
        # Obtener ventanas deslizantes sin copiar memoria
        if es_color:
            # Ventanas sobre los ejes 0 y 1: forma (alto_orig, ancho_orig, canales, tam, tam)
            ventanas = sliding_window_view(img_pad, (tam_mascara, tam_mascara), axis=(0, 1))[::stride, ::stride]
        else:
            ventanas = sliding_window_view(img_pad, (tam_mascara, tam_mascara), axis=(0, 1))[::stride, ::stride]

        if es_lineal and kernel is not None:
            k_f32 = kernel.astype(np.float32)
            if es_color:
                # Contracción de tensores altamente optimizada en C (BLAS)
                salida = np.einsum("ijklm,lm->ijk", ventanas.astype(np.float32), k_f32)
            else:
                salida = np.einsum("ijlm,lm->ij", ventanas.astype(np.float32), k_f32)

        elif tipo_filtro == "moda":
            salida = _calcular_moda_vectorizada(ventanas)

        elif tipo_filtro == "mediana":
            salida = np.median(ventanas, axis=(-2, -1))

        elif tipo_filtro == "maximo":
            salida = np.max(ventanas, axis=(-2, -1))

        elif tipo_filtro == "minimo":
            salida = np.min(ventanas, axis=(-2, -1))
        else:
            salida = img_pad[radio:radio + alto_orig:stride, radio:radio + ancho_orig:stride]

    except Exception:
        # Fallback seguro con bucles
        filas_muestreo = np.arange(0, alto_orig, stride)
        columnas_muestreo = np.arange(0, ancho_orig, stride)
        if es_color:
            salida = np.zeros((len(filas_muestreo), len(columnas_muestreo), canales), dtype=np.float32)
            for idx_r, r in enumerate(filas_muestreo):
                for idx_c, c in enumerate(columnas_muestreo):
                    parche = img_pad[r:r + tam_mascara, c:c + tam_mascara, :].astype(np.float32)
                    if es_lineal and kernel is not None:
                        salida[idx_r, idx_c, :] = np.sum(parche * kernel[:, :, np.newaxis], axis=(0, 1))
                    elif tipo_filtro == "moda":
                        for c_ch in range(canales):
                            vals, counts = np.unique(parche[:, :, c_ch], return_counts=True)
                            max_idx = counts.argmax()
                            salida[idx_r, idx_c, c_ch] = vals[max_idx] if counts[max_idx] > 1 else np.median(parche[:, :, c_ch])
                    elif tipo_filtro == "mediana":
                        salida[idx_r, idx_c, :] = np.median(parche, axis=(0, 1))
                    elif tipo_filtro == "maximo":
                        salida[idx_r, idx_c, :] = np.max(parche, axis=(0, 1))
                    elif tipo_filtro == "minimo":
                        salida[idx_r, idx_c, :] = np.min(parche, axis=(0, 1))
        else:
            salida = np.zeros((len(filas_muestreo), len(columnas_muestreo)), dtype=np.float32)
            for idx_r, r in enumerate(filas_muestreo):
                for idx_c, c in enumerate(columnas_muestreo):
                    parche = img_pad[r:r + tam_mascara, c:c + tam_mascara].astype(np.float32)
                    if es_lineal and kernel is not None:
                        salida[idx_r, idx_c] = np.sum(parche * kernel)
                    elif tipo_filtro == "moda":
                        vals, counts = np.unique(parche, return_counts=True)
                        max_idx = counts.argmax()
                        salida[idx_r, idx_c] = vals[max_idx] if counts[max_idx] > 1 else np.median(parche)
                    elif tipo_filtro == "mediana":
                        salida[idx_r, idx_c] = np.median(parche)
                    elif tipo_filtro == "maximo":
                        salida[idx_r, idx_c] = np.max(parche)
                    elif tipo_filtro == "minimo":
                        salida[idx_r, idx_c] = np.min(parche)

    return np.clip(np.round(salida), 0, 255).astype(np.uint8)


def cargar_imagen(ruta: str) -> np.ndarray:
    """Carga una imagen en formato NumPy array uint8 (RGB o escala de grises)."""
    if not os.path.isfile(ruta):
        raise FileNotFoundError(f"No se encontró el archivo: {ruta}")

    with Image.open(ruta) as im:
        im = im.convert("RGB")
        return np.array(im, dtype=np.uint8)


def guardar_imagen(img: np.ndarray, ruta: str) -> str:
    """Guarda una imagen NumPy en disco."""
    os.makedirs(os.path.dirname(os.path.abspath(ruta)), exist_ok=True)
    im = Image.fromarray(img)
    im.save(ruta)
    return ruta


def calcular_metricas(img_orig: np.ndarray, img_red: np.ndarray, stride: int, tam_mascara: int) -> dict:
    """Calcula estadísticas y dimensiones comparativas entre original y reducida."""
    alto_o, ancho_o = img_orig.shape[:2]
    alto_r, ancho_r = img_red.shape[:2]

    pixeles_orig = alto_o * ancho_o
    pixeles_red = alto_r * ancho_r
    factor_reduccion = (pixeles_orig / pixeles_red) if pixeles_red > 0 else 0
    porcentaje_reduccion = ((pixeles_orig - pixeles_red) / pixeles_orig * 100.0) if pixeles_orig > 0 else 0

    return {
        "alto_orig": alto_o,
        "ancho_orig": ancho_o,
        "canales_orig": img_orig.shape[2] if img_orig.ndim == 3 else 1,
        "pixeles_orig": pixeles_orig,
        "alto_red": alto_r,
        "ancho_red": ancho_r,
        "canales_red": img_red.shape[2] if img_red.ndim == 3 else 1,
        "pixeles_red": pixeles_red,
        "factor_reduccion": factor_reduccion,
        "porcentaje_reduccion": porcentaje_reduccion,
        "stride": stride,
        "tam_mascara": tam_mascara,
        "tam_bytes_estimado_orig": img_orig.nbytes,
        "tam_bytes_estimado_red": img_red.nbytes,
    }


def calcular_metricas_ruido(img_orig: np.ndarray, img_filt: np.ndarray, stride: int, tam_mascara: int, tipo_filtro: str) -> dict:
    """Calcula estadísticas de corrección de ruido y dimensiones entre la imagen original y la filtrada."""
    alto_o, ancho_o = img_orig.shape[:2]
    alto_f, ancho_f = img_filt.shape[:2]

    if alto_o == alto_f and ancho_o == ancho_f:
        diff = np.abs(img_orig.astype(np.float32) - img_filt.astype(np.float32))
        mae = float(np.mean(diff))
        max_diff = float(np.max(diff))
    else:
        mae = 0.0
        max_diff = 0.0

    return {
        "alto_orig": alto_o,
        "ancho_orig": ancho_o,
        "canales_orig": img_orig.shape[2] if img_orig.ndim == 3 else 1,
        "alto_filt": alto_f,
        "ancho_filt": ancho_f,
        "canales_filt": img_filt.shape[2] if img_filt.ndim == 3 else 1,
        "pixeles_filt": alto_f * ancho_f,
        "stride": stride,
        "tam_mascara": tam_mascara,
        "tipo_filtro": tipo_filtro,
        "mae_correccion": mae,
        "max_correccion": max_diff,
        "tam_bytes_estimado": img_filt.nbytes,
    }


def generar_mascara_fourier(
    alto: int,
    ancho: int,
    tipo_filtro: str = "pasa_bajas",
    modelo: str = "gaussiano",
    d0: float = 60.0,
    d1: float = 120.0,
    orden: int = 2
) -> np.ndarray:
    """
    Genera la máscara frecuencial 2D centrada H(u, v) con valores en [0.0, 1.0].
      - Puntos blancos (1.0): Frecuencias que se dejan pasar.
      - Puntos negros (0.0): Frecuencias que se atenúan o eliminan (supresión de ruido).

    Parámetros:
      - alto, ancho: Dimensiones espaciales de la imagen / espectro.
      - tipo_filtro: 'pasa_bajas', 'pasa_altas', 'rechazo_banda', 'pasa_banda'.
      - modelo: 'gaussiano', 'butterworth', 'ideal'.
      - d0: Frecuencia de corte principal (radio respecto al centro).
      - d1: Frecuencia secundaria para filtros de banda.
      - orden: Orden del filtro Butterworth (n >= 1).
    """
    u = np.arange(alto) - alto // 2
    v = np.arange(ancho) - ancho // 2
    V, U = np.meshgrid(v, u)
    D = np.sqrt(U**2 + V**2)

    d0_safe = max(float(d0), 1e-5)
    d1_safe = max(float(d1), 1e-5)

    # 1. Base Pasa-Bajas
    if modelo == "ideal":
        h_lpf = (D <= d0_safe).astype(np.float64)
    elif modelo == "butterworth":
        h_lpf = 1.0 / (1.0 + (D / d0_safe)**(2 * max(1, orden)))
    else:  # gaussiano
        h_lpf = np.exp(-(D**2) / (2.0 * (d0_safe**2)))

    # 2. Configuración según el tipo de filtro
    if tipo_filtro == "pasa_bajas":
        # Blanco (1) en el centro (bajas frecuencias), negro (0) en la periferia (altas frecuencias con ruido)
        return h_lpf

    elif tipo_filtro == "pasa_altas":
        # Negro (0) en el centro (bajas frecuencias), blanco (1) en la periferia (altas frecuencias)
        return 1.0 - h_lpf

    elif tipo_filtro == "rechazo_banda":
        # Elimina una banda anular de frecuencias intermedias
        d_min, d_max = min(d0_safe, d1_safe), max(d0_safe, d1_safe)
        if modelo == "gaussiano":
            d_mid = (d_min + d_max) / 2.0
            ancho_banda = max(d_max - d_min, 1e-5)
            return 1.0 - np.exp(-((D - d_mid)**2) / (2.0 * (ancho_banda / 2.0)**2))
        else:
            return 1.0 - ((D >= d_min) & (D <= d_max)).astype(np.float64)

    elif tipo_filtro == "pasa_banda":
        # Permite únicamente el paso de la banda anular especificada
        d_min, d_max = min(d0_safe, d1_safe), max(d0_safe, d1_safe)
        if modelo == "gaussiano":
            d_mid = (d_min + d_max) / 2.0
            ancho_banda = max(d_max - d_min, 1e-5)
            return np.exp(-((D - d_mid)**2) / (2.0 * (ancho_banda / 2.0)**2))
        else:
            return ((D >= d_min) & (D <= d_max)).astype(np.float64)

    return h_lpf


def aplicar_filtro_fourier(
    img: np.ndarray,
    tipo_filtro: str = "pasa_bajas",
    modelo: str = "gaussiano",
    d0: float = 60.0,
    d1: float = 120.0,
    orden: int = 2
) -> dict:
    """
    Aplica filtrado en el dominio de la frecuencia mediante la Transformada Rápida de Fourier 2D (FFT2).
    Soporta imágenes monocromáticas y policromáticas (RGB).

    Retorna un diccionario con:
      - imagen_salida: Imagen reconstruida por IFFT2 en uint8.
      - mascara_visual: Máscara H(u,v) en uint8 [0..255] (puntos blancos = frecuencias que pasan, negros = bloqueadas).
      - espectro_orig_visual: Espectro de magnitud original normalizado a [0..255].
      - espectro_filt_visual: Espectro de magnitud filtrado normalizado a [0..255].
      - mae_correccion: Desviación media absoluta respecto a la imagen de entrada.
      - tiempo_ms: Tiempo total de cómputo en milisegundos.
    """
    alto, ancho = img.shape[:2]
    es_color = (img.ndim == 3 and img.shape[2] in (3, 4))
    canales = img.shape[2] if es_color else 1

    H_mask = generar_mascara_fourier(alto, ancho, tipo_filtro, modelo, d0, d1, orden)

    t0 = time.time()
    if es_color:
        # FFT 2D en los ejes espaciales (0 y 1)
        F = np.fft.fftshift(np.fft.fft2(img.astype(np.float64), axes=(0, 1)), axes=(0, 1))
        # Multiplicación frecuencial (Broadcasting a los canales de color)
        G = F * H_mask[:, :, np.newaxis]
        # IFFT 2D inversa para reconstruir la imagen
        salida_comp = np.fft.ifft2(np.fft.ifftshift(G, axes=(0, 1)), axes=(0, 1))
        salida = np.real(salida_comp)
        mag_orig = np.mean(np.abs(F), axis=2)
        mag_filt = np.mean(np.abs(G), axis=2)
    else:
        F = np.fft.fftshift(np.fft.fft2(img.astype(np.float64)))
        G = F * H_mask
        salida_comp = np.fft.ifft2(np.fft.ifftshift(G))
        salida = np.real(salida_comp)
        mag_orig = np.abs(F)
        mag_filt = np.abs(G)

    t_ms = (time.time() - t0) * 1000.0

    salida_uint8 = np.clip(np.round(salida), 0, 255).astype(np.uint8)

    # Normalización del espectro logarítmico a uint8 para visualización de alta calidad
    def _normalizar_espectro(mag):
        esp = np.log1p(mag)
        mi, ma = esp.min(), esp.max()
        if ma - mi > 1e-9:
            return ((esp - mi) / (ma - mi) * 255.0).astype(np.uint8)
        return np.zeros_like(esp, dtype=np.uint8)

    espectro_orig_vis = _normalizar_espectro(mag_orig)
    espectro_filt_vis = _normalizar_espectro(mag_filt)
    mascara_vis = (np.clip(H_mask, 0.0, 1.0) * 255.0).astype(np.uint8)

    diff = np.abs(img.astype(np.float32) - salida_uint8.astype(np.float32))
    mae = float(np.mean(diff))

    return {
        "imagen_salida": salida_uint8,
        "mascara_visual": mascara_vis,
        "espectro_orig_visual": espectro_orig_vis,
        "espectro_filt_visual": espectro_filt_vis,
        "mae_correccion": mae,
        "tiempo_ms": t_ms,
        "alto": alto,
        "ancho": ancho,
        "d0": d0,
        "d1": d1,
        "orden": orden,
        "tipo_filtro": tipo_filtro,
        "modelo": modelo,
    }
