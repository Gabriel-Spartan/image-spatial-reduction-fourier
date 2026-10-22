# Sistema de Reducción Espacial de Imágenes

Aplicación de escritorio para el **procesamiento digital de imágenes y visión por computadora**, diseñada para realizar la **reducción espacial (downsampling o escalado descendente)** mediante **filtrado por máscara (kernel de vecindad) y salto de muestreo (*stride*)**, **sin utilizar funciones automáticas de biblioteca** como `imresize`, `cv2.resize` ni métodos automáticos de interpolación de OpenCV o PIL.

Todo el cálculo matricial y algorítmico está implementado de forma pura con **NumPy** y optimizado mediante operaciones tensoriales vectorizadas en C.

---

## 📋 Tabla de Contenidos
- [Características Principales](#-características-principales)
- [Fundamento Matemático y Algorítmico](#-fundamento-matemático-y-algorítmico)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Requisitos del Sistema](#-requisitos-del-sistema)
- [Instalación](#-instalación)
- [Modo de Ejecución](#-modo-de-ejecución)
- [Guía de Uso de la Interfaz](#-guía-de-uso-de-la-interfaz)
- [Tipos de Filtros Disponibles](#-tipos-de-filtros-disponibles)
- [Licencia y Créditos](#-licencia-y-créditos)

---

## ✨ Características Principales

- **Reducción por Stride Configurable:** Salto de muestreo ajustable de $1$ a $8$ píxeles (equivalente a reducciones a $1/2$, $1/3$, $1/4$, ..., $1/8$ de la escala original).
- **Filtrado Antialiasing:** Suavizado espacial previo al diezmado para mitigar el fenómeno de *aliasing* (efecto Moiré y bordes dentados).
- **Mitigación de Ruido en el Dominio Espacial:**
  - **Filtro de la Media (Box Blur):** Promedio aritmético uniforme; atenúa el ruido gaussiano al reducir la varianza local por un factor de $1/N^2$.
  - **Filtro de Moda (Estadístico de Frecuencia):** Selecciona el valor más repetido en la vecindad; ideal para suprimir perturbaciones impulsivas y aislar los tonos dominantes sin oscurecer la imagen.
  - **Filtro Gaussiano 2D:** Convolución ponderada simétrica con parámetro $\sigma$ configurable; elimina ruido conservando bordes y transiciones suaves.
- **Filtrado en el Dominio de la Frecuencia (Transformada de Fourier 2D - FFT2):**
  - Descomposición ortogonal mediante `np.fft.fft2` y centrado de bajas frecuencias con `fftshift`.
  - **Máscaras Frecuenciales $H(u,v)$:** Puntos blancos ($1.0$) dejan pasar las frecuencias deseadas; puntos negros ($0.0$) bloquean y eliminan las frecuencias espurias de ruido.
  - **Filtros Pasa-Bajas (LPF):** Centro blanco (bajas frecuencias) y periferia negra (altas frecuencias); elimina el ruido de alta frecuencia de forma radical.
  - **Filtros Pasa-Altas (HPF):** Centro negro y periferia blanca; resalta bordes y transiciones rápidas.
  - **Rechazo de Banda (Notch / Band-Stop):** Bloquea anillos o bandas de frecuencia específicas para eliminar ruido periódico o interferencias.
  - Perfiles matemáticos disponibles: **Gaussiano**, **Butterworth** (con orden $n$ configurable) e **Ideal**.
- **Editor Interactivo de Máscara Espacial (Kernel):**
  - Permite visualizar los coeficientes numéricos en coma flotante.
  - Permite editar manualmente cualquier valor de la matriz directamente en la interfaz.
  - Botón de auto-normalización ($\sum = 1.0$) para evitar saturación luminosa.
- **Cómputo Tensorial de Alto Rendimiento:**
  - Implementación con `numpy.lib.stride_tricks.sliding_window_view` y `numpy.einsum` en espacio.
  - Transformada rápida de Fourier `np.fft` multicanal vectorizada en frecuencia con tiempos de respuesta inferiores a $150\text{ ms}$.
- **Interfaz Gráfica Profesional con 3 Pestañas Especializadas:**
  - Pestaña 1: Reducción Espacial (Salto / Stride).
  - Pestaña 2: Mitigación Espacial de Ruido (Media, Moda, Gaussiano).
  - Pestaña 3: Filtrado en Frecuencia (Transformada de Fourier 2D, Espectro Logarítmico y Máscara $H(u,v)$).

---

## 🧠 Fundamento Matemático y Algorítmico

### 1. El Problema del Muestreo y el Aliasing
De acuerdo con el **Teorema de Muestreo de Nyquist-Shannon**, para muestrear una señal sin perder información ni generar frecuencias espaciales espurias (*aliasing*), la frecuencia de muestreo debe ser al menos el doble de la frecuencia máxima presente en la imagen. Al reducir una imagen por un factor de salto $S$ (*stride*), se reduce la frecuencia de corte permitida; por ende, es imperativo aplicar un **filtro pasa-bajas** (como el Gaussiano o de la Media) para atenuar las altas frecuencias antes del submuestreo.

### 2. Convolución Discreta 2D y Salto
Dado un kernel normalizado $K$ de dimensiones $M \times M$ y una imagen acolchada $I_{pad}$, cada píxel de salida $(i, j)$ en la imagen reducida se calcula en la posición correspondiente al salto $S$:

$$I_{red}(i, j) = \sum_{u=-r}^{r} \sum_{v=-r}^{r} I_{pad}(i \cdot S + r + u, \; j \cdot S + r + v) \cdot K(u + r, \; v + r)$$

donde $r = \lfloor M / 2 \rfloor$ es el radio de la máscara.

### 3. Filtro Gaussiano 2D
La máscara gaussiana se genera mediante la distribución normal bivariada simétrica:

$$G(x, y) = \frac{1}{2\pi\sigma^2} \exp\left(-\frac{x^2 + y^2}{2\sigma^2}\right)$$

y se normaliza para asegurar ganancia unitaria:

$$K(x, y) = \frac{G(x, y)}{\sum_u \sum_v G(u, v)}$$

### 4. Filtrado en el Dominio de la Frecuencia (Transformada de Fourier 2D)
Por el **Teorema de Convolución**, la convolución espacial de una imagen $f(x, y)$ con un kernel $h(x, y)$ equivale a una multiplicación directa punto a punto de sus transformadas en el dominio frecuencial:

$$g(x, y) = \mathcal{F}^{-1}\big\{ F(u, v) \cdot H(u, v) \big\}$$

donde:
- $F(u, v) = \mathcal{F}\{f(x, y)\}$ es el espectro complejo obtenido mediante `np.fft.fft2` y centrado con `np.fft.fftshift`.
- $H(u, v)$ es la **máscara de transferencia frecuencial** ($0.0 \le H(u, v) \le 1.0$):
  - **⚪ Puntos Blancos ($H = 1.0$):** Frecuencias que se dejan pasar íntegramente.
  - **⚫ Puntos Negros ($H = 0.0$):** Frecuencias que se atenúan o anulan completamente para eliminar el ruido.
- $D(u, v) = \sqrt{(u - M/2)^2 + (v - N/2)^2}$ representa la distancia euclidiana de cada coeficiente respecto al centro (origen de frecuencias).

#### Tipos de Filtros Frecuenciales:
- **Pasa-Bajas (LPF):** Centro blanco (deja pasar bajas frecuencias: iluminación y formas globales) y periferia negra (bloquea altas frecuencias: ruido, picos y fluctuaciones bruscas).
  - *Gaussiano:* $H(u, v) = \exp\left(-\frac{D^2(u, v)}{2 D_0^2}\right)$ (sin oscilaciones ni efecto Gibbs).
  - *Butterworth:* $H(u, v) = \frac{1}{1 + [D(u, v) / D_0]^{2n}}$ (transición gradual según orden $n$).
  - *Ideal:* $H(u, v) = 1$ si $D(u, v) \le D_0$, de lo contrario $0$.
- **Pasa-Altas (HPF):** Centro negro (bloquea bajas frecuencias) y periferia blanca (pasan altas frecuencias: aristas, contornos nítidos). Se calcula como $H_{HPF} = 1 - H_{LPF}$.
- **Rechazo de Banda (Band-Stop / Notch):** Bloquea un anillo de frecuencias entre $D_0$ y $D_1$ para suprimir ruido periódico o interferencias sinusoidales.

---

## 📁 Estructura del Proyecto

```text
Reduccion Imagen/
│
├── main.py              # Punto de entrada principal para iniciar la aplicación
├── logica.py            # Motor algorítmico puro en NumPy (sin dependencias de interfaz)
├── interfaz.py          # Interfaz gráfica de usuario en Tkinter / ttk
├── README.md            # Documentación completa del sistema
│
├── img/
│   ├── original/        # Carpeta para colocar imágenes de entrada de prueba
│   │   └── landscapeNature5.jpg
│   └── reducida/        # Carpeta de destino predeterminada para exportaciones
│       └── landscapeNature5_reducida_s2_m3x3.jpg
│
└── __pycache__/         # Bytecode de Python generado automáticamente
```

---

## 💻 Requisitos del Sistema

- **Sistema Operativo:** Linux, Windows o macOS.
- **Python:** Versión `3.8` o superior (probado en Python 3.10, 3.11 y 3.12).
- **Paquetes Requeridos:**
  - `numpy >= 1.20.0`
  - `pillow >= 9.0.0`
  - `tkinter` (incluido en Python en Windows/macOS; en distribuciones Linux suele requerir el paquete del gestor del sistema).

---

## 🚀 Instalación

### 1. Clonar o acceder al directorio del proyecto:
```bash
cd "Reduccion Imagen"
```

### 2. (Recomendado) Crear y activar un entorno virtual:

- **En Linux / macOS:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

- **En Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```

### 3. Instalar las dependencias de Python:
```bash
pip install numpy pillow
```

> **Nota para usuarios de Linux:**
> Si al ejecutar la aplicación aparece un error relacionado con `tkinter`, instala el soporte con el gestor de paquetes de tu distribución:
> - En Debian / Ubuntu / Mint: `sudo apt install python3-tk`
> - En Fedora / RHEL: `sudo dnf install python3-tkinter`
> - En Arch Linux: `sudo pacman -S tk`

---

## ⚡ Modo de Ejecución

Para iniciar la aplicación, ejecuta desde la raíz del proyecto:

```bash
python3 main.py
```

o bien:

```bash
python main.py
```

También puedes invocar directamente el módulo de interfaz:
```bash
python3 interfaz.py
```

---

## 🖥️ Guía de Uso de la Interfaz (Sistema de 3 Pestañas)

La aplicación cuenta con tres pestañas de trabajo especializadas accesibles desde la barra superior:

```
+-------------------------------------------------------------------------------------------------------------------+
| [Header] ●●● PROCESAMIENTO ESPACIAL Y FRECUENCIAL  | Muestra: [image.jpg] | [📂 Cargar] | [💾 Exportar]           |
+-------------------------------------------------------------------------------------------------------------------+
| [ 📉 Pestaña 1: Reducción Espacial ]  [ 🛡️ Pestaña 2: Mitigación Espacial ]  [ 🌊 Pestaña 3: Transformada Fourier ]  |
+-------------------------------------------------------------------------------------------------------------------+
| PESTAÑA 1 (REDUCCIÓN ESPACIAL):                                                                                   |
| - Parámetros: Máscara (3x3 a 15x15), Stride (2 a 8 px), Filtro antialiasing, Sigma                              |
| - Centro: Editor y visualizador interactivo de matriz de kernel normalizado                                       |
| - Derecha: Imagen Original (Entrada) vs Imagen Reducida (Salida)                                                  |
|                                                                                                                   |
| PESTAÑA 2 (MITIGACIÓN ESPACIAL DE RUIDO):                                                                         |
| - Parámetros: Filtro Especializado (Media, Moda, Gaussiano), Vecindad, Sigma, Stride=1                             |
| - Centro: Diagnóstico de Comportamiento del Filtro frente al Ruido + Matriz Vecindad                              |
| - Derecha: Imagen con Ruido (Entrada) vs Imagen Restaurada / Limpia (Salida)                                      |
|                                                                                                                   |
| PESTAÑA 3 (TRANSFORMADA DE FOURIER 2D - FRECUENCIA):                                                              |
| - Parámetros: Tipo (Pasa-Bajas, Pasa-Altas, Rechazo de Banda), Perfil (Gaussiano, Butterworth, Ideal), Radio D₀    |
| - Centro: Máscara H(u,v) (⚪ Puntos blancos pasan, ⚫ Puntos negros bloquean) + Espectro log(1 + |F|)             |
| - Derecha: Imagen de Entrada (Espacial) vs Imagen Reconstruida por IFFT2 (Salida Limpia)                         |
+-------------------------------------------------------------------------------------------------------------------+
| [Footer] Estado de la operación • Dimensiones Entrada -> Salida • Métricas (MAE / Factor) • Tiempo de cómputo ms  |
+-------------------------------------------------------------------------------------------------------------------+
```

### 📉 Pestaña 1: Reducción Espacial (Salto / Stride)
Diseñada para **reducir las dimensiones físicas de la imagen** sin distorsión por aliasing:
1. **Configurar Stride:** Ajusta el salto de $2$ a $8\text{ px}$ (para reducciones de $1/2$ a $1/8$).
2. **Seleccionar Filtro Antialiasing:** Elige entre `gaussiano`, `media`, `ponderado` o `personalizado`.
3. **Editor de Matriz:** Visualiza los coeficientes exactos del kernel o introduce valores numéricos manualmente.
4. **Métricas de Compresión:** Revisa el factor de escala ($2.00\times$, $4.00\times$), porcentaje de reducción de área y tiempo de ejecución.

### 🛡️ Pestaña 2: Mitigación de Ruido en Espacio (Media / Moda / Gaussiano)
Diseñada específicamente para **limpiar y restaurar imágenes ruidosas** mediante vecindades espaciales:
1. **Filtro Especializado:**
   - **`media`:** Suavizado homogéneo para reducir la varianza de ruido gaussiano blanco.
   - **`moda`:** Selección por mayor frecuencia estadística local; elimina picos y valores atípicos preservando colores dominantes.
   - **`gaussiano`:** Atenúa el ruido ponderando por distancia espacial ($\sigma$), conservando contornos nítidos.
2. **Resolución Original (Stride = 1 px):** Por defecto opera a resolución completa ($100\%$ escala) para limpiar el ruido sin reducir la imagen. Si también se desea reducir, se puede ajustar el Stride a $2, 3...$
3. **Diagnóstico en Tiempo Real:** El panel central detalla el fundamento matemático del filtro activo y muestra los pesos de la ventana.
4. **Métricas de Ruido:** Calcula la corrección media absoluta (**MAE**) respecto a la imagen ruidosa original.

### 🌊 Pestaña 3: Filtrado en Frecuencia (Transformada de Fourier 2D)
Diseñada para **eliminar ruido en el espectro frecuencial** aislando o suprimiendo bandas armónicas:
1. **Filtro Frecuencial:**
   - **`pasa_bajas` (LPF):** Elimina las frecuencias altas donde se concentra el ruido aleatorio y las variaciones bruscas.
   - **`pasa_altas` (HPF):** Elimina las frecuencias bajas y resalta contornos, bordes y transiciones nítidas.
   - **`rechazo_banda` (Notch):** Elimina un anillo o banda específica de frecuencias para suprimir ruido periódico.
2. **Interpretación de la Máscara $H(u,v)$:**
   - **⚪ Puntos Blancos ($1.0$):** Frecuencias que se dejan pasar libremente al dominio espacial.
   - **⚫ Puntos Negros ($0.0$):** Frecuencias que se anulan o filtran para erradicar el ruido.
3. **Radio de Corte ($D_0$):** Controla el límite circular en píxeles del espectro. Para mitigar ruido gaussiano en imágenes de 1000px, valores de $D_0 \in [40, 90]\text{ px}$ ofrecen un balance óptimo.
4. **Perfiles:**
   - **Gaussiano (Recomendado):** Transición suave sin ondulaciones ni efecto de campana/Gibbs.
   - **Butterworth:** Control de orden $n$ para regular la pendiente de atenuación.
   - **Ideal:** Corte circular estricto.

### 💾 Exportación Inteligente
Al pulsar **"💾 Exportar Resultado"**, el sistema detecta automáticamente qué pestaña está activa y sugiere el nombre adecuado:
- En Pestaña 1: `nombre_reducida_s{stride}_m{mascara}.ext`
- En Pestaña 2: `nombre_filtrada_{filtro}_m{mascara}.ext`
- En Pestaña 3: `nombre_fourier_{tipo}_{modelo}_D{d0}.ext`

---

## 🔬 Catálogo General de Filtros Disponibles

| Dominio | Filtro | Mecanismo | Comportamiento frente al Ruido |
| :--- | :--- | :--- | :--- |
| **Espacial** | **Media** | Convolución uniforme ($1/N^2$) | Reduce drásticamente la varianza del ruido gaussiano mediante suavizado uniforme (*box blur*). |
| **Espacial** | **Moda** | Frecuencia estadística en ventana | Elimina impurezas numéricas y ruido impulsivo preservando colores dominantes. |
| **Espacial** | **Gaussiano** | Campana normal 2D según $\sigma$ | Atenúa el ruido de forma suave y continua, preservando bordes principales con nitidez. |
| **Frecuencial** | **Fourier Pasa-Bajas** | $H(u,v)$: Centro blanco, exterior negro | **Elimina las altas frecuencias (ruido y picos)** dejando pasar la información estructural suave. |
| **Frecuencial** | **Fourier Pasa-Altas** | $H(u,v)$: Centro negro, exterior blanco | Elimina las frecuencias bajas (fondos planos) y resalta bordes y transiciones rápidas. |
| **Frecuencial** | **Fourier Rechazo Banda** | $H(u,v)$: Anillo negro entre $D_0$ y $D_1$ | Erradica patrones periódicos de ruido, líneas de interferencia o armónicos no deseados. |

---

## 🛡️ Guía Práctica de Mitigación de Ruido

Para limpiar una imagen ruidosa (como `img/original/image.jpg`):

1. **Mitigación de Ruido sin Reducción (100% Escala):**
   - Configura el slider **Salto (Stride)** en `1 px (Sin reducción / 100%)`.
   - Selecciona el tamaño de máscara ($3\times3$ para ruido leve o moderado, $5\times5$ para ruido intenso).
   - Prueba el filtro de **Media** para suavizado uniforme, **Moda** para homogeneizar áreas continuas, o **Gaussiano** ($\sigma = 1.0$ a $1.8$) para atenuación de ruido con bordes naturales.
2. **Mitigación de Ruido con Reducción Simultánea:**
   - Ajusta **Salto (Stride)** a `2 px` o más.
   - El sistema aplicará el filtro antialiasing y mitigación de ruido sobre la vecindad antes de realizar el remuestreo, eliminando las frecuencias de ruido y reduciendo la resolución de forma limpia.

---

## 🛠️ Tecnologías Empleadas

- **Lenguaje:** Python 3
- **Procesamiento de Arreglos Numéricos:** NumPy (`sliding_window_view`, `einsum`, `pad`)
- **Interfaz de Usuario:** Tkinter & ttk
- **E/S de Archivos Gráficos:** Pillow (PIL)
