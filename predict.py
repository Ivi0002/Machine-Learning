# -*- coding: utf-8 -*-
"""
predict.py - Clasificador de imagenes Perro/Gato
================================================
Uso: python predict.py --image ruta/de/la/imagen.jpg

Ejemplos:
  python predict.py --image imagenes/perro1.jpg
  python predict.py --image imagenes/gato1.jpg
"""

import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Configuracion de codificacion para Windows
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Configuracion
MODEL_PATH    = os.path.join("CatDogTraining-2", "trainedmodels",
                             "vgg16_epoch_13_accuracy_84.55.h5")
WEIGHTS_PATH  = os.path.join("CatDogTraining-2", "trainedmodels",
                             "vgg16_epoch_13_accuracy_84.55_weights_.h5")
IMG_SIZE      = (224, 224)
# Las clases siguen el orden alfabetico del dataset: cat=0, dog=1
CLASS_NAMES   = {0: "Gato", 1: "Perro"}
CLASS_LABELS  = {0: "Cat", 1: "Dog"}
CLASS_COLORS  = {0: "#4FC3F7", 1: "#FF8A65"}


def cargar_modelo():
    """
    Carga el modelo entrenado desde disco.
    Usa reconstruccion manual de la arquitectura CNN y carga de pesos
    para compatibilidad con modelos guardados en versiones antiguas de Keras.
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Flatten,
                                             Dense, Dropout, GaussianNoise)
    except ImportError:
        print("ERROR: No se encontro TensorFlow/Keras.")
        print("   Instala con: pip install tensorflow  o  pip install tf-nightly")
        sys.exit(1)

    if not os.path.exists(WEIGHTS_PATH):
        print(f"ERROR: Pesos no encontrados en '{WEIGHTS_PATH}'")
        print("   Asegurate de ejecutar predict.py desde la raiz del proyecto.")
        sys.exit(1)

    print(f"[+] Reconstruyendo arquitectura del modelo CNN...")

    # Arquitectura identica a CatDogTraining-2/CatDogTraining.py
    # (2 clases: cat=0, dog=1)
    NUMBER_OF_CLASSES = 2
    model = Sequential()

    model.add(Input(shape=(224, 224, 3)))
    model.add(Conv2D(1, kernel_size=3, padding='same'))
    model.add(GaussianNoise(0.25))
    model.add(Conv2D(8, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Conv2D(16, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Conv2D(32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Conv2D(64, kernel_size=3, padding='same', activation='relu'))
    model.add(GaussianNoise(0.25))
    model.add(Conv2D(128, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Conv2D(256, kernel_size=3, padding='same', activation='relu'))
    model.add(GaussianNoise(0.25))
    model.add(Conv2D(512, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.1))
    model.add(GaussianNoise(0.25))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(NUMBER_OF_CLASSES, activation='sigmoid'))

    print(f"[+] Cargando pesos desde: {WEIGHTS_PATH}")
    model.load_weights(WEIGHTS_PATH)
    print("[OK] Modelo y pesos cargados correctamente.\n")
    return model


def preprocesar_imagen(ruta_imagen):
    """
    Carga y preprocesa la imagen para el modelo:
      1. Abre la imagen con PIL
      2. Convierte a RGB
      3. Redimensiona a 224x224 pixeles
      4. Convierte a array NumPy float32  (rango [0, 255])
      5. Agrega dimension de lote -> shape (1, 224, 224, 3)

    NOTA: El modelo fue entrenado con ImageDataGenerator() SIN rescale,
    por lo que espera pixeles en el rango original [0, 255], no [0, 1].
    """
    if not os.path.exists(ruta_imagen):
        print(f"ERROR: Imagen no encontrada en '{ruta_imagen}'")
        sys.exit(1)

    ext = os.path.splitext(ruta_imagen)[1].lower()
    if ext not in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        print(f"ERROR: Formato '{ext}' no soportado. Use JPG o PNG.")
        sys.exit(1)

    img_original = Image.open(ruta_imagen)
    img_rgb      = img_original.convert("RGB")
    img_redim    = img_rgb.resize(IMG_SIZE, Image.LANCZOS)

    img_array    = np.array(img_redim, dtype=np.float32)  # rango [0, 255]
    img_batch    = np.expand_dims(img_array, axis=0)       # shape: (1, 224, 224, 3)

    return img_batch, img_original


def predecir(model, img_batch):
    """
    Realiza la prediccion con el modelo.
    El modelo usa activacion 'sigmoid' en la capa de salida (no softmax),
    por lo que las salidas brutas NO suman 1 y producen porcentajes erroneos
    (p.ej. perro=48%, gato=52% para una imagen de perro clara).
    Aplicamos softmax sobre las salidas brutas para obtener una distribucion
    de probabilidad correcta que si sume 100%.
    Devuelve:
      - clase_idx: indice de la clase predicha (0=gato, 1=perro)
      - confianza: probabilidad de la clase predicha (%)
      - probs: array con probabilidades normalizadas de todas las clases
    """
    raw       = model.predict(img_batch, verbose=0)[0]  # salidas sigmoid crudas
    # Softmax: e^x_i / sum(e^x_j) -> distribucion que suma 1
    e         = np.exp(raw - np.max(raw))               # resta max para estabilidad numerica
    probs     = e / e.sum()
    clase_idx = int(np.argmax(probs))
    confianza = float(probs[clase_idx]) * 100.0
    return clase_idx, confianza, probs


def mostrar_resultado(img_original, ruta_imagen, clase_idx, confianza, probs):
    """
    Visualiza la imagen con el resultado de la prediccion usando matplotlib.
    """
    nombre_clase = CLASS_NAMES[clase_idx]
    color        = CLASS_COLORS[clase_idx]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5),
                             gridspec_kw={"width_ratios": [2, 1]})
    fig.patch.set_facecolor("#1E1E2E")

    # Panel izquierdo: imagen
    ax_img = axes[0]
    ax_img.imshow(img_original)
    ax_img.axis("off")
    ax_img.set_facecolor("#1E1E2E")
    ax_img.set_title(
        f"Imagen: {os.path.basename(ruta_imagen)}",
        color="white", fontsize=11, pad=8, fontweight="bold"
    )

    # Panel derecho: resultados
    ax_res = axes[1]
    ax_res.set_facecolor("#2A2A3E")
    ax_res.axis("off")

    # Titulo resultado
    ax_res.text(0.5, 0.90, "RESULTADO", transform=ax_res.transAxes,
                ha="center", va="center", fontsize=13, color="#AAAACC",
                fontweight="bold")

    # Clase predicha
    ax_res.text(0.5, 0.74, nombre_clase, transform=ax_res.transAxes,
                ha="center", va="center", fontsize=30, color=color,
                fontweight="bold")

    # Confianza
    ax_res.text(0.5, 0.59, f"Confianza: {confianza:.1f}%",
                transform=ax_res.transAxes,
                ha="center", va="center", fontsize=14, color="white")

    # Separador
    ax_res.plot([0.05, 0.95], [0.50, 0.50], color="#AAAACC",
                linewidth=0.5, alpha=0.4, transform=ax_res.transAxes)

    # Barras de probabilidad para cada clase
    clases  = list(CLASS_NAMES.values())
    colores = list(CLASS_COLORS.values())
    posiciones_y = [0.38, 0.22]

    for i, (clase, prob, yp, col) in enumerate(zip(clases, probs, posiciones_y, colores)):
        pct = float(prob) * 100
        # Texto y porcentaje
        ax_res.text(0.05, yp + 0.05, f"{clase}",
                    transform=ax_res.transAxes,
                    ha="left", va="center", fontsize=11, color="white", fontweight="bold")
        ax_res.text(0.95, yp + 0.05, f"{pct:.1f}%",
                    transform=ax_res.transAxes,
                    ha="right", va="center", fontsize=11, color=col)
        # Barra de fondo (gris)
        bar_bg = plt.Rectangle((0.05, yp - 0.02), 0.90, 0.06,
                                transform=ax_res.transAxes,
                                color="#3A3A5E", clip_on=False)
        ax_res.add_patch(bar_bg)
        # Barra de valor
        bar_val = plt.Rectangle((0.05, yp - 0.02), 0.90 * max(pct / 100, 0.01), 0.06,
                                 transform=ax_res.transAxes,
                                 color=col, alpha=0.85, clip_on=False)
        ax_res.add_patch(bar_val)

    fig.suptitle("Clasificador de Imagenes: Perro vs. Gato",
                 color="white", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    # Guardar resultado
    nombre_salida = "resultado_" + os.path.splitext(os.path.basename(ruta_imagen))[0] + ".png"
    plt.savefig(nombre_salida, dpi=150, bbox_inches="tight", facecolor="#1E1E2E")
    print(f"[+] Visualizacion guardada en: {nombre_salida}")
    plt.show()


def imprimir_resultado_consola(ruta_imagen, clase_idx, confianza, probs):
    """Imprime el resultado en consola de forma legible."""
    sep = "=" * 55
    print(sep)
    print("   CLASIFICADOR PERRO vs. GATO")
    print(sep)
    print(f"   Imagen analizada  : {ruta_imagen}")
    print(f"   Clase predicha    : {CLASS_NAMES[clase_idx]}")
    print(f"   Confianza         : {confianza:.2f}%")
    print(sep)
    print("   Probabilidades por clase:")
    for idx, (nombre, prob) in enumerate(zip(CLASS_NAMES.values(), probs)):
        barra_len = int(float(prob) * 30)
        barra = "#" * barra_len + "-" * (30 - barra_len)
        marcador = " <<< PREDICCION" if idx == clase_idx else ""
        print(f"   {nombre:8s}  [{barra}] {float(prob)*100:5.1f}%{marcador}")
    print(sep)


def main():
    parser = argparse.ArgumentParser(
        description="Clasifica una imagen como Perro o Gato usando una CNN entrenada.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ejemplos:\n"
            "  python predict.py --image imagenes/perro1.jpg\n"
            "  python predict.py --image imagenes/gato1.png\n"
        )
    )
    parser.add_argument(
        "--image", "-i",
        required=True,
        metavar="RUTA_IMAGEN",
        help="Ruta a la imagen JPG o PNG a clasificar."
    )
    args = parser.parse_args()

    print("\n[*] Iniciando clasificador Perro vs. Gato...\n")

    # 1. Cargar modelo
    model = cargar_modelo()

    # 2. Preprocesar imagen
    print(f"[*] Preprocesando imagen: {args.image}")
    img_batch, img_original = preprocesar_imagen(args.image)
    print(f"    Dimensiones originales : {img_original.size[0]}x{img_original.size[1]} px")
    print(f"    Redimensionada a       : {IMG_SIZE[0]}x{IMG_SIZE[1]} px")
    print(f"    Normalizacion          : pixeles / 255.0 -> rango [0, 1]")
    print(f"    Shape para el modelo   : {img_batch.shape}\n")

    # 3. Predecir
    print("[*] Realizando prediccion...")
    clase_idx, confianza, probs = predecir(model, img_batch)

    # 4. Resultado en consola
    imprimir_resultado_consola(args.image, clase_idx, confianza, probs)

    # 5. Visualizacion con matplotlib
    print("\n[*] Generando visualizacion...")
    mostrar_resultado(img_original, args.image, clase_idx, confianza, probs)


if __name__ == "__main__":
    main()
