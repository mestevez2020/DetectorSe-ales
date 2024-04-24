import cv2
import numpy as np
import math


def load_and_convert_to_grayscale(image_path):
    """Carga una imagen y la convierte a escala de grises."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: No se pudo cargar la imagen desde {image_path}.")
        return None
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image


def detect_mser_regions(gray_image):
    """Detecta regiones de alto contraste en una imagen en escala de grises usando MSER."""
    mser = cv2.MSER_create(_delta=5, _min_area=100, _max_area=14400)
    regions, _ = mser.detectRegions(gray_image)
    return regions


def filter_by_size(regions, min_area=500, max_area=5000):
    """Filtra regiones por tamaño."""
    filtered_regions = [region for region in regions if min_area < cv2.contourArea(region) < max_area]
    return filtered_regions


def filter_by_aspect_ratio(regions, aspect_ratio_range=(0.8, 1.2)):
    """Filtra regiones por la proporción de aspecto."""
    filtered_regions = []
    for region in regions:
        x, y, w, h = cv2.boundingRect(region)
        aspect_ratio = w / float(h)
        if aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1]:
            filtered_regions.append(region)
    return filtered_regions


def filter_by_shape(regions, circularity_threshold=0.75):
    """Filtra regiones por circularidad para identificar formas más regulares."""
    filtered_regions = []
    for region in regions:
        perimeter = cv2.arcLength(region, True)
        area = cv2.contourArea(region)
        if perimeter == 0:
            continue
        circularity = 4 * math.pi * (area / (perimeter * perimeter))
        if circularity > circularity_threshold:
            filtered_regions.append(region)
    return filtered_regions


def visualize_regions(image, regions):
    """Visualiza las regiones filtradas sobre la imagen original."""
    for region in regions:
        hull = cv2.convexHull(region)
        cv2.polylines(image, [hull], True, (0, 255, 0), 2)
    cv2.imshow('Filtered Regions', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_image(image_path):
    """Procesa una imagen completa, desde la carga hasta la visualización de regiones detectadas."""
    gray_image = load_and_convert_to_grayscale(image_path)
    if gray_image is None:
        return
    regions = detect_mser_regions(gray_image)
    regions = filter_by_size(regions)
    regions = filter_by_aspect_ratio(regions)
    regions = filter_by_shape(regions)

    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Error: No se pudo cargar la imagen original desde {image_path}.")
        return
    visualize_regions(original_image, regions)


# El siguiente bloque se usa solo para pruebas directas de este módulo.
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        process_image(sys.argv[1])
    else:
        print("Usage: python detector.py path_to_image")
