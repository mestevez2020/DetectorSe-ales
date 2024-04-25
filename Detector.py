import cv2
import matplotlib.pyplot as plt
import numpy as np


def convert_to_grayscale(original_image):
    # Convertir la imagen a escala de grises
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Mostrar la imagen en escala de grises
    cv2.imshow('Imagen en escala de grises', gray_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def expand_detected_regions(regions,original_image, expand_factor=1.5):
    expanded_regions = []
    for region in regions:
        x, y, w, h = cv2.boundingRect(region)
        aspect_ratio = w / float(h)
        if 0.9 <= aspect_ratio <= 1.1:  # Aceptar regiones cuya relación de aspecto es cercana a 1.0
        # Calcular las nuevas coordenadas y dimensiones del cuadro expandido
            new_x = max(0, int(x - (expand_factor - 1) / 2 * w))
            new_y = max(0, int(y - (expand_factor - 1) / 2 * h))
            new_w = min(original_image.shape[1], int(w * expand_factor))
            new_h = min(original_image.shape[0], int(h * expand_factor))
            expanded_regions.append((new_x, new_y, new_w, new_h))
    return expanded_regions






def enhance_contrast(original_image):


    # Convertir la imagen a escala de grises
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Mostrar la imagen en escala de grises
    plt.imshow(gray_image, cmap='gray')
    plt.title('Imagen en escala de grises')
    plt.show()

    # Calcular y mostrar el histograma de la imagen en escala de grises
    hist_original = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    plt.plot(hist_original)
    plt.title('Histograma de la imagen en escala de grises')
    plt.show()

    # Aplicar ecualización del histograma
    equ_image = cv2.equalizeHist(gray_image)

    # Mostrar la imagen ecualizada
    plt.imshow(equ_image, cmap='gray')
    plt.title('Imagen con contraste mejorado')
    plt.show()

    # Calcular y mostrar el histograma de la imagen ecualizada
    hist_equ = cv2.calcHist([equ_image], [0], None, [256], [0, 256])
    plt.plot(hist_equ)
    plt.title('Histograma de la imagen con contraste mejorado')
    plt.show()

    return equ_image


#Cambiar el tamaño de las imagenes que conseguimos recortando
def resize_regions(regions, image, target_size=(250, 250)):
    resized_regions = []
    for x, y, w, h in regions:
        # Recortar la región de interés de la imagen original
        region_image = image[y:y+h, x:x+w]

        # Cambiar el tamaño de la región recortada al tamaño objetivo
        resized_region = cv2.resize(region_image, target_size)

        # Agregar la región redimensionada a la lista de regiones redimensionadas
        resized_regions.append(resized_region)

    return resized_regions



def apply_mser(image_path):
    # Cargar la imagen en color
    original_image = cv2.imread(image_path)

    # Verificar si la imagen fue cargada correctamente
    if original_image is None:
        print(f"No se pudo cargar la imagen desde {image_path}")
        return

    # Convertir la imagen a escala de grises y filtrala
    gray_image = enhance_contrast(original_image)

    # Crear el detector MSER con parámetros personalizados
    mser = cv2.MSER_create(delta=5, min_area=200, max_area=2000)

    # Detectar regiones de alto contraste
    regions, _ = mser.detectRegions(gray_image)
    print(len(regions))
    expanded_regions = expand_detected_regions(regions,gray_image)
    print(len(expanded_regions))
    # Dibujar los contornos de las regiones detectadas sobre la imagen original


    #dibujar los cuadrados en la imagen
    for x, y, w, h in expanded_regions:

        cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)



    resized_regions = resize_regions(expanded_regions, original_image)




    # Mostrar la imagen con las regiones detectadas
    cv2.imshow('MSER', original_image)
    cv2.waitKey(0)

    for region in resized_regions:
        detectar_colores(region)
    cv2.destroyAllWindows()



def detectar_colores(imagen):
    # Convertir la imagen al espacio de color HSV
    imagen_hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

    # Definir rangos de colores
    rango_azul_bajo = np.array([100, 50, 50])
    rango_azul_alto = np.array([140, 255, 255])

    rango_rojo_bajo1 = np.array([0, 50, 50])
    rango_rojo_alto1 = np.array([10, 255, 255])
    rango_rojo_bajo2 = np.array([160, 50, 50])
    rango_rojo_alto2 = np.array([180, 255, 255])

    # Filtrar los píxeles azules
    mascara_azul = cv2.inRange(imagen_hsv, rango_azul_bajo, rango_azul_alto)
    pixeles_azules = cv2.bitwise_and(imagen, imagen, mask=mascara_azul)

    # Filtrar los píxeles rojos
    mascara_roja1 = cv2.inRange(imagen_hsv, rango_rojo_bajo1, rango_rojo_alto1)
    mascara_roja2 = cv2.inRange(imagen_hsv, rango_rojo_bajo2, rango_rojo_alto2)
    mascara_roja = cv2.bitwise_or(mascara_roja1, mascara_roja2)
    pixeles_rojos = cv2.bitwise_and(imagen, imagen, mask=mascara_roja)
    cv2.imshow("original", imagen)
    cv2.imshow('Píxeles Azules', pixeles_azules)
    cv2.imshow('Píxeles Rojos', pixeles_rojos)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return pixeles_azules, pixeles_rojos
# El siguiente bloque se usa solo para pruebas directas de este módulo.
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Aplicar MSER a la imagen con parámetros personalizados
        apply_mser(sys.argv[1])
    else:
        print("Usage: python detector.py path_to_image")
