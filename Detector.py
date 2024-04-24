import cv2
import matplotlib.pyplot as plt


def convert_to_grayscale(image_path):
    # Cargar la imagen en color
    original_image = cv2.imread(image_path)

    # Convertir la imagen a escala de grises
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Mostrar la imagen en escala de grises
    cv2.imshow('Imagen en escala de grises', gray_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def enhance_contrast(image_path):
    # Cargar la imagen en color
    original_image = cv2.imread(image_path)

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

def apply_mser(image_path):
    # Cargar la imagen en color
    original_image = cv2.imread(image_path)

    # Verificar si la imagen fue cargada correctamente
    if original_image is None:
        print(f"No se pudo cargar la imagen desde {image_path}")
        return

    # Convertir la imagen a escala de grises y filtrala
    gray_image = enhance_contrast(image_path)

    # Crear el detector MSER con parámetros personalizados
    mser = cv2.MSER_create(delta=5, min_area=100, max_area=2000)

    # Detectar regiones de alto contraste
    regions, _ = mser.detectRegions(gray_image)

    # Dibujar los contornos de las regiones detectadas sobre la imagen original
    for region in regions:
        x, y, w, h = cv2.boundingRect(region)

        # Calcular la relación de aspecto
        aspect_ratio = w / float(h)

        # Filtrar regiones por la relación de aspecto
        if 0.9 <= aspect_ratio <= 1.1:  # Aceptar regiones cuya relación de aspecto es cercana a 1.0
            cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Mostrar la imagen con las regiones detectadas
    cv2.imshow('MSER', original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# El siguiente bloque se usa solo para pruebas directas de este módulo.
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Aplicar MSER a la imagen con parámetros personalizados
        apply_mser(sys.argv[1])
    else:
        print("Usage: python detector.py path_to_image")
