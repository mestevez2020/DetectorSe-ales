import cv2
import matplotlib.pyplot as plt
import numpy as np

num_senal1 = ['0', '1', '2', '3', '4', '5', '7', '8', '9', '10', '15', '16']
num_senal2 = ['11', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31']
num_senal3 = ['14']
num_senal4 = ['17']
num_senal5 = ['13']
num_senal6 = ['38']


senal1=[]
senal2=[]
senal3=[]
senal4=[]
senal5=[]
senal6=[]

def expand_detected_regions(regions,original_image,datos, expand_factor=1.5):
    expanded_regions = []
    for region in regions:
        x, y, w, h = cv2.boundingRect(region)
        aspect_ratio = w / float(h)
        if 0.9 <= aspect_ratio <= 1.1:
            new_x = max(0, int(x - (expand_factor - 1) / 2 * w))
            new_y = max(0, int(y - (expand_factor - 1) / 2 * h))
            new_w = min(original_image.shape[1], int(w * expand_factor))
            new_h = min(original_image.shape[0], int(h * expand_factor))
            encontrado=False
            for dato in datos:
                if (int(new_x) - 10 <= int(dato[1]) <= int(new_x) + 10 and
                        int(new_y) - 10 <= int(dato[2]) <= int(new_y) + 10 and
                        int(new_w) + int(new_x) - 10 <= int(dato[3]) <= int(new_w) + int(new_x) + 10 and
                        int(new_h) + int(new_y) - 10 <= int(dato[4]) <= int(new_h) + int(new_y) + 10):

                    encontrado=True


                    if encontrado:

                        if dato[5] in num_senal1:
                            senal1.append(region)
                            expanded_regions.append((new_x, new_y, new_w, new_h))
                        elif dato[5] in num_senal2:
                            senal2.append(region)
                            expanded_regions.append((new_x, new_y, new_w, new_h))
                        elif dato[5] in num_senal3:
                            senal3.append(region)
                            expanded_regions.append((new_x, new_y, new_w, new_h))
                        elif dato[5] in num_senal4:
                            senal4.append(region)
                            expanded_regions.append((new_x, new_y, new_w, new_h))
                        elif dato[5] in num_senal5:
                            senal5.append(region)
                            expanded_regions.append((new_x, new_y, new_w, new_h))
                        elif dato[5] in num_senal6:
                            senal6.append(region)
                            expanded_regions.append((new_x, new_y, new_w, new_h))

                    break
    return expanded_regions

#Contraste y equalizacion de la imagen
def enhance_contrast(original_image):

    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    plt.imshow(gray_image, cmap='gray')
    plt.title('Imagen en escala de grises')
    plt.show()

    hist_original = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    plt.plot(hist_original)
    plt.title('Histograma de la imagen en escala de grises')
    plt.show()

    equ_image = cv2.equalizeHist(gray_image)

    plt.imshow(equ_image, cmap='gray')
    plt.title('Imagen con contraste mejorado')
    plt.show()

    hist_equ = cv2.calcHist([equ_image], [0], None, [256], [0, 256])
    plt.plot(hist_equ)
    plt.title('Histograma de la imagen con contraste mejorado')
    plt.show()

    return equ_image


#Cambiar el tamaño de las imagenes que conseguimos recortando
def resize_regions(regions, image, target_size=(250, 250)):
    resized_regions = []
    for x, y, w, h in regions:
        region_image = image[y:y+h, x:x+w]

        resized_region = cv2.resize(region_image, target_size)

        resized_regions.append(resized_region)

    return resized_regions


def mser_func(original_image, min, max,datos):
    gray_image = enhance_contrast(original_image)

    mser = cv2.MSER_create(delta=5, min_area=min, max_area=max)

    regions, _ = mser.detectRegions(gray_image)
    expanded_regions = expand_detected_regions(regions, gray_image,datos)
    return expanded_regions

def apply_mser(image_paths,gt_txt=''):
    regiones = []
    datos = [linea.strip().split(';') for linea in open(gt_txt, 'r')]
    for image_path in image_paths:
        original_image = cv2.imread(image_path)
        if original_image is None:
            print(f"No se pudo cargar la imagen desde {image_path}")
            return


        datos_imagen = [arr for arr in datos if arr[0]==image_path[8:]]

        expanded_regions=mser_func(original_image,200,20000,datos_imagen)

    #dibujar los cuadrados en la imagen
        for x, y, w, h in expanded_regions:

            cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 0, 255), 1)

        #cv2.imshow('MSER', original_image)
        #cv2.waitKey(0)

        resized_regions = resize_regions(expanded_regions, original_image)



    for region in senal1:
        imagen=np.array(region)
        cv2.imshow("original", imagen)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        detectar_colores(region)


#esta parte esta mal
def detectar_colores(imagen):
    imagen_gray = np.array(imagen, dtype=np.uint8)

    # Convertir la imagen a formato BGR
    imagen_bgr = cv2.cvtColor(imagen_gray, cv2.COLOR_GRAY2BGR)

    # Convertir la imagen BGR a HSV
    imagen_hsv = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2HSV)

    rango_azul_bajo = np.array([100, 50, 50])
    rango_azul_alto = np.array([140, 255, 255])

    rango_rojo_bajo1 = np.array([0, 50, 50])
    rango_rojo_alto1 = np.array([10, 255, 255])
    rango_rojo_bajo2 = np.array([160, 50, 50])
    rango_rojo_alto2 = np.array([180, 255, 255])

    mascara_azul = cv2.inRange(imagen_hsv, rango_azul_bajo, rango_azul_alto)
    pixeles_azules = cv2.bitwise_and(imagen_bgr, imagen_bgr, mask=mascara_azul)

    mascara_roja1 = cv2.inRange(imagen_hsv, rango_rojo_bajo1, rango_rojo_alto1)
    mascara_roja2 = cv2.inRange(imagen_hsv, rango_rojo_bajo2, rango_rojo_alto2)
    mascara_roja = cv2.bitwise_or(mascara_roja1, mascara_roja2)
    pixeles_rojos = cv2.bitwise_and(imagen_bgr, imagen_bgr, mask=mascara_roja)

    pixeles_rojos_gris = cv2.cvtColor(pixeles_rojos, cv2.COLOR_BGR2GRAY)

    cantidad_pixeles_rojos = cv2.countNonZero(pixeles_rojos_gris)

    umbral= 6000


    regiones=[]
    print(cantidad_pixeles_rojos)
    if cantidad_pixeles_rojos > umbral:
        cv2.imshow("original", imagen_bgr)
        cv2.imshow('Píxeles Rojos', pixeles_rojos)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        regiones.append(pixeles_rojos)

    pixeles_azules_gris = cv2.cvtColor(pixeles_azules, cv2.COLOR_BGR2GRAY)

    cantidad_pixeles_azules = cv2.countNonZero(pixeles_azules_gris)
    umbral = 15000
    if cantidad_pixeles_azules > umbral:
        print(cantidad_pixeles_azules)
     #   cv2.imshow("original", imagen)
     #   cv2.imshow('Píxeles Azules', pixeles_azules)

     #   cv2.waitKey(0)
     #   cv2.destroyAllWindows()
        regiones.append(pixeles_azules)

    return regiones

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:

        apply_mser(sys.argv[1])
    else:
        print("Usage: python detector.py path_to_image")
