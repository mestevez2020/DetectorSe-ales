import shutil

import cv2

import numpy as np

num_senal1 = ['0', '1', '2', '3', '4', '5', '7', '8', '9', '10', '15', '16']
num_senal2 = ['11', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31']
num_senal3 = ['14']
num_senal4 = ['17']
num_senal5 = ['13']
num_senal6 = ['38']

senal1 = []
senal2 = []
senal3 = []
senal4 = []
senal5 = []
senal6 = []

mascaras = []
mascaras_inversa = []


def obtener_senales(train_path):
    obtener_senales_tipo1(train_path)
    obtener_senales_tipo2(train_path)
    obtener_senales_tipo3(train_path)
    obtener_senales_tipo4(train_path)
    obtener_senales_tipo5(train_path)
    obtener_senales_tipo6(train_path)


def calculo_mascaras(imagen_promedio):
    mascara_ = mascara(imagen_promedio, 1, 0)
    #cv2.imshow('Píxeles Rojos', mascara_)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    mascaras.append(mascara_)
    mascara_inversa = 255 - mascara_
    mascaras_inversa.append(mascara_inversa)


def obtener_senales_tipo1(train_path):
    senal1.append(mascara_roja(lector_imagen(train_path + "/00/00000.ppm")))
    senal1.append(mascara_roja(lector_imagen(train_path + "/01/00000.ppm")))
    senal1.append(mascara_roja(lector_imagen(train_path + "/02/00000.ppm")))
    senal1.append(mascara_roja(lector_imagen(train_path + "/03/00000.ppm")))
    senal1.append(mascara_roja(lector_imagen(train_path + "/04/00000.ppm")))
    senal1.append(mascara_roja(lector_imagen(train_path + "/05/00000.ppm")))

    imagen_promedio = obtener_imagen_promedio(senal1, 2, 45)
    calculo_mascaras(imagen_promedio)


def obtener_senales_tipo2(train_path):
    senal2.append(mascara_roja(lector_imagen(train_path + "/11/00000.ppm")))
    senal2.append(mascara_roja(lector_imagen(train_path + "/18/00000.ppm")))
    senal2.append(mascara_roja(lector_imagen(train_path + "/19/00000.ppm")))
    senal2.append(mascara_roja(lector_imagen(train_path + "/20/00000.ppm")))
    senal2.append(mascara_roja(lector_imagen(train_path + "/21/00000.ppm")))
    senal2.append(mascara_roja(lector_imagen(train_path + "/22/00000.ppm")))

    imagen_promedio = obtener_imagen_promedio(senal2, 2, 55)
    calculo_mascaras(imagen_promedio)


def obtener_senales_tipo3(train_path):
    senal3.append(mascara_roja(lector_imagen(train_path + "/14/00000.ppm")))
    senal3.append(mascara_roja(lector_imagen(train_path + "/14/00001.ppm")))
    senal3.append(mascara_roja(lector_imagen(train_path + "/14/00005.ppm")))
    senal3.append(mascara_roja(lector_imagen(train_path + "/14/00009.ppm")))
    senal3.append(mascara_roja(lector_imagen(train_path + "/14/00014.ppm")))
    senal3.append(mascara_roja(lector_imagen(train_path + "/14/00019.ppm")))

    imagen_promedio = obtener_imagen_promedio(senal3, 2, 70)
    calculo_mascaras(imagen_promedio)


def obtener_senales_tipo4(train_path):
    senal4.append(mascara_roja(lector_imagen(train_path + "/17/00000.ppm")))
    senal4.append(mascara_roja(lector_imagen(train_path + "/17/00001.ppm")))
    senal4.append(mascara_roja(lector_imagen(train_path + "/17/00005.ppm")))
    senal4.append(mascara_roja(lector_imagen(train_path + "/17/00009.ppm")))
    senal4.append(mascara_roja(lector_imagen(train_path + "/17/00014.ppm")))
    senal4.append(mascara_roja(lector_imagen(train_path + "/17/00019.ppm")))

    imagen_promedio = obtener_imagen_promedio(senal4, 2, 55)
    calculo_mascaras(imagen_promedio)


def obtener_senales_tipo5(train_path):
    senal5.append(mascara_roja(lector_imagen(train_path + "/13/00000.ppm")))
    senal5.append(mascara_roja(lector_imagen(train_path + "/13/00001.ppm")))
    senal5.append(mascara_roja(lector_imagen(train_path + "/13/00005.ppm")))
    senal5.append(mascara_roja(lector_imagen(train_path + "/13/00009.ppm")))
    senal5.append(mascara_roja(lector_imagen(train_path + "/13/00014.ppm")))
    senal5.append(mascara_roja(lector_imagen(train_path + "/13/00019.ppm")))

    imagen_promedio = obtener_imagen_promedio(senal5, 2, 55)
    calculo_mascaras(imagen_promedio)


def obtener_senales_tipo6(train_path):
    senal6.append(mascara_azul(lector_imagen(train_path + "/38/00000.ppm")))
    senal6.append(mascara_azul(lector_imagen(train_path + "/38/00001.ppm")))
    senal6.append(mascara_azul(lector_imagen(train_path + "/38/00005.ppm")))
    senal6.append(mascara_azul(lector_imagen(train_path + "/38/00009.ppm")))
    senal6.append(mascara_azul(lector_imagen(train_path + "/38/00014.ppm")))
    senal6.append(mascara_azul(lector_imagen(train_path + "/38/00019.ppm")))

    imagen_promedio = obtener_imagen_promedio(senal6, 0, 60)
    calculo_mascaras(imagen_promedio)


def lector_imagen(imagen_path):
    original_image = cv2.imread(imagen_path)
    return resize_regions(original_image, target_size=(25, 25))


def obtener_imagen_promedio(lista_imagenes, color_valor, umbral):
    shape = lista_imagenes[0].shape
    suma_imagenes = np.zeros(shape, dtype=np.uint64)
    for imagen in lista_imagenes:
        suma_imagenes += imagen.astype(np.uint64)

    imagen_promedio = (suma_imagenes / 6).astype(np.uint8)
    mask = imagen_promedio[:, :, color_valor] < umbral
    imagen_promedio[mask] = [0, 0, 0]
    return imagen_promedio

#expandir regiones y filtrar repetidas
def expand_detected_regions(regions, gray_image, original_image, expand_factor=1.2):
    expanded_regions = []
    for region in regions:
        if len(region) == 4:
            x, y, w, h = region
        else:
            x, y, w, h = cv2.boundingRect(region)
        aspect_ratio = w / float(h)
        if 0.9 <= aspect_ratio <= 1.1:
            new_x = max(0, int(x - (expand_factor - 1) / 2 * w))
            new_y = max(0, int(y - (expand_factor - 1) / 2 * h))
            new_w = min(gray_image.shape[1], int(w * expand_factor))
            new_h = min(gray_image.shape[0], int(h * expand_factor))
            imagen_recortada = original_image[new_y:new_y + new_h, new_x:new_x + new_w]
            img = resize_regions(imagen_recortada)
            score = -1
            number_senal = -1
            for i in range(6):
                number, porcentaje_total = comparacion_con_mascaras(img, i)

                if number != 0:
                    if score < porcentaje_total:
                        score = porcentaje_total
                        number_senal = number

            if number_senal != -1:
                encontrado = False
                for reg in expanded_regions:
                    if comparar_rectangulos(reg[0], reg[1], reg[2] + reg[0], reg[3] + reg[1], new_x,
                                            new_y, new_w + new_x, new_h + new_y):
                        encontrado = True
                        if score > reg[5]:
                            expanded_regions.remove(reg)
                            expanded_regions.append((new_x, new_y, new_w, new_h, number_senal, score))
                        break
                if not encontrado:
                    expanded_regions.append((new_x, new_y, new_w, new_h, number_senal, score))

    return expanded_regions

def comparacion_con_mascaras(img, number_mascara):
    if number_mascara != 5:
        img = mascara_roja(img)
        img = mascara(img, 1, 0)
    else:
        img = mascara_azul(img)
        img = mascara(img, 0, 1)

    blancos_mascara = np.where(mascaras[number_mascara] == 255, 1, 0)

    blancos_mascara_inversa = np.where(mascaras_inversa[number_mascara] == 255, 1, 0)

    correlation = np.sum(mascaras[number_mascara] * img)
    correlation_inversa = np.sum(mascaras_inversa[number_mascara] * img)

    # Contar los píxeles blancos en la imagen 1
    total_pixeles_blancos_imagen1 = np.sum(blancos_mascara)

    total_pixeles_blancos_inversa = np.sum(blancos_mascara_inversa)

    porcentaje_similitud_blancos = correlation / total_pixeles_blancos_imagen1 * 100
    porcentaje_similitud_blancos_inversa = correlation_inversa / total_pixeles_blancos_inversa * 100
    porcentaje_total = (float(porcentaje_similitud_blancos) * 0.7) + (
            (100 - float(porcentaje_similitud_blancos_inversa)) * 0.3)

    if porcentaje_similitud_blancos > 40 and porcentaje_total > 40 and porcentaje_similitud_blancos_inversa < 40:
        return number_mascara + 1, porcentaje_total
    else:
        return 0, porcentaje_total


#Contraste y equalizacion de la imagen
def enhance_contrast(original_image):
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    equ_image = cv2.equalizeHist(gray_image)

    return equ_image


def resize_regions(image, target_size=(25, 25)):
    return cv2.resize(image, target_size)

def mser_func( gray, mini, maxi):
    mser = cv2.MSER_create(delta=3, min_area=mini, max_area=maxi)
    regions, _ = mser.detectRegions(gray)
    return regions


def transformada_Hough( gray):
    # Aplica un desenfoque gaussiano para reducir el ruido
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                               param1=200, param2=30, minRadius=10, maxRadius=100)
    regions = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            # Calcula las coordenadas del cuadrado
            x1 = x - r
            y1 = y - r
            w = h = 2 * r
            regions.append([x1, y1, w, h])
    return regions


def detected_regions(image_paths):
    nombre_archivo = "resultado.txt"
    with open(nombre_archivo, 'w') as archivo:
        pass
    for image_path in image_paths:
        original_image = cv2.imread(image_path)
        if original_image is None:
            print(f"No se pudo cargar la imagen desde {image_path}")
            return

        gray_image = enhance_contrast(original_image)
        regs = transformada_Hough( gray_image)
        regs.extend(mser_func( gray_image, 200, 10000))

        expanded_regions = expand_detected_regions(regs, gray_image, original_image)

        #dibujar los cuadrados en la imagen
        for x, y, w, h, number_senal, score in expanded_regions:
            cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 0, 255), 1)
            with open(nombre_archivo, 'a') as archivo:
                archivo.write(f"{image_path[-9:]};{x};{y};{w + x};{h + y};{number_senal};{round(score/100,2)}\n")

            #print(image_path, x, y, w + x, h + y, number_senal, score, sep=';')
        ruta_imagen = "./resultado_imgs/" + image_path[-9:]
        cv2.imwrite( ruta_imagen, original_image)

def mascara_roja(imagen):
    imagen_hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
    rango_rojo_bajo1 = np.array([0, 50, 50])
    rango_rojo_alto1 = np.array([10, 255, 255])
    rango_rojo_bajo2 = np.array([160, 50, 50])
    rango_rojo_alto2 = np.array([180, 255, 255])
    mascara_roja1 = cv2.inRange(imagen_hsv, rango_rojo_bajo1, rango_rojo_alto1)
    mascara_roja2 = cv2.inRange(imagen_hsv, rango_rojo_bajo2, rango_rojo_alto2)
    mascara_roja = cv2.bitwise_or(mascara_roja1, mascara_roja2)

    pixeles_rojos = cv2.bitwise_and(imagen, imagen, mask=mascara_roja)

    #cv2.imshow("original", imagen)
    #cv2.imshow('Píxeles Rojos', pixeles_rojos)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return pixeles_rojos


def mascara_azul(imagen):
    imagen_hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
    rango_azul_bajo = np.array([90, 100, 100])
    rango_azul_alto = np.array([130, 255, 255])
    mascara_azul = cv2.inRange(imagen_hsv, rango_azul_bajo, rango_azul_alto)
    pixeles_azules = cv2.bitwise_and(imagen, imagen, mask=mascara_azul)

    #cv2.imshow("original", imagen)
    #cv2.imshow('Píxeles Azules', pixeles_azules)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return pixeles_azules

def mascara(imagen, r, b):
    imagen_hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
    rango_bajo = np.array([b, 0, r])
    rango_alto = np.array([255, 255, 255])
    return cv2.inRange(imagen_hsv, rango_bajo, rango_alto)

#interseccion over union
def comparar_rectangulos(x11, y11, x12, y12, x21, y21, x22, y22):
    # Coordenadas del rectángulo de intersección
    x1 = max(x11, x21)
    y1 = max(y11, y21)
    x2 = min(x12, x22)
    y2 = min(y12, y22)
    # Calcular la superficie de la intersección
    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)
    intersection_area = intersection_width * intersection_height

    box1_area = (x12 - x11 + 1) * (y12 - y11 + 1)
    box2_area = (x22 - x21 + 1) * (y22 - y21 + 1)
    union_area = box1_area + box2_area - intersection_area

    iou = intersection_area / union_area
    return iou > 0.55


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        detected_regions(sys.argv[1])
    else:
        print("Usage: python detector.py path_to_image")
