import argparse
import os
import shutil

import cv2

import Detector

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Crea y ejecuta un detector sobre las imágenes de test')
    parser.add_argument(
        '--detector', type=str, nargs="?", default="", help='Nombre del detector a ejecutar')
    parser.add_argument(
        '--train_path', default="", help='Carpeta con las imágenes de entrenamiento')
    parser.add_argument(
        '--test_path', default="", help='Carpeta con las imágenes de test')

    args = parser.parse_args()

    train_path = args.train_path

    test_path = args.test_path


    # Cargar los datos de entrenamiento sin se necesita
    print("Cargando datos de entrenamiento desde " + args.train_path)

    image_paths = []

    # Obtener la lista de archivos en la carpeta train_path
    gt_txt=''
    for filename in os.listdir(test_path+"_alumnos"):
        if filename.endswith(".ppm"):  # Filtrar solo archivos de imagen
            image_paths.append(os.path.join(test_path+"_alumnos", filename))
        else:
            gt_txt=train_path+'/'+filename

    if os.path.exists("resultado_imgs"):
        # Elimina el directorio existente y su contenido de manera recursiva
        shutil.rmtree("resultado_imgs")
        print("Directorio existente eliminado en", "resultado_imgs")

    # Crea el directorio)
    os.makedirs("resultado_imgs")
    Detector.obtener_senales(train_path)
    Detector.detected_regions(image_paths)


    #Detector.apply_mser(image_paths,gt_txt)
    # Create the detector
    print("Creando el detector " + args.detector)

    # Cargar los datos de test y ejecutar el detector en esas imágenes
    print("Probando el detector " + args.detector + " en " + args.test_path)



    # Guardar resultados en el fichero resultado.txt

    # Guardar resultados en el fichero resultado_por_tipo.txt






