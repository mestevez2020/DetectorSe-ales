import argparse
import cv2

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


    # Cargar los datos de entrenamiento sin se necesita
    print("Cargando datos de entrenamiento desde " + args.train_path)



    # Create the detector
    print("Creando el detector " + args.detector)
    
    # Cargar los datos de test y ejecutar el detector en esas imágenes
    print("Probando el detector " + args.detector + " en " + args.test_path)



    # Guardar resultados en el fichero resultado.txt

    # Guardar resultados en el fichero resultado_por_tipo.txt






