import cv2
import numpy as np
import os
import glob
import tqdm
# from google.colab.patches import cv2_imshow
import shutil 

# from google.colab import drive
# drive.mount('/content/drive')


def cv2_imshow(image):
    name = 'Imagen'
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name,image)
    cv2.waitKey()
    cv2.destroyAllWindows()

def preproceso_plantilla(plantilla):
    '''
    (Function)
        Esta funcion hace un preproceso para la imagen que funge como plantilla
    (Parameters)
        - plantilla: [str] ruta a la imagen plantilla
    (Author)
        - Hector Limon
    '''
    plantilla = cv2.imread(plantilla)

    # Convertir la plantilla a escala de grises
    plantilla_gris = cv2.cvtColor(plantilla, cv2.COLOR_BGR2GRAY)
    return plantilla_gris


def comparacion_imagen(img,plantilla_gris,umbral=0.8,want_resize=True,resize=(224,224)):
    '''
    (Function)
        Esta funcion hace la comparacion entre una imagen y la plantilla
        en caso de que la plantilla este en la imagen la funcion retorna True
    (Parameters)
        - img: Es el arreglo numpy de la imagen que queremos contrastar contral plantilla
        - plantilla_gris: Es el arreglo numpy de plantilla en una escala de grises
        - umbral: Umbral de permeabilidad para coincidencia (default=False)
        - want_resize: [bool] De preferencia resize a 224 
        - resize: [tuple] Valores del nuevo size
    (Author)
        - Hector Limon
    '''
    if want_resize:
        img = cv2.resize(img, resize)
    # print(img.shape, img.shape)
    

    # Obtener las dimensiones de la plantilla
    alto, ancho = plantilla_gris.shape[::-1]

    # Realizar la correlación cruzada
    resultado = cv2.matchTemplate(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), plantilla_gris, cv2.TM_CCOEFF_NORMED)

    # Establecer un umbral
    umbral = 0.8

    # Encontrar las posiciones donde el resultado es mayor que el umbral
    posiciones = np.where(resultado >= umbral)

    # Crear una lista de las posiciones encontradas
    lista_posiciones = list(zip(*posiciones[::-1]))

    # Si la lista de posiciones no está vacía, entonces la plantilla se encontró en la imagen
    if lista_posiciones:
        # print('La plantilla se encuentra en la imagen')
        return True
    else:
        # print('La plantilla no se encuentra en la imagen')
        return False


def crear_directorio(ruta_root:str,name_folder:str):
    ruta_crear = ruta_root+'/'+name_folder
    try:
        os.mkdir(ruta_crear)
        print(f'Carpeta {name_folder} creada')
    except Exception as e:
        print(f'Carpeta {name_folder} no creada',e)
        pass
    return ruta_crear

def get_dict_plantilla_gris(ruta_plantillas):
    '''
    (Function)
        Esta funcion obtiene un diccionario de todas las plantillas que tenemos
        Esta compuesto por la ruta y el arreglo numpy en escala de grises para cada clase.
    (Parameters)
        - ruta_plantillas: [str] Ruta a la carpeta de las plantillas
    (Example)
        En la practica lo que mas nos servira sera la matriz numpy, suponga la clase bar
        - plantilla_gris = dict_plantilla['bar']['plantilla_gris']
    (Author)
        Hector Limon
    '''
    nombres = glob.glob(ruta_plantillas+'/*')
    dic_plantillas = {}
    for template in nombres:
        name_planilla = template.replace('\\','/').split('/')[-1].split('Plantilla_')[-1].split('.')[0]
        # print(name_planilla)
        
        plantilla_gris = preproceso_plantilla(template)
        dic_plantillas[name_planilla] = {'ruta':template, 'plantilla_gris':plantilla_gris}
    return dic_plantillas


# Cargar la imagen y la plantilla

def detectar_clase(imagen:str,dict_plantillas, return_list=False):
    '''
    (Function)
        Esta funcion recibe una imagen y retorna la clase a la que pertenece segun el diccionario
    (Parameters)
        - imagen: [str|np.array] Puede ser la matriz numpy o bien la ruta a la imagen
        - dict_plantillas: [dict] Diccionario que puede otener con la funcion "get_dict_plantilla_gris"
        - return_list: [bool] Puede llegarse a darse el caso que una imagen tenga mas de 1 clase, 
                    si desea ver las diferentes clases, use True, de lo contrario retorna la primera que encuentra
    (Author)
        - Hector Limon

    '''
    if isinstance(imagen,str): img = cv2.imread(imagen)
    else: img = imagen
    list_clases = []
    result = 0
    for key in dict_plantillas.keys():
        # print(key)
        plantilla_gris = dict_plantillas[key]['plantilla_gris']
        result = comparacion_imagen(img, plantilla_gris, want_resize=True, resize=(224,224))

        if result:
            if return_list: list_clases.append(key)
            else: return key
    if result == 0 and return_list==True:
        return [0]
    elif result == 0 and return_list==False: return 0
    else:
        return list_clases
   

def move_fotos_from_folder(ruta_plantillas,ruta_fotos,ruta_root,
                           create_folder=False,
                           listar_clases=False,show_cont=True
                           ):
    '''
    (Function)
        Esta funcion selecciona cada foto y segun su clasificacion los mete en una nueva carpeta con
        el nombre de la clase dentro de la ruta_root. 
    (Parameters)
        - ruta_plantillas: [str] Ruta a la carpeta de las plantillas
        - ruta_fotos: [str] Ruta a la carpeta de las fotos que se van a clasificar
        - ruta_root: [str] Ruta a la carpeta raiz donde se haran las subcarpetas con el nombre de clasificacion
        - listar_clase: [bool] Puede llegarse a darse el caso que una imagen tenga mas de 1 clase, 
                    si desea ver las diferentes clases, use True, de lo contrario retorna la primera que encuentra
        - show_cont: [bool] True para ver la cantidad de imagenes que ha clasificado
    (Author)
        Hector Limon
    '''
    
    # obtener diccionario de nombre, ruta a planillas y array de la imagen    
    dict_plantillas = get_dict_plantilla_gris(ruta_plantillas)
    # Creamos directorios
    if create_folder:
        for k in dict_plantillas.keys():
            _ = crear_directorio(ruta_root,k)
    # Lista de fotos
    fotos = glob.glob(ruta_fotos+'/*.png')

    # Contador
    cont = 0

    # Iteramos cada foto para generar clasificacion
    for foto in tqdm.tqdm(fotos[:]):
        # Nombre de la foto
        name_foto = foto.replace('\\','/').split('/')[-1]

        # Clasificacion de la foto
        name_class = detectar_clase(foto,dict_plantillas, return_list=listar_clases)
        if listar_clases:
            if len(name_class)>1:
                img = cv2.imread(foto)
                cv2_imshow(img)
                print('Se encontro mas de 1 clase para la foto de arriba, con cual desea quedarse?')
                print('Seleccione el indice de la lista ',name_class )
                ans = input('')
                l = 0
                while True:
                    try:
                        l = name_class[int(ans)]
                        break
                    except:
                        print('Inidice fuera de lugar, intentalo nuevamente')
                name_class = l
            else:
                name_class = name_class[0]
        
        # Encontro una clase
        if name_class != 0:
            # Movemos la imagen a la clase correspondiente
            shutil.move(foto, ruta_root+'/'+name_class+'/'+name_class+'_'+name_foto)
            if show_cont: print(cont)
            cont += 1