import cv2
import numpy as np
import os
import glob
import tqdm
# from google.colab.patches import cv2_imshow
import shutil 

# holis
# from google.colab import drive
# drive.mount('/content/drive')
def clusterizar_color(img,k=2,want_resize=True, resize=(240,240)):
    '''
    (Function)
        Esta funcion clusteriza una imagen a k-cluster de colores
    (Parameters)
        -img: [array] de la imagen
        - k: Numero de cluster a formar
        - want_resize: [bool] En caso de querer redimensionar poner True
        - resize: [tuple] En caso de querer resize se ingresa la nueva dimension.
    (Returns)
        - res2: Imagen original clusterizada
        - ret: Es la suma de la distancia al cuadrado desde cada punto a sus centros correspondientes.
        - label: Esta es la matriz de etiquetas  donde cada elemento marcado '0', '1'.....
        - center: Esta es una serie de centros de grupos.
    (Authors)
        - Hector limon
        (Por favor siga enriqueciendo el proceso y escriba su nombre como author)
    '''
    # Reajustar imagen
    if want_resize:
        img = cv2.resize(img, resize)
    # aplanar
    Z = img.reshape((-1,3))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = k
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    # imagen_original_clusterizada, 
    return res2, ret, label, center


def cv2_imshow(image):
    name = 'Imagen'
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name,image)
    cv2.waitKey()
    cv2.destroyAllWindows()

def preproceso_plantilla(plantilla,want_resize=True,resize=(224,224)):
    '''
    (Function)
        Esta funcion hace un preproceso para la imagen que funge como plantilla
    (Parameters)
        - plantilla: [str] ruta a la imagen plantilla
    (Author)
        - Hector Limon
    '''
    plantilla = cv2.imread(plantilla)
    if want_resize:
        plantilla = cv2.resize(plantilla,resize)
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

def get_dict_plantilla_gris(ruta_plantillas,want_resize=True,resize=(220,220),k=2):
    '''
    (Function)
        Esta funcion obtiene un diccionario de todas las plantillas que tenemos
        Esta compuesto por la ruta y el arreglo numpy en escala de grises para cada clase.
    (Parameters)
        - ruta_plantillas: [str] Ruta a la carpeta de las plantillas
        - want_resize: [bool] True si quiere redimensionar la imagen
        - resize: [tuple] En caso de redimensionar la tupla con las nuevas dimensiones
        - k: Numero de cluster de color a obtener
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
        
        plantilla_gris = preproceso_plantilla(template,want_resize,resize)
        img = cv2.imread(template)
        if want_resize:
            img = cv2.resize(img,resize)
        res2, ret, label, center = clusterizar_color(img,k=k,want_resize=True, resize=(240,240))
        dic_plantillas[name_planilla] = {'ruta':template, 
                                         'plantilla_gris':plantilla_gris,
                                         'imagen':img,
                                         'centro_color':center,
                                         'imagen_clus':res2}
    return dic_plantillas


# Cargar la imagen y la plantilla

def detectar_clase(imagen:str,dict_plantillas:dict, 
                   resize:tuple=(224,224),return_list:bool=False,
                   umbral:float=0.8,use_cluster:bool=True):
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
    result = False
    for key in dict_plantillas.keys():
        if use_cluster:
            # Como el cluster es imagen real 
            plantilla_gris = cv2.cvtColor(dict_plantillas[key]['imagen_clus'], cv2.COLOR_BGR2GRAY)
        else:
            plantilla_gris = dict_plantillas[key]['plantilla_gris']
        result = comparacion_imagen(img, plantilla_gris, want_resize=True, resize=resize,umbral=umbral)

        if result:
            if return_list: list_clases.append(key)
            else: return key
        result = False
    if len(list_clases) == 0 and return_list==True:
        return [0]
    elif result == 0 and return_list==False: return 0
    else:
        return list_clases
   

def clasificacion_iter_size(img,dict_gris,n=200,salto=10,umbral=0.85,use_cluster:bool=True):
    '''
    (Function)
        Esta funcion esta diseñada para usarla de manera mas general, ya que mueve el resize
        debido a la variabilidad de pixelaje de la plantilla original con las distintas que se 
        pueden sucitar en los raters.
        Se recomienda usar n=200 y el de la plantilla resize=(30,30)
        de igual manera hacer pruebas seria bueno, ya que si hay una diferencia muy grande entre n
        y resize[0], puede dar malas clasificaciones
    (Parameters)
        - img: [str|np.aray] imagen que se desea clasificar
        - dict_gris: [dict] diccionario que se extrae
        - n: [int] tamaño de resize inicial considerelo en relacion con el resize del
        dict_gris
        - salto: [int|float] Tamaño de salto de descenso, si lo deja en 10 puede funcionar bien
        pero en general entre mas pequeño mas tardado
        - umbral: [float] porcentaje de aceptacion
    (Authors)
        - Hector Limon
    '''
    clase = 0
    while True:
        if n == 0:
            break
        clase =  detectar_clase(img,dict_gris,(n,n),False,umbral,use_cluster)
        n -= salto
        if clase != 0:
            # Detecto una clase, revisamos colores para los mas confusos
            #if clase == 'establecimiento_google':
                
            break
    
    return clase, n

def iter_umbral_fn (img:np.array,dict_gris:dict, n:int=100,salto_n:int=10,
                    umbral:float=0.95,salto_umbral:float=0.02,min_umbral:float=0.5,
                    use_cluster:bool=False):
    '''
    (Function)
        Esta funcion retoma la funcion clasificacion_iter_size agregando la iteracion sobre el umbral, debido a que 
        no siempre se puede tener un umbral especifico, podemos jugar con el para tener un clasificacion excata, tenga 
        cuidado de asignar un valor de n, en relacion al resize del dict_gris
    (Paremeters)
        - img: [np.array] arreglo matricial de imagen a clasificar
        - dict_gris: [dict] diccionario de plantillas
        - n: [int] se considera para el resize donde empezara use 100 si para dict_gris es 30 
        - salto_n: [int] Tamaño de salto para el nuevo resize (disminuye n de n a n-salto_n) en cada iteracion
        - umbral: [float] Umbral a usar en un principio, no sea austero.
        - salto_umbral: En caso de ser necesario bajaremos el umbral de salto_umbral en salto_umbral hasta llegar min_umbral
        - min_umbral: [float] el umbral minimo significativo, default 0.5, por lo que si llega a 0.5 ya no bajara mas
        - use_cluster: [bool] Si es True usa imagenes clusterizadas a 2 colores, comete mas errores
    (Returns)
        - clase: [str] clase detectada, si es 0, no encontro clase
        - n_1: [int] resize en el que encontro la clase
        - umbral: [float] umbral en el que se detecto la clase
    (Authors)
        - Hector limon
        (Por favor siga enriqueciendo el proceso y escriba su nombre como author)
    '''
    n_1 = 0
    clase = 0
    while n_1 == 0 and clase == 0 and umbral >= min_umbral:
        clase, n_1 = clasificacion_iter_size(img,dict_gris,n=n,salto=salto_n,umbral=umbral,use_cluster=use_cluster)
        umbral -= salto_umbral
        
    return clase, n_1 , umbral

def move_fotos_from_folder(ruta_plantillas,ruta_fotos,ruta_root,
                           create_folder=False,listar_clases=False,
                           show_cont=True, use_iter_fn=2,k=30,n=200,
                           salto_n=10,umbral=0.85, salto_umbral=0.02,
                           min_umbral=0.6,use_cluster=False, show_detalles=False):
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
        - use_iter_fn: [int] 0 para usar la deteccion directa, 1 para usar la funcion de iteracion sobre size
                     y 2 para usar la funcion de iteracion sobre size con umbral
    (Author)
        Hector Limon
    '''
    
    # obtener diccionario de nombre, ruta a planillas y array de la imagen    
    dict_plantillas = get_dict_plantilla_gris(ruta_plantillas,True,(k,k))
    
    # Creamos directorios
    if create_folder:
        for k in dict_plantillas.keys():
            _ = crear_directorio(ruta_root,k)
    # Lista de fotos
    fotos = glob.glob(ruta_fotos+'/*.png')

    # Contador
    cont = 0

    # Iteramos cada foto para generar clasificacion
    for foto in tqdm.tqdm(fotos):
        # Nombre de la foto
        name_foto = foto.replace('\\','/').split('/')[-1]
        
        # Abrir foto
        img = cv2.imread(foto)

        # Clasificacion de la foto
        if use_iter_fn in [1, '1']:
            
            name_class, n1 = clasificacion_iter_size(img,dict_plantillas,
                                                    n=n,salto=salto_n,
                                                    umbral=umbral,use_cluster=use_cluster)
            if show_detalles:
                print(cont,'  ',name_class,'  ', n1, '  ', name_foto)
            
        elif use_iter_fn in [0, '0']:
            name_class = detectar_clase(img,dict_plantillas, resize=(n,n),
                                        return_list=listar_clases,umbral=umbral,use_cluster=use_cluster)
            if show_detalles:
                print(name_class)
        elif use_iter_fn in [2,'2']:
            
            name_class, n1, umbral1 = iter_umbral_fn(img,dict_plantillas,n=n,salto_n=salto_n,
                                                    umbral=umbral,salto_umbral=salto_umbral,min_umbral=min_umbral,
                                                    use_cluster=use_cluster)
            if show_detalles:
                print(name_class,n1,umbral1)
    
        if listar_clases:
            if len(name_class)>1:
                
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
            if show_cont: print('contador = ', cont)
            cont += 1

            


def comparacion_color(center1, center2,dis_min:float=8):
    # Calcula la distancia entre cada elemento correspondiente de las dos matrices
    distancias = np.linalg.norm(center1 - center2, axis=1)

    if np.sum(distancias)<= dis_min:
        return True
    else:
        return False
    
def areasv_vs_estabgoogle(img,dict_gris):
    ''' 
    (Function)
        Esta funcion especifica que tipo clase en caso de que haya sido detectado en un principio 
        como "a_verdes" o bien "establecimiento_google" debido a que son el mismo icono, tenemos
        que hacer una segunda comparativa por el colo.
    (Parameters)
        - img: [np.array] Imagen que se desea clasificar
        - dict_gris: [dict] Diccionario de plantillas
    (Returns)
        - clase: [str] la clase correcta segun el color
    (Authors)
        - Hector Limon
    '''
    centro_plant = dict_gris['a_verdes']['centro_color']
    _, _, _, centro_img = clusterizar_color(img)
    result = comparacion_color(centro_plant,centro_img,10)
    if result:
        clase = 'a_verdes' 
    else:
        clase = 'establecimiento_google'
    return clase