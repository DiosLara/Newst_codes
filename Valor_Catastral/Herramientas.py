import pandas as pd
import geopandas as gpd 
from geopy.geocoders import Nominatim
from shapely.geometry import Point, Polygon, LineString
import pyproj
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import warnings
import tqdm
from scipy import stats


warnings.filterwarnings('ignore')

def corregir_formato(cadena):
    '''
    (Function)
        Esta funcion corrige el formato principalmante de la base SQL
    (Parameters)
        - cadena: [str] cadena de texto que representa un monto
    (Author)
        - Hector Limon
    '''
    if cadena[-3] == ',':
        cadena_aux = cadena[-3:]
        cadena_aux = cadena_aux.replace(',','.')
        cadena = cadena[:-4]+cadena_aux
    
    return float(cadena.replace(',',''))

def transform_df_to_gdf(df: pd.DataFrame,
                lon_col: str = 'INTFIS_RS_FD_LON_1',
                lat_col: str = 'INTFIS_RS_FD_LAT_1', crs= 'EPSG:4326'):
    '''Transforma base a partir de dos columnas de latitud y longitud'''
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(
            df[lon_col], df[lat_col]
        ),
        crs=crs,
    )
    return gdf

def validacion_zona_mz_cod(df,col_zona:str='Zona', col_cod:str='COD',
                           col_mz:str='Manzanas'):
    '''
    (Function)
        Esta funcion valida cada campo de la base para Area Homogenea
    (Parameters)
        - df: DataFrame que lee el excel
    (Author)
        Hector Limon
    '''
    
    df['len_zona'] = df[col_zona].map(lambda x: len(str(x)))
    len_zona = df['len_zona'].value_counts().index.values
    if len(len_zona) >1:
        raise ValueError(f'El largo de la zona varia, reviselos valores obtenidos {len_zona}')
    else:
        if len_zona[0] != 2:
            raise ValueError(f'El largo de la {col_zona} no coincide, es: {len_zona}, debe ser 2')
    print('Campo Zona Valido')

    df['len_cod'] = df[col_cod].map(lambda x: len(str(x)))
    len_cod = df['len_cod'].value_counts().index.values
    if len(len_cod) >1:
        raise ValueError(f'El largo del COD varia, reviselos valores obtenidos {len_cod}')
    else:
        if len_cod[0] != 3:
            raise ValueError(f'El largo de la {col_zona} no coincide, es: {len_zona} debe ser 3')
    print('Campo COD Valido')
    
    df['len_mz'] = df[col_mz].map(lambda x: len(str(x)))
    len_mz = df['len_mz'].value_counts().index.values
    if len(len_mz) >1:
        raise ValueError(f'El largo de la MZ varia, reviselos valores obtenidos {len_mz}')
    else:
        if len_mz[0] != 3:
            raise ValueError(f'El largo de la {col_zona} no coincide, es: {len_zona} debe ser 3')
    print('Campo MZ Valido')


def completacion_Campos(x:str, n:int, valor='0'):
    '''
(Function)
    Esta funcion completa el valor x (numerico o string) agregando el valor a la izquierda
(Parameters)
    - x: Valor a completar
    - n: largo al que debe llegar x
    - valor: Caracter que pondra a la izquierda en caso de tener una longitud menor a n
(Example)
    >>completacion_Campos('3',4,)
    0003
(Author)
    - Hector Limon
    '''
    x = str(x)
    while len(x)<n:
        x = valor+x
    return x

def get_adrress(latlon:str='', poligono:str='', user_agent='kaggle_learn'):
    '''
(Function)
    Esta funcion recibe latitud y lonngitud y regresa la direccion
(Parameters)
    latlon: Latitud y longitud como estring separadas por una coma, ejemplo: '19.4852442,	-99.2798696'
(Autor)
    Hector Limon
    '''
    if (isinstance(latlon, str)) :
        geolocator = Nominatim(user_agent=user_agent)
        location = geolocator.reverse(latlon)
        return location.address
    
def Categorizar_por_intervalos(x:int, li:int, ls:int):
    '''
(Function)
    Esta funcion retorna un 1 si el valor ingresado esta en el intervalo definido
    de  lo contrario regresa un 0, en caso de ser un tipo de dato no valido
    returna un 9
(Parameters)
    - x: Valor que se desea clasificar
    - li: Limite inferior
    - ls: Limite superior
(Author)
    Hector Limon
    '''
    if isinstance(x,int) or isinstance(x,float):
        if x<=ls and x>= li:
            return 1
        else:
            return 0
    else:
        return 9


def get_lados_xy(poligono, n_digits:int=4):
    
    '''
(Function)
    Esta funcion calcula la distancia del poligono asumiendo que es un paralelogramo, es decir toma el valor minimo en el eje x y el maximo
    para obtener la distancia en x, y lo mismo en el eje y. No importa que tipo de poligono le pases, siempre returna 2 valores, el valor del
    lado x, y el del lado y.
    Muy importante, asegurarse que su crs sea 3857, de lo contrario, los valores retornados seran otra interpretacion.
(Parameters):
    - poligono: corresponde a un registro de la GeoSerie geometry
    - n_digits: cantidad de digitos a la que se redondeara la distancia
    '''
    if isinstance(poligono,Polygon):
        valores = poligono.bounds
        ladox = round(abs(valores[0] - valores[2]),n_digits)
        ladoy = round(abs(valores[1] - valores[3]),n_digits)
        return ladox, ladoy



def get_distance_vertices(poligono, get_distancia:str='euclidiana',ver_distancias:bool=False, n_digits:int=4 ):
    '''
(Function)
    Esta funcion calcula los lados de un poligono, basicamente obtiene las cordenadas de cada vertice y luego calcula la distancia entre un 
    vertice y otro, solo trae valores unicos, por lo que para un paralelogramo regular, solo returna 2 valores, ya que en realidad son 4, pero 
    2 de ellos estan repetidos. En general returna valores unicos de los valores de lados
(Parameters)
    - poligono: es un poligono determinado en la geometry de cualquier geoserie
    - get_distancia: Si tu crs es 6364, usa 'euclidiana', pero si tu crs es 3857 usa 'point'
    - ver_distancias: True para imprimir las distancias
    - n_digits: Cantidad de digitos a la que se redondeara la distancia
(Author)
    Hector Limon
    '''
    coords = poligono.exterior.coords
    if get_distancia.lower() == 'euclidiana':
        # definir la proyección cartográfica
        proyeccion = pyproj.Proj(proj='merc', lat_ts=0, lon_0=0, x_0=0, y_0=0, ellps='WGS84')
    l_distancias = np.array([])
    for i in range(len(coords)-1):
        if get_distancia.lower() == 'euclidiana':
            # convertir las coordenadas a coordenadas planas
            x1, y1 = proyeccion(coords[i][0], coords[i][1])
            x2, y2 = proyeccion(coords[i+1][0],coords[i+1][1])
            # calcular la distancia euclidiana
            distancia = round(sqrt((x2 - x1)**2 + (y2 - y1)**2),n_digits)
        elif get_distancia.lower() == 'point':
            # print(type(coords[i]))
            p1 = Point(coords[i])
            # print(coords[i+1])
            p2 = Point(coords[i+1])

            distancia = round(p1.distance(p2),n_digits)
        else:
            distancia = None

        if ver_distancias:
            print(f"La distancia entre los vértices {i} y {i+1} es de {distancia} unidades.")
        l_distancias = np.append(l_distancias, distancia)
    return np.unique(l_distancias)

def find_corner_polygons(df_large,df_small, how_corner:str='difference'):
    '''
(Function)
    Esta funcion recibe dos GeoDataFrames uno que contiene un poligono grande y el otro 
    contiene poligonos pequeños contenidos dentro del grandote, y lo que retorna esta funcion
    sera todos aquellos poligonos que estan en la frontera interna
(Parameters)
    - df_large: GeoDataFrame que contiene el poligono grande
    - df_small: GeoDataFrame que contiene los poligonos pequeños que se supone estan dentro 
                del grande
    - how_corner: tipo de overlay que hara, puede usar tambie symmetric_difference pero esto genera el doble 
                 de columnas
(Author)
    Hector Limon
    '''
    # Encuentra la intersección entre los polígonos pequeños y el polígono grande
    intersection = gpd.overlay(df_small, df_large, how='intersection')

    # Encuentra los polígonos pequeños que están completamente dentro del polígono grande
    difference = gpd.overlay(df_small, intersection, how='difference')

    # Encuentra la frontera del polígono grande
    boundary = gpd.GeoDataFrame(geometry=df_large.boundary)

    # Encuentra los polígonos pequeños que intersectan con la frontera del polígono grande
    intersects = gpd.sjoin(df_small, boundary, how='inner', op='intersects')

    # Encuentra los polígonos que están en las esquinas por dentro del polígono grande
    corner_polygons = gpd.overlay(intersects, difference, how=how_corner, )

    # Imprime los polígonos que están en las esquinas por dentro del polígono grande
    # print(corner_polygons.shape)
    return corner_polygons

def get_distancia_vertices_to_frontera(poligono1, poligono2, show_distance:bool=False, clasifier:float=None,
                                       show_map:bool=False, show_inersection_line:bool=False):
    '''
(Function)
    Esta funcion mide la distancia mas corta de los vertices del poligono2 al contorno del poligono1
    puede especificar clasifier para detectar una distancia minima, por ejemplo, si pone 3, y encuentra al 
    menos un vertice cuya distancia al contorno sea menor o igual que 3 la funcion retorna 1 (esto sirve para de
    tectar algunos casos particulares), en caso de no especificar clasifier con ningun numero, la funcion 
    retorna un array con las distancias calculadas
(Parameters)
    - poligono1: geometria del poligono mas grande
    - poligono2: geometria del poligono pequeño
    - show_distance: True para imprimir las distancias
    - clasifier: Especificar en caso de quere clasificar la distancia;  returna 1 si distancia <= clasifier
    - show_map: True para visualizar los mapas, funciona si clasifier no se cumple.
(Example)
    cve_cat = '0980540500000000'
    poligono2 = gdf[gdf['cve_cat']==cve_cat]
    poligono2.reset_index(drop=True, inplace=True)
    poligono = shape[shape['cve_cat']==cve_cat]
    poligono.reset_index(drop=True, inplace=True)
    poligono = poligono['geometry'].iloc[0]
    poligono2 = poligono2['geometry'].iloc[12]
    get_distancia_vertices_to_frontera(poligono, poligono2, show_distance=True, clasifier=None,
                                        show_map=True, show_inersection_line=1)
(Author)
    Hector Limon
    '''
    # Generamos el  permietro del poligono1
    line = LineString(poligono1.exterior.coords)
    # Listas que almacenaran info
    puntos = np.array([])
    distancias = np.array([])
    # Coordenadas de los vertices del poligono1
    coords = poligono2.exterior.coords
    if show_map:
        fig, ax = plt.subplots()
        gpd.GeoSeries(line).plot(ax=ax)
        gpd.GeoSeries([Point(z) for z in coords]).plot(ax=ax, color='red')
        plt.show()
    # Revisamos la distancia minima del vertice al perimetro del poligono1
    for c in coords:
        point = Point(c)
        puntos = np.append(puntos, point)
        distancia = point.distance(line)
        distancias = np.append(distancias, distancia)
        if show_distance:
            print(distancia)
        # La distancia es menor o iggual al criterio retornamos 1
        if (isinstance(clasifier,float) or isinstance(clasifier,int)) and distancia <= clasifier:
            return 1
    
    # Para llegar aqui o no hay criterio o nunca se cumplio el criterio
    for i in range(len(coords)-1):
        line_c = LineString([coords[i],coords[i+1]])
        if show_inersection_line:
            print(line_c.intersects(line))
        if (line_c.intersects(line)):
            return 1
    return np.unique(distancias)
    
    
def statistics_values(muestra):
    '''
(Function)
    Esta funcion obtiene estadisticos relevante de la muestra, cabe señalar que toma
    el array de la muestra de manera que ninguno sea 0, es decir si alguno elemento 
    de la muestra es 0 lo borra, para no sesgar los datos, ademas toma los percentiles
    del al 100, con un salto de 10, por lo que regresa una cadena de texto en el 
    siguiente orden
    - Minimo, Maximo, Media, desviacion_estandar, largo, media_percentil, desviacion_std_percentiles, moda_percentiles

    Esta funcion esta diseñada para usarla en un DataFrame de manera que se pueda usar;
    str.split('|',expand=True)
    para desglosar los valores.
(Parameters)
    -muestra: array con elementos para calcular los estadisticos
(Example)
    df.apply(lambda x: statistics_values(x[cols_terreno]), axis=1)
    '''
    muestra = muestra.values
    # print('Shape original ', len(muestra))
    muestra = np.delete(muestra, np.where(muestra==0))
    if len(muestra)==0:
        return 'Revisar'
    l_p = [10,20,30,40,50,60,70,80,90,100]
    p = np.percentile(muestra, l_p, method='nearest')
    # print('Shape final ', len(muestra))
    # print('---------------------------')
    
    return str(np.min(muestra)) +'|'+ str(np.max(muestra))+'|'+ str(np.mean(muestra))+'|'+ str(np.std(muestra))+'|'+ str(len(muestra))+'|'+ str(np.mean(p))+'|'+ str(np.std(p))+'|'+ str(stats.mode(p,keepdims=True)[0][0])
    
def transponer_datos(df,col_id, cols_terreno, cols_construccion):
    '''
(Function)
    Esta funcion genera dos DataFrames para separar segun las columnas, la idea principal es que toma toda las columnas de cols_terreno
    y las transpone para generar solo 1 columna, de igual manera con las columnas cols_construccion, de manera que tendremos 2 DataFrames
    cuyas columnas seran:
    - Tipo_Valor: Concerniente al nombre de la columna en el DataFrame original
    - V_Terreno:  El valor numerico 
    - Indice_Original: Indice para encontrarlo en el DataFrame original
(Parameters)
    - df: DataFrame original, que contiene muchas columnas y se desea transponer
    - col_id: Nombre de la columna que funge de ID en el DataFrame
    - cols_terreno: Lista con los nombres de columnas que contiene valores catastrales del terreno
    - cols_construccion: Lista con los nombres de las columnas que contiene valores catastrales de la construccion
(Example)
    df_construccion, df_terreno = transponer_datos(df_prueba,col_id='Indice_Modelo', cols_terreno=cols_terreno,cols_construccion=cols_const)
(Author)
    Hector Limon
    '''
    df_final_terreno = pd.DataFrame()
    df_final_construccion = pd.DataFrame()
    for indice in df[col_id]:
        # print(indice)
        df_terreno = df[df[col_id]==indice][cols_terreno].T
        df_terreno = pd.DataFrame({'V_Terreno':df_terreno[indice].values}, index=df_terreno.index)
        df_terreno.reset_index(drop=False,inplace=True)
        df_terreno.rename(columns={'index':'Tipo_Valor'}, inplace=True)
        df_terreno['Indice_Original'] = indice
        df_construcion = df[df[col_id]==indice][cols_construccion].T
        df_construcion = pd.DataFrame({'V_Construccion':df_construcion[indice].values}, index=df_construcion.index)
        df_construcion.reset_index(drop=False, inplace=True)
        df_construcion.rename(columns={'index':'Tipo_Valor'}, inplace=True)
        df_construcion['Indice_Original'] = indice

        df_final_terreno = pd.concat([df_final_terreno, df_terreno])
        df_final_construccion = pd.concat([df_final_construccion, df_construcion])

    df_final_terreno.reset_index(drop=True, inplace=True)
    df_final_construccion.reset_index(drop=True, inplace=True)
    return df_final_construccion, df_final_terreno
    
def transponer_df(df_prueba, cols_transponer, col_id='Indice_Modelo') -> pd.DataFrame:
    '''
(Function)
    Esta funcion genera dos DataFrames para separar segun las columnas, la idea principal es que toma toda las columnas de cols_transponer
    y las transpone para generar solo 1 columna
(Parameters)
    - df_prueba: DataFrame original el cual se desea transponer
    - cols_transponer: Lista con las columnas que se desea tener en filas
    - col_id: Identificador del DataFrame, este se repetira tantas veces como columnas haya
(Example)
    transponer_df(df_prueba=df,
                 cols_transponer=cols_terreno,)
(Author)
    Hector Limon

    '''
    df_apilado = df_prueba.melt(id_vars=[col_id],
                            value_vars=cols_transponer,
                            var_name='Tipo_val',
                            value_name='Valor_catastral')
    # Ordenar el DataFrame apilado por 'Id' y 'Tipo'
    df_apilado = df_apilado.sort_values(by=[col_id, 'Tipo_val'])

    # Restablecer los índices del DataFrame apilado
    df_apilado = df_apilado.reset_index(drop=True)
    return df_apilado


def get_polygons_nearest_perimeter(falta_fp:gpd.GeoDataFrame,shape:gpd.GeoDataFrame,
                                   col_cve_cat:str,
                                   distancia_max:float,
                                   col_id:str='Indice_gdf'):
    '''
(Function)
    Esta funcion revisa si los poligonos pequeño (falta_fp) estan cerca del perimetro de los 
    poligonos grandes (shape) de manera que si la distancia entre el pequeño al vertice es menor 
    o igual que "ditancia_max" se asume que esta en el perimetro
(Parameters)
    - falta_fp: GeoDataFrame que contiene los poligonos pequeños
    - shape: GeoDataFrame que contiene los poligonos grandes
    - col_cve_cat: Nombre de la columna que tiene la clave catastral
    - distancia_max: Distancia maxima permitida para asumir que esta en el perimetro del poligono
                     grande
    - col_id: Nombre de la columna que tiene un identificador en falta_fp
(Author)
    Hector Limon
    '''
    nuevos = pd.DataFrame()
    count = 0

    for cve_cat in tqdm.tqdm(falta_fp[col_cve_cat].unique()):
        falta_fp.reset_index(drop=True, inplace=True)
        # Poligonos pequeños
        df_poligono2 = falta_fp[falta_fp[col_cve_cat]==cve_cat]
        # Poligonos grandes
        df_poligono = shape[shape[col_cve_cat]==cve_cat]

        for poligono in df_poligono['geometry']:
            for poligono2 in df_poligono2['geometry']:
                calif = get_distancia_vertices_to_frontera(poligono, poligono2, clasifier=distancia_max,
                                                        show_distance        =0, 
                                                        show_map             =0, 
                                                        show_inersection_line=0)
                count += 1
                if isinstance(calif, int) and calif == 1:
                    nuevos = pd.concat([nuevos, falta_fp.iloc[df_poligono2[df_poligono2['geometry']==poligono2].index][[col_id,'geometry']]])
                    
                    continue
            
        
    nuevos = gpd.GeoDataFrame(nuevos, geometry='geometry')
    print('\n Faltan ', falta_fp.shape[0],' Claves y se revisaron ',count,' Claves')
    print('Se encontraron: ',len(nuevos[col_id].unique()),'\n Shape final ->',nuevos.shape )
    return nuevos

def cruzar_chinchetas_vs_dnue(path_chinchetas,path_dnue,municipio=None,dist_max=10, 
                              schema_crs={'set_ch':3857,'to_ch':3857,
                                        'set_dnue':6364,'to_dnue':3857}, 
                              show_crs=False, show_results=True):
    '''
    (Function)
        Esta funcion cruza y agrupa los posibles nombres que se obtienen al poligono
        de una chincheta con relacion a la informacion extraida en dnue.
    (Parameters)
        - path_chinchetas: [str|GeoDataFrame] relativo a las chinchetas
        - path_dnue: [str|GeoDataFrame] relativo a dnue
        - municipio: [str] Nombre del municipio que desea analizar en relacion a dnue opcional
                    se considera para ver la numeralia con show_results
        - dist_max: [int|float] distancia maxima para considerar en el cruce, en relacion con schema_crs "to_ch and to_dnue"
        - schema_crs: Respete las llaves, en caso de abrir el shape, setea el shape con el valor de set,
                     y el to_ es para hacer una reproyecciones, estos deben ser iguales
        - show_crs: [bool] True para ver el crs que tiene cada shape
        - show_results: [bool] True para ver la numeralia del cruce, tendra mas sentido si pone el muncipio
    '''
    if isinstance(path_chinchetas,str):
        # Cargamos chinchetas
        gdf_ch = gpd.read_file(path_chinchetas)
        # print(gdf_ch.crs)
        gdf_ch = gdf_ch.set_crs(schema_crs['set_ch'],allow_override=True)
    elif isinstance(path_chinchetas,gpd.GeoDataFrame):
        gdf_ch = path_chinchetas
    else:
        raise ValueError('Definio mal path_chinchetas, debe ser un GeoDataFrame o bien la ruta al shape')
    gdf_ch = gdf_ch.to_crs(schema_crs['to_ch'])

    if isinstance(path_dnue,str):
        # Cargamos dnue
        gdf_dnue = gpd.read_file(path_dnue)
        # Seteamos el crs en caso de leerlo
        gdf_dnue   = gdf_dnue.set_crs(schema_crs['set_dnue'],allow_override=True)
    elif isinstance(path_dnue,gpd.GeoDataFrame):
        gdf_dnue = path_dnue
    else:
        raise ValueError('Definio mal path_dnue, debe ser un GeoDataFrame o bien la ruta al shape')
    gdf_dnue = gdf_dnue.to_crs(schema_crs['to_dnue'])

    if show_crs:
        print('CRS chinchetas: ',gdf_ch.crs)
        print('CRS Dnue      : ',gdf_dnue.crs)
    
    # Filtro
    if municipio != None:
        gdf_dnue = gdf_dnue[gdf_dnue['municipio']==municipio]

    # Primer cruce por interseccion
    cruce1 = gpd.sjoin(gdf_ch, gdf_dnue,
          how='left',)
    
    # Ids encontrados
    ids_find = cruce1[cruce1['id'].fillna('vacio')!='vacio'].id.astype(float).unique()
    ids_find = [int(x)  for x in ids_find]

    # Hacemos todo str menos la geometry para agrupar
    cruce1.fillna('vacio', inplace=True)
    for col in cruce1:
        if col == 'geometry':
            print('omitio ', col)
            continue
        cruce1[col] = cruce1[col].astype(str)

    '''
    Luego de hacer un primer cruce, es razonable pensar que un id, este en 1 o mas poligonos de chinchetas, las razones pueden ser; <br>
    - Es una plaza <br>
    - Es un mercado <br>
    - La separacion entre poligonos es minima <br>
    - otra <br>

    Entonces compactaremos todos esos id, en un solo registro de manere que concatenemos los posibles id que se corresponden a esa chincheta.
    '''
    # Agrupamos para concatenar los datos duplicados de cada geometria
    #print('Antes de cruce')
    cruce1 = pd.DataFrame(cruce1)
    cruce1_ = cruce1.groupby('geometry', as_index=False,sort=False).agg({'tipoCenCom': '|'.join, 'tipo_asent': '|'.join,
                                    'nombre_act': '|'.join,
                                    'id': '|'.join, 'clee': '|'.join})#.reset_index(drop=True)
    # Hacemos GeoDataFrame
    #print('Postcruce')
    cruce1_ = gpd.GeoDataFrame(cruce1_, geometry='geometry')

    # Asignamos lo encontrado al de chinchetas

    for i in tqdm.tqdm(cruce1_[cruce1_['id'].fillna('vacio')!='vacio'].index):
        gdf_ch.loc[gdf_ch['geometry'] == cruce1_.loc[i,'geometry'],'tipoCenCom'] = cruce1_.loc[i,'tipoCenCom']
        gdf_ch.loc[gdf_ch['geometry'] == cruce1_.loc[i,'geometry'],'tipo_asent'] = cruce1_.loc[i,'tipo_asent']
        gdf_ch.loc[gdf_ch['geometry'] == cruce1_.loc[i,'geometry'],'nombre_act'] = cruce1_.loc[i,'nombre_act']
        gdf_ch.loc[gdf_ch['geometry'] == cruce1_.loc[i,'geometry'],'clee'] = cruce1_.loc[i,'clee']
        gdf_ch.loc[gdf_ch['geometry'] == cruce1_.loc[i,'geometry'],'id']   = cruce1_.loc[i,'id']
    
    # Buscamos los que no hicieron match de chinchetas y de dnue
    falta_dnue = gdf_dnue[~gdf_dnue.id.isin(ids_find)]
    falta_ch = gdf_ch[gdf_ch.id.fillna('vacio')=='vacio']

    # Hacemos un segundo cruce por vecinos cercanos tomando distancia maxima
    cruce2 = gpd.sjoin_nearest(falta_ch[['Clase', 'geometry']], falta_dnue,
          how='left',max_distance=dist_max,distance_col='distancias',)
    
    # Hacemos todo str menos la geometry para agrupar
    cruce2.fillna('vacio', inplace=True)
    for col in cruce2:
        if col == 'geometry':
            print('omitio ', col)
            continue
        cruce2[col] = cruce2[col].astype(str)
    
    # Agrupamos para concatenar los datos duplicados de cada geometria
    cruce2_ = cruce2.groupby('geometry', as_index=False,sort=False).agg({'tipoCenCom': '|'.join, 'tipo_asent': '|'.join,
                                    'nombre_act': '|'.join,
                                    'id': '|'.join, 'clee': '|'.join}).reset_index(drop=True)
    # Hacemos GeoDataFrame
    cruce2_ = gpd.GeoDataFrame(cruce2_, geometry='geometry')

    # Asignamos los valores encontrados
    for i in tqdm.tqdm(cruce2_[cruce2_['id'].fillna('vacio')!='vacio'].index):
        gdf_ch.loc[gdf_ch['geometry'] == cruce2_.loc[i,'geometry'],'tipoCenCom'] = cruce2_.loc[i,'tipoCenCom']
        gdf_ch.loc[gdf_ch['geometry'] == cruce2_.loc[i,'geometry'],'tipo_asent'] = cruce2_.loc[i,'tipo_asent']
        gdf_ch.loc[gdf_ch['geometry'] == cruce2_.loc[i,'geometry'],'nombre_act'] = cruce2_.loc[i,'nombre_act']
        gdf_ch.loc[gdf_ch['geometry'] == cruce2_.loc[i,'geometry'],'clee']       = cruce2_.loc[i,'clee']
        gdf_ch.loc[gdf_ch['geometry'] == cruce2_.loc[i,'geometry'],'id']         = cruce2_.loc[i,'id']

    # Encontrados en el segundo cruce
    ids_find2 = cruce2[cruce2['id'].fillna('vacio')!='vacio'].id.astype(float).unique()
    ids_find2 = [int(x)  for x in ids_find2]

    if show_results:
        print('\nDnue ids encontrados    : ', len(ids_find2)+len(ids_find))
        print('Dnue faltan por asignar : ',gdf_dnue.shape[0]-len(ids_find)-len(ids_find2))
        print('Dnue total de id en dnue: ', gdf_dnue.shape[0])
        print('Chinchetas se encontraron     : ', gdf_ch[gdf_ch.id.fillna('vacio')!='vacio'].shape[0])
        print('Chinchetas falta por encontrar: ', gdf_ch[gdf_ch.id.fillna('vacio')=='vacio'].shape[0])
        print('Total chichentas              : ', gdf_ch.shape[0])
    return gdf_ch

 def agrupar_cruce_chinchetas( cruce1, cols_agg):
    '''
    Luego de hacer un primer cruce, es razonable pensar que un id, este en 1 o mas poligonos de chinchetas, las razones pueden ser; <br>
    - Es una plaza <br>
    - Es un mercado <br>
    - La separacion entre poligonos es minima <br>
    - otra <br>

    Entonces compactaremos todos esos id, en un solo registro de manere que concatenemos los posibles id que se corresponden a esa chincheta.
    '''
    # Hacemos todo str menos la geometry para agrupar
    for col in cols_agg:
        if col == 'geometry':
            print('omitio ', col)
            continue
        cruce1[col] = cruce1[col].astype(str)


    # Creamos diccionario de agrupacion
    dict_agg = {}
    for k in cols_agg:
        dict_agg[k] = '|'.join


    # Agrupamos para concatenar los datos duplicados de cada geometria
    cruce1_ = cruce1.groupby('geometry', as_index=False,sort=False).agg(dict_agg).reset_index(drop=True)
    # Hacemos GeoDataFrame
    cruce1_ = gpd.GeoDataFrame(cruce1_, geometry='geometry')
    
    # Encontrar los repetidos
    ind_f = pd.Series(cruce1_.id.str.replace('nan','').str.strip('|').value_counts().index)
    ind_f = ind_f.map(lambda y: y if str(y).find('|')>0 else 'vacio')
    ind_f[ind_f != 'vacio']
    
    # Borramos duplicados
    print('Shape original: ', cruce1.shape)
    cruce1.drop_duplicates(['geometry','cve_cat'],inplace=True)
    print('Shape final  : ',cruce1.shape)

    # Asignamos lo encontrado al de chinchetas
    for i in tqdm.tqdm(cruce1_[cruce1_.id.isin(ind_f)].index):
        for col in cols_agg:
            cruce1.loc[cruce1['geometry'] == cruce1.loc[i,'geometry'], col] = cruce1_.loc[i,col]
    return cruce1