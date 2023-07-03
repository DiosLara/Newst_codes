import pandas as pd
import geopandas as gpd
def combine_first_drop(gdf):
    for i in (gdf.columns):
        if str('_1') in i :
            n = i
        if str('_2') in i :
            k = i
            z = i[:i.rfind('_1')-1]
            gdf[z] = gdf[n]
            gdf.drop(columns=[n,k],inplace=True)
        else:
            pass
def delete_empty(df):

    for i in df.columns:
        df.loc[(df[i] =="") |(df[i] =="0") | (df[i] ==0) | (df[i].isna()) | (df[i] ==" " ), i]=float('Nan')
    df.dropna(axis=1, how='all', inplace=True)
    
    return df 
def separar_digitos(prueba):

    prueba.CLAVECATASTRAL = prueba.CLAVECATASTRAL.astype(str)
    for i in prueba.CLAVECATASTRAL:
        if str('000000') in i:
            cat = i
            cat=cat[:-6]
            prueba.loc[prueba.CLAVECATASTRAL== i, 'CLAVE_CATASTRAL']=cat
        elif str('00000000') in i :
            cat2=i
            cat2=cat2[:-8]
            prueba.loc[prueba.CLAVECATASTRAL== i, 'CLAVE_CATASTRAL']=cat2
        else:
            pass

def choose_city(gdf_zonas):
    '''
        Función para localizar el municipio a tratar dentro del shape de zonas rurales y urbanas
    '''
    zona = input('Digita el código de la zona que quieres (Ej. Ixtapan = 040):')
    gdf_mun = gdf_zonas[gdf_zonas.CVE_MUN == zona]
    #gdf_mun.plot()
    return gdf_mun
def cruce_limpieza_shapes(gdf_zonas,gdf_construcciones):
    gdf_mun = choose_city(gdf_zonas)
    gdf_mun = gdf_mun.to_crs(3857)
    gdf_cruce= gdf_construcciones.sjoin(gdf_mun,how='left')
    
    return gdf_cruce

def obtener_curt(data:gpd.GeoDataFrame, geom_col:str):
    '''
    (Function)
        Función que recibe un gdf y devuelve el mismo geodata con la clave unica de registro territorial
        en formato latitud(10 dígitos) + longitud(10 dígitos)
    (Parameters)
        - data    : GeoDataFrame al cual se le obtendrán las curt
        - geom_col: Str del nombre de la columna la cual contiene los polígonos 
    
    '''
    def convert_to_dms(coord):
        '''
    (Function)
        Funcion interna que recibe una coordenada y devuelve la misma pero en formato
        str de la forma: grados + minutos + segundos + diezmilesimas de segundo
    (Parameters)
        - coord: Coordenadas del centroide     
    '''
        degrees = int(coord)
        minutes = int((coord - degrees) * 60)
        seconds = int(((coord - degrees) * 60 - minutes) * 60)
        milliseconds = int((((coord - degrees) * 60 - minutes) * 60 - seconds) * 10000)
        return str(abs(degrees)).zfill(2)+str(abs(minutes)).zfill(2)+str(abs(seconds)).zfill(2)+str(abs(milliseconds)).zfill(4)

    data['centroid'] = data[geom_col].to_crs(4326).centroid
    data['lat'] = data['centroid'].y
    data['lon'] = data['centroid'].x
    data['CURT_f'] = data['lat'].apply(convert_to_dms)+data['lon'].apply(convert_to_dms)
    return data.drop(columns=['centroid','lat','lon'])

def combine_cols(casas:gpd.GeoDataFrame, col_left, col_right, priority=None, new_col=None):
    '''
    (Function)

    (Parameters)
        - casas; [GeoDataFrame] Contiene las columnas que se van a combinar
        - col_left; [str] Nombre de la columna que se tomara como left, de este nombre se obtiene el nuevo 
                         nombre de la columna, en caso de no especificar "new_col"
        - col_right; [str] Nombre de la columna que se tomara como right
        - priority; [str] En caso de ser None La prioridad sera tener menos nulos, pero puede especificar 
                    'left' o 'right' para dar una prioridad a las columnas
        - new_col; [str] Por default toma col_left para generar el nuevo nombre de la columna,  pero en caso de 
                        querer un nombre particular, especificarlo aqui.
    '''
    
    # Revisamos que tengan el _ para no generar errores el nombre de las columnas
    if not col_right.find('_')>0:
        col_right1 = col_right+'_right'
        casas[col_right1] = casas[col_right]
        col_right = col_right1
    
    if not col_left.find('_')>0:
        col_left1 = col_left + '_left'
        casas[col_left1] = casas[col_left]
        col_left = col_left1
        
    if new_col == None:
        # El nombre de la columna nueva en caso de no definirla sera:
        if len(col_left.lower().split('_')) >2:
            new_nombre = '_'.join(col_left.lower().split('_')[0:-1])
        else:
            new_nombre = col_left.lower().split('_')[0]
    else:
        # El nombre de la nueva columnas ya esta definido
        new_nombre = new_col
    
    # Definimos la prioridad sobre el cual se hara el combine first
    
    ## La prioridad es tener la menor cantidad de Nans
    if priority == None:
        if casas[col_right].combine_first( casas[col_left]).fillna(0).value_counts()[0] >= casas[col_left].combine_first( casas[col_right]).fillna(0).value_counts()[0]:
            casas[new_nombre] = casas[col_left].combine_first( casas[col_right])
        else:
            casas[new_nombre] = casas[col_right].combine_first( casas[col_left])
    
    ## La prioridad es la columna left
    elif priority.lower() == 'left':
        casas[new_nombre] = casas[col_left].combine_first( casas[col_right])
    
    ## La prioridad es la columna right
    elif priority.lower() == 'right':
        casas[new_nombre] = casas[col_right].combine_first( casas[col_left])
        
    # Borramos las columnas left y right
    casas.drop([col_left, col_right],axis=1,inplace=True)
    return casas

def clean_base_z(base_z):
    base_z['area']=base_z.geometry.area
    base_z=base_z[['ID_casas', 'clase_dete', 'Clase', 'id_cat','curt', 'CURT_f','ID_ind','ID_curt','CURT_si','dupGEO', 
             'dupCURT', 'YOLO_si','Ind_si','Induxcasa','ID_chin','ID_denue', 'ID_15m', 'Z','geometry','area',]]
    return base_z
    