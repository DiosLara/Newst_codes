import pandas as pd
import geopandas as gpd
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

def combine_cols(casas:gpd.GeoDataFrame, col_left, col_right, priority=None):
    
    # Revisamos que tengan el _ para no generar errores el nombre de las columnas
    if not col_right.find('_')>0:
        col_right1 = col_right+'_right'
        casas[col_right1] = casas[col_right]
        col_right = col_right1
    
    if not col_left.find('_')>0:
        col_left1 = col_left + '_left'
        casas[col_left1] = casas[col_left]
        col_left = col_left1
        

    # El nombre de la columna nueva sera sin el _
    if len(col_left.lower().split('_')) >2:
        new_nombre = '_'.join(col_left.lower().split('_')[0:-1])
    else:
        new_nombre = col_left.lower().split('_')[0]
    
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