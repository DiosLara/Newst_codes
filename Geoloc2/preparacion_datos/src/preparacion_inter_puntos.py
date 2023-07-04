import geopandas as gpd
import pandas as pd
#from Geolocalizacion.separacion_domicilios.funciones_diccionarios import *

#from Geolocalizacion.separacion_domicilios.funciones_diccionarios import procesamiento_con_diccionarios 
from tqdm import tqdm
import concurrent.futures
#from OracleIntfis.db_connection import OracleDB
import os
from scipy.spatial import cKDTree
from shapely.geometry import Point
from dataprep.clean import clean_lat_long
from dask.dataframe import from_pandas
import numpy as np

'''Integración de todos los elementos necesarios para el prep de bases geo y con clave catastral'''


def bases_externas_oracle(oracle_path, env):
    db = OracleDB(env)
    base_externa = db.query_to_df("SELECT * FROM "+str(oracle_path)+" *")
    
    return base_externa
class prep:
    def __init__(self, shape_pred_path: str, shape_mza_path:str, base_folder:str, oracle_path: str,areas_verdes_path:str):
        self.shape_mza_path = shape_mza_path
        self.shape_pred_path= shape_pred_path
        self.oracle_path = oracle_path
        self.base_folder = base_folder
        self.areas_verdes_path = areas_verdes_path
        # print(self.file_name)
        try:
            os.mkdir(base_folder +"/"+str(municipios)+"/")
        except:
            pass
    ###Mau
    def try_load_bases(self):
        '''Lectura de bases
        fuckit permite probar si se puede hacer un proceso, sino aplica un pass y sigue con el siguiente'''
        shapes_mza =gpd.read_file(self.shape_mza_path)
        data_pred= bases_externas_oracle(self.oracle_path, self.env)
        shapes_pred= gpd.read_file(self.shape_pred_path)
        return(shapes_mza, data_pred, shapes_pred)
    def transform_df_to_gpd(df: pd.DataFrame,
                    lon_col: str = 'LON',
                    lat_col: str = 'LAT', crs= 'EPSG:4326'):
        '''Transforma base a partir de dos columnas de latitud y longitud'''
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(
                df[lon_col], df[lat_col]
            ),
            crs=crs,
        )
        return gdf
    def sjoin_2phases(base_p: gpd.GeoDataFrame, base2:  gpd.GeoDataFrame, llave='ORDEN'):
        '''Spatial join en dos fases, una es un join normal y el remanente es un nearest a 10 metros máximos de distancia'''
        df= gpd.sjoin(base_p,base2)
        cut= base_p.loc[base_p[llave].isin(df[llave])]
        df_r= gpd.sjoin_nearest(cut, base2, max_distance=10)
        df_f= pd.concat([df,df_r], axis=0)
        return

        
    def concat_row_puntos(grouped_df,B1 : pd.DataFrame,B2: pd.DataFrame, id_key= 'id_cat', asc=True): 
        '''Concatena dos bases que contienen la misma llave, pero la segunda debe contener puntos corregidos y generados sobre el polígono
        y en algunos casos es mayor a la base inicial, por lo que se requiere concatenar por llave
        '''
        PB2= pd.DataFrame(columns= B2.columns)
        for cve in grouped_df[id_key]:
            PBCPE = pd.concat([B1.loc[B1[id_key].astype(str).str.contains(cve)][['CLAVESXPREDIO','LON','LAT', 'geometry']].sort_values(['LON','LAT'], ascending=asc).reset_index(drop=True),B2.loc[B2[id_key].astype(str).str.contains(cve)].sort_values('ID_n').drop(columns='geometry').reset_index(drop=True)], axis=1)
            PB2 = pd.concat([PB2.reset_index(drop=True), PBCPE.reset_index(drop=True)], axis=0)
        return(PB2)
    def lat_lon_final(base_geo_corr: gpd.GeoDataFrame):
        dirs= {'lat': '_LAT|Latitud|latitud', 'lon': '_LON|Longitud|longitud'}
        for n in dirs: 
                coords= base_geo_corr.columns[base_geo_corr.columns.str.contains(dirs[n])]
                base_geo_corr[coords[0]]= base_geo_corr[coords].apply(
                lambda x: x.combine_first(x.astype(str)),
                axis=1
                )
        return(base_geo_corr)

    def filter_by_mun(m: gpd.GeoDataFrame, dir_municipios: dir, filtro_a: str):
        for cve in dir_municipios['cve_inegi']:
            m.loc[m['CVE_MUN']== cve, 'MUN']= dir_municipios['cve_inegi'][cve]['Nombre']
            m.to_csv()
            return(m)


    def transform_df_to_gpd(df: pd.DataFrame,
                    lon_col: str = 'LON',
                    lat_col: str = 'LAT', crs= 'EPSG:4326'):
        '''Transforma base a partir de dos columnas de latitud y longitud'''
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(
                df[lon_col], df[lat_col]
            ),
            crs=crs,
        )
        return gdf

    def save_by_id_correcciones(base: pd.DataFrame, S: gpd.GeoDataFrame):
        return(base.to_csv('BASE_CORRECCIONES_PUNTOS_'+str(S['MUN'][0])+'.csv', encoding='utf-8-sig'))

    def combinar_PREDIOs(base, llave='CVE_MUN', buffer=0.005) -> gpd.GeoSeries:
        """
        Lee un zip con archivos shape, une las geometrias existentes, y regresa el poligono resultante
        """

        # base = gpd.read_file(shp_file_zip, crs="EPSG:4326")

        # base["union"] = 0
        base.geometry = base.geometry.buffer(buffer)
        base = base.dissolve(by=llave)

        border = base.geometry.to_crs("EPSG:4326")
        
    


        # Nos aseguramos que el resultado sea un poligono, y no un multipoligono
        # assert isinstance(border)

        return base
    def replace_columns(df):

        dict_buenas=[]

        dict_malas = []
        
        for i in range(len(dict_malas)):
            for j in range(len(dict_buenas)):
                n= dict_malas[i]
                if n in dict_buenas[j]:
                    df.rename(columns={n:dict_buenas[j]},inplace=True) 
    
    def ckdnearest(gdA, gdB, k=1):
        # gdA= puntos
        # gdB=puntos[['geometry']]
        # k=2
        gdA.reset_index(drop=True, inplace=True)
        gdB.reset_index(drop=True, inplace=True)
        nA = np.array(list(gdA.geometry.apply(lambda x: (x.x, x.y))))
        nB = np.array(list(gdB.geometry.apply(lambda x: (x.x, x.y))))
        btree = cKDTree(nB)
        dist, idx = btree.query(nA, k= k)
        if k>1:
            idx=idx.T[0]
            dist=dist.T[0]
        gdB_nearest = gdB.iloc[idx].drop(columns="geometry").reset_index(drop=True)

        gdf = pd.concat(
            [
                gdA.reset_index(drop=True),
                gdB_nearest,
                pd.Series(dist, name='min_dist_2')
            ],
            axis=1)

        return gdf
    
    
