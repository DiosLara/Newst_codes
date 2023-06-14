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
import xml.etree.ElementTree as ET

'''Integración de todos los elementos necesarios para el prep de bases geo y con clave catastral'''

diccionarios= {'Naucalpan':
    {'inegi': {
            'path': r'C:\Users\dlara\ECATEPEC_PREDIOS_INEGI.shp',
            'key_PREDIO': 'CVEMZA',
            'key_conteo': 'CONTEO_INEGI', # Columna en la que se va a guardar el conteo 
            'key_municipio': 'CVE_MUN',
            'key_agregacion': 'TVIVHAB'
            ,'areas_verdes' : 'path'
        },
        'predial_gcp':
        {
            'env_predial': r'C:/Users/dlara/.env',
            'oracle_path': 'INTFIS_NAUCALPAN_PADRON_MUNICIPAL_NOMINA_GEO'
        }}
}

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
    def sjoin_2phases(base_p: gpd.GeoDataFrame, base2:  gpd.GeoDataFrame, llave='ORDEN'):
        '''Spatial join en dos fases, una es un join normal y el remanente es un nearest a 10 metros máximos de distancia'''
        df= gpd.sjoin(base_p,base2)
        cut= base_p.loc[base_p[llave].isin(df[llave])]
        df_r= gpd.sjoin_nearest(cut, base2, max_distance=10)
        df_f= pd.concat([df,df_r], axis=0)
        return
    
    def curts_prep(base_shp: gpd.GeoDataFrame):
        '''preparación de bases con curt'''
        base_shp['LAT_DMS'] = base_shp['curt'].str[:11]
        base_shp['LON_DMS'] = base_shp['curt'].str[11:]
        base_shp['Latitude'] = base_shp['LAT_DMS'].astype(str).str[0:2].str.cat(base_shp['LAT_DMS'].astype(str).str[2:4], '°').str.cat(base_shp['LAT_DMS'].astype(
            str).str[4:6].astype(str) + str('.') + base_shp['LAT_DMS'].astype(str).str[6:].astype(str).str.replace('.0', '', regex=False), "'") + str("''N")
        base_shp['Longitude'] = base_shp['LON_DMS'].astype(str).str[0:2].str.cat(base_shp['LON_DMS'].astype(str).str[2:4], '°').str.cat(base_shp['LON_DMS'].astype(
            str).str[4:6].astype(str) + str('.') + base_shp['LAT_DMS'].astype(str).str[6:].astype(str).str.replace('.0', '', regex=False), "'") + str("''W")

        pi = clean_lat_long(base_shp[['Latitude','Longitude']], lat_col="Latitude",
                            long_col="Longitude", split=True)
        try: 
            base_shp.drop(columns=['Longitude','Latitude'], inplace=True)
        except:
            pass
        base_shp= pd.concat([base_shp, pi], axis=1)
    
        return(base_shp)
        
    def concat_row_puntos(grouped_df,B1 : pd.DataFrame,B2: pd.DataFrame, id_key= 'id_cat', asc=True): 
        '''Concatena dos bases que contienen la misma llave, pero la segunda debe contener puntos corregidos y generados sobre el polígono
        y en algunos casos es mayor a la base inicial, por lo que se requiere concatenar por llave
        '''
        PB2= pd.DataFrame(columns= B2.columns)
        for cve in grouped_df[id_key]:
            PBCPE = pd.concat([B1.loc[B1[id_key].astype(str).str.contains(cve)][['CLAVESXPREDIO','INTFIS_RS_FD_LON_1','INTFIS_RS_FD_LAT_1', 'geometry']].sort_values(['INTFIS_RS_FD_LON_1','INTFIS_RS_FD_LAT_1'], ascending=asc).reset_index(drop=True),B2.loc[B2[id_key].astype(str).str.contains(cve)].sort_values('CLAVECATASTRAL').drop(columns='geometry').reset_index(drop=True)], axis=1)
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
    def proceso_inegi_correccion(self,m_inegi:gpd.GeoDataFrame, conteo_censo: gpd.GeoDataFrame,base_folder=str, z_verdes:gpd.GeoDataFrame=True):
        '''Es el procesamiento de una base de inegi diferenciando PREDIOs contenedoras 
        y generando elementos como el típo de polígono y la geometría deseada
        Si las zonas verdes son necesarias, se leeran desde una carpeta especial y se aplicara un overlay (para quitarlas)'''
        
        # Corta en caso de que las PREDIOs no existen en censo y en inegi sí
        m_inegi.loc[(~m_inegi['CVEMZA_PREDIO'].isin(conteo_censo['CVEMZA_PREDIO'])), 'CVELOC']=m_inegi.loc[(~m_inegi['CVEMZA_PREDIO'].isin(conteo_censo['CVEMZA_PREDIO']))]['CVEMZA_PREDIO'].astype(str).str[:-7]
        m_inegi.loc[(~m_inegi['CVEMZA_PREDIO'].isin(conteo_censo['CVEMZA_PREDIO'])), 'CVEMZA_PREDIO']=m_inegi.loc[(~m_inegi['CVEMZA_PREDIO'].isin(conteo_censo['CVEMZA_PREDIO']))]['CVEMZA_PREDIO'].astype(str).str[:-7]
        m_inegi.loc[m_inegi.convex_hull.area> m_inegi.area, 'Tipo polígono'] = 'IRREGULAR'
        m_inegi.loc[m_inegi.convex_hull.area<= m_inegi.area, 'Tipo polígono'] = 'REGULAR'
        T_INEGI=m_inegi.loc[m_inegi['TIPOMZA']!='Contenedora'].overlay(m_inegi.loc[(m_inegi['TIPOMZA']=='Contenedora')], how='symmetric_difference')
        for cols in T_INEGI.columns[T_INEGI.columns.str.contains('_1')] : 
            T_INEGI[cols.replace('_1','')]=T_INEGI[cols].combine_first(T_INEGI[cols.replace('_1','_2')])
        T_INEGI=T_INEGI[T_INEGI.columns[~T_INEGI.columns.str.contains('_1|_2')]]
        T_INEGI2=pd.concat([T_INEGI,m_inegi.loc[m_inegi['TIPOMZA']=='Contenida']], axis=0)
        mm_inegi=pd.concat([T_INEGI2,m_inegi.loc[~m_inegi['CVEMZA'].isin(T_INEGI2['CVEMZA'])]], axis=0)
        mm_inegi=gpd.GeoDataFrame(mm_inegi, geometry='geometry', crs='EPSG:3006')
        mm_inegi.dissolve('CVEMZA_PREDIO').reset_index()
        mm_inegi['CVEMZA_PREDIO']=mm_inegi['CVEMZA_PREDIO'].astype(str)
        m1=mm_inegi.dissolve('CVEMZA_PREDIO').reset_index()
        S=m1.merge(conteo_censo.drop(columns= ['CVEMZA']), on='CVEMZA_PREDIO')
        if z_verdes== True:
            av= gpd.read_file(self.areas_verdes_path)
            av=av.to_crs(3006)
            S1 = S1.overlay(av, how='difference')
            areas_verdes='areas_verdes'
        else:
            areas_verdes=''

        S.to_file('C:/Users/dlara/ITER_AGEB_NAUCALPAN_'+str(areas_verdes)+'_20.shp', encoding='utf-8')

        return(S)

    def transform_df_to_gpd(df: pd.DataFrame,
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

    def data_prep_catastro(path_base, path_shp):
        BPCE = pd.read_excel(path_base)
        # BPCE = pd.read_csv(path_base, encoding='utf-8-sig',
        #                     header=0, engine='python')
        #BPCE['CVEMZA'] = BPCE['CVEMZA'].astype(str).str.zfill(16)
        m_igecem = gpd.read_file(path_shp) ##Lee desde shp
        #print(m_igecem.columns)
        #m_igecem = m_igecem.to_crs(4326)
        
        # try:
        #     m_igecem = m_igecem.loc[~m_igecem['manz'].astype(str).str.endswith('000')]
        # except: 
        #     pass
        # BPCE['CVEMZA'] = BPCE['ESTIMADO'].str[4:12] + '00000000'
        m_igecem.rename(columns={'CLAVECATAS':'CLAVECATASTRAL'}, inplace=True)
        m_igecem['CLAVECATASTRAL']=m_igecem['CLAVECATASTRAL'].astype(str)
        try:
            BPCE['CLAVE_PREDIO'] = BPCE['CLAVE_PREDIO'].astype(str).str.replace("\n1","",regex=False).str.replace("\n4","",regex=False).str.zfill(16)
        except:
            BPCE['CLAVE_PREDIO'] = BPCE['CLAVECATASTRAL'].str[0:10] + '000000' 
            BPCE['CLAVE_PREDIO'] = BPCE['CLAVE_PREDIO'].astype(str).str.replace("\n1","",regex=False).str.replace("\n4","",regex=False).str.zfill(16)
        BPCE.loc[BPCE['CURT'] == ' ', 'CURT'] = float('NaN')
        curts = BPCE.loc[BPCE['CURT'].astype(str).notna()]
        curts['LAT_DMS'] = curts['CURT'].astype(str).str[:11]
        curts['LON_DMS'] = curts['CURT'].astype(str).str[11:]
        curts['Latitude'] = curts['LAT_DMS'].astype(str).str[0:2].str.cat(curts['LAT_DMS'].astype(str).str[2:4], '°').str.cat(curts['LAT_DMS'].astype(
            str).str[4:6].astype(str) + str('.') + curts['LAT_DMS'].astype(str).str[6:].astype(str).str.replace('.0', '', regex=False), "'") + str("''N")
        curts['Longitude'] = curts['LON_DMS'].astype(str).str[0:2].str.cat(curts['LON_DMS'].astype(str).str[2:4], '°').str.cat(curts['LON_DMS'].astype(
            str).str[4:6].astype(str) + str('.') + curts['LAT_DMS'].astype(str).str[6:].astype(str).str.replace('.0', '', regex=False), "'") + str("''W")
        #curts_chunks = from_pandas(curts, npartitions=20) ## Se agrego por un bug que tiene clean_lat_long
        curts = clean_lat_long(curts, lat_col="Latitude",
                             long_col="Longitude", split=True)
        BPCE = BPCE.loc[BPCE['CURT'].isna()]
        BPCE = pd.concat([BPCE.loc[BPCE['CURT'].isna()], curts], axis=0)
        predios = BPCE.drop_duplicates('CLAVE_PREDIO')
        print('Esto es clave cat: ',BPCE['CLAVE_PREDIO'])
        m_igecem['CLAVE_PREDIO']= m_igecem['CLAVECATASTRAL'].str[0:10] + '000000'
       # print(m_igecem['CLAVE_PREDIO'])
        test_igecem = m_igecem.merge(BPCE.groupby('CLAVE_PREDIO').count().reset_index()[
            ['ESTIMADO', 'CURT', 'CLAVE_PREDIO']],on='CLAVE_PREDIO')
        #print(BPCE.CLAVE_PREDIO)    
        # # predios=predios.loc[(predios['CATASTRO_DOMICILIO_INMUEBLE_CONSTRUIDO'].astype(str).str.replace('S/N','NaN').str.split('NUMERO INTERIOR').str[1]!='NaN') & (predios['CATASTRO_DOMICILIO_INMUEBLE_CONSTRUIDO'].astype(str).str.replace('S/N','NaN').str.split('NUMERO INTERIOR').str[1].notna())]
        # test_igecem= m_igecem
        # test_igecem.rename(
        #     columns={'ESTIMADO': 'CLAVESXPREDIO'}, inplace=True)
        test_igecem.sort_values('ESTIMADO', ascending=False, inplace=True)
        # test_igecem=test_igecem.loc[test_igecem['CLAVESXPREDIO']>1]
        

        return(test_igecem.tail(5000))
    



def ckdnearest(gdA, gdB):
    gdA.reset_index(drop=True, inplace=True)
    gdB.reset_index(drop=True, inplace=True)
    nA = np.array(list(gdA.geometry.apply(lambda x: (x.x, x.y))))
    nB = np.array(list(gdB.geometry.apply(lambda x: (x.x, x.y))))
    btree = cKDTree(nB)
    dist, idx = btree.query(nA, k=1)
    gdB_nearest = gdB.iloc[idx].drop(columns="geometry").reset_index(drop=True)

    gdf = pd.concat(
        [
            gdA.reset_index(drop=True),
            gdB_nearest,
            pd.Series(dist, name='min_dist_2')
        ],
        axis=1)

    return gdf