import pandas as pd
import geopandas as gpd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from shapely.geometry import MultiPoint
from tqdm import tqdm
import shapely
import concurrent.futures
import os
import sys
import math
sys.path.append(r'C:\Users\dlara\Proyectos_git\geoloc2\preparacion_datos\src')
from preparacion_inter_puntos import prep
'''Interpolación de puntos'''
def _to_2d(x, y, z):
    return tuple(filter(None, [x, y]))

### TODO: Pasar a la clase de prep
def combinar_manzanas(base, llave='CVE_MUN', buffer=0.005) -> gpd.GeoSeries:
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

def points_bounds(df, simplify=True):
    '''Interpola puntos generados a partir de la proporción de claves sobre el polígono'''
    geom = df.geometry
    simplify=True
    list_points = gpd.GeoSeries()
    if simplify==True:
        n = gpd.GeoSeries(geom.convex_hull.boundary)
    else:
        n = gpd.GeoSeries(geom.boundary)
    n = n.to_crs('epsg:3857')
    distance_delta = int(n.geometry.length.unique())/(int(df.ESTIMADO))
    
    distances = np.arange(0, n.geometry.length.unique(), float(distance_delta))
    '''Aquí se aplica la interpolación'''

    for distance in distances:
        points = n.geometry.interpolate(distance)
        points = gpd.GeoSeries(points)
        list_points = list_points.append(points)
       
    return(list_points, distance_delta)

def points_polis(df):
    """
        Aplica buffer dentro de la linea de punto de un poligono, este buffer cambia dependiendo el tipo de tamaño, si el poligono es irregular
        y si es muy pequeño. Apartir de esta lista marca limites que caen dentro del poligono
        distance_delta ==> genera puntos dentro del poligono 
    """
    df.sort_values('ESTIMADO', ascending=False, inplace=True)
    df = df.to_crs('epsg:3857')
    
    # df['ESTIMADO']=df['ESTIMADO'].astype(int)
    ndb2= gpd.GeoDataFrame(columns=[0])
    ndb= gpd.GeoDataFrame(columns=[0])

    '''Se generan dos bases m1 y m2 para analizar 
    como se reparten los puntos sobre dos direcciones contrarias, y se debe asegurar que tengan la misma longitud'''
    
    if (df.ESTIMADO!=1)[0]:
  
        
        if (df.geometry.convex_hull.area > df.geometry.area)[0]: 
         
            for i in range(1,4):
                list_points, distance_delta = points_bounds(df, simplify=True)
               
                ndb0 = gpd.GeoDataFrame(gpd.points_from_xy(gpd.GeoSeries(list_points).buffer(
                    float(distance_delta*(1.5))*i).bounds['maxx'], gpd.GeoSeries(list_points).buffer(float(distance_delta*(1.3))*i).bounds['maxy']))
                ndb00 = gpd.GeoDataFrame(gpd.points_from_xy(gpd.GeoSeries(list_points).buffer(
                    float(distance_delta*(1.5))*i).bounds['minx'], gpd.GeoSeries(list_points).buffer(float(distance_delta*(1.3))*i).bounds['miny']))
                ndb2 =pd.concat([ndb2 , ndb0], axis=0)
                ndb =pd.concat([ndb , ndb00], axis=0)
                
            ndb.crs='epsg:3857'
            ndb2.crs='epsg:3857'
           
            # print(gpd.sjoin(gpd.GeoDataFrame(df).set_geometry('geometry'), gpd.GeoDataFrame(
            #     ndb2).set_geometry(0)).dropna(how='all'))
        
            m2 = gpd.sjoin(gpd.GeoDataFrame(df).set_geometry('geometry'), gpd.GeoDataFrame(
                ndb2).set_geometry(0), how='right',predicate='intersects').dropna(how='all')
            
        
            m1 = gpd.sjoin(gpd.GeoDataFrame(df).set_geometry('geometry'), gpd.GeoDataFrame(
                ndb).set_geometry(0), how='right', predicate='intersects').dropna()
        
            
        elif (df.geometry.convex_hull.area <= df.geometry.area)[0] & (df.geometry.area[0]>50000):
         
            for i in range(1,5):
                list_points, distance_delta= points_bounds(df, simplify=False )

                ndb0 = gpd.GeoDataFrame(gpd.points_from_xy(gpd.GeoSeries(list_points).buffer(
                    float(distance_delta)*2*i).bounds['maxx'], gpd.GeoSeries(list_points).buffer(float(distance_delta)).bounds['maxy']),geometry=0)
                ndb00 = gpd.GeoDataFrame(gpd.points_from_xy(gpd.GeoSeries(list_points).buffer(
                    float(distance_delta)*2*i).bounds['minx'], gpd.GeoSeries(list_points).buffer(float(distance_delta)).bounds['miny']),geometry=0)
                ndb2 =pd.concat([ndb2 , ndb0], axis=0)
                ndb =pd.concat([ndb , ndb00], axis=0)
            ndb.crs='epsg:3857'
            ndb2.crs='epsg:3857'
            m2 = gpd.sjoin(gpd.GeoDataFrame(df).set_geometry('geometry'), gpd.GeoDataFrame(
                ndb2).set_geometry(0), how='right', predicate='intersects').dropna(how='all')

        
            m1 = gpd.sjoin(gpd.GeoDataFrame(df).set_geometry('geometry'), gpd.GeoDataFrame(
                ndb).set_geometry(0), how='right', predicate='intersects').dropna()
        elif (df.geometry.convex_hull.area <= df.geometry.area)[0] & (df.geometry.area[0]<=50000) & (df.geometry.area[0]>=6000):
          
            for i in range(1,4):
                list_points, distance_delta = points_bounds(df, simplify=False )
                  
                ndb0 = gpd.GeoDataFrame(gpd.points_from_xy(gpd.GeoSeries(list_points).buffer(
                    float(distance_delta)*1.09*i).bounds['maxx'], gpd.GeoSeries(list_points).buffer(float(distance_delta)).bounds['maxy']),geometry=0)
                ndb00 = gpd.GeoDataFrame(gpd.points_from_xy(gpd.GeoSeries(list_points).buffer(
                    float(distance_delta)*1.09*i).bounds['minx'], gpd.GeoSeries(list_points).buffer(float(distance_delta)).bounds['miny']),geometry=0)
                ndb2 =pd.concat([ndb2 , ndb0], axis=0)
                ndb =pd.concat([ndb , ndb00], axis=0)
            ndb.crs='epsg:3857'
            ndb2.crs='epsg:3857'

            m2 = gpd.sjoin(gpd.GeoDataFrame(df).set_geometry('geometry'), gpd.GeoDataFrame(
                ndb2).set_geometry(0), how='right', predicate='intersects').dropna(how='all')

        
            m1 = gpd.sjoin(gpd.GeoDataFrame(df).set_geometry('geometry'), gpd.GeoDataFrame(
                ndb).set_geometry(0), how='right', predicate='intersects').dropna()
           
            
        elif (df.geometry.area[0]<6000) & (df.geometry.convex_hull.area <= df.geometry.area)[0] :
     
            for i in range(1,2):
                list_points, distance_delta = points_bounds(df, simplify=False )
                
                ndb0 = gpd.GeoDataFrame(gpd.points_from_xy(gpd.GeoSeries(list_points).buffer(
                float(distance_delta)*1.4*i).bounds['maxx'], gpd.GeoSeries(list_points).buffer(float(distance_delta)).bounds['maxy']),geometry=0)
                ndb00 = gpd.GeoDataFrame(gpd.points_from_xy(gpd.GeoSeries(list_points).buffer(
                float(distance_delta)*1.4*i).bounds['minx'], gpd.GeoSeries(list_points).buffer(float(distance_delta)).bounds['miny']),geometry=0)
                ndb2 =pd.concat([ndb2 , ndb0], axis=0)
                ndb =pd.concat([ndb , ndb00], axis=0)
            ndb.crs='epsg:3857'
            ndb2.crs='epsg:3857'
            m2 = gpd.sjoin(gpd.GeoDataFrame(df).set_geometry('geometry'), gpd.GeoDataFrame(
                ndb2).set_geometry(0), how='right', predicate='intersects').dropna(how='all')

        
            m1 = gpd.sjoin(gpd.GeoDataFrame(df).set_geometry('geometry'), gpd.GeoDataFrame(
                ndb).set_geometry(0), how='right', predicate='intersects').dropna()
     
        if ((len(m1)==len(m2))| (len(m1)==0) |(len(m2)==0)):
            pass
        elif len(m2)> len(m1):
            m2=m2.iloc[:len(m1)]
        elif len(m1)>len(m2):
            m1=m1.iloc[:len(m2)]

        '''Estas características se aplicarán fuera de este py, 
        si la clave cat ==1 implica que es único y un posible domicilio único,
         por lo que sólo se saca el centroide como se acostumbraba inicialmente'''
        
                
                
    elif (df.ESTIMADO==1)[0]:
  
        
        ndb2= df
        ndb2[0]= ndb2.centroid
        m2 = ndb2
        m1=ndb
  
    return(m2, m1)

def simulacion_poli(chunks):    
    '''Ciclo a partir de la cve cat nivel predio y aplica las funciones arriba,
     una vez aplicada la que genera los puntos sobre el polígono,
      convierte de nuevo m1 y m2 a geometría y genera las coordenadas en columnas separadas'''
    df_final = pd.DataFrame(columns=['index_left', 0, 'CLAVE_PREDIO'])
    for i ,cve in tqdm(enumerate(chunks['CLAVE_PREDIO']),total = len(chunks)):
        df = chunks.loc[chunks['CLAVE_PREDIO'].str.contains(cve)]
        
        assert any(df['ESTIMADO']>0)
        df = combinar_manzanas(df, llave='CLAVE_PREDIO')
        
        df.reset_index(inplace=True)
        df.drop_duplicates('CLAVE_PREDIO', inplace=True)
        m2, m1 = points_polis(df)
        
        m1 = gpd.GeoDataFrame(m1).set_geometry(0)
        m1.crs= 'epsg:3857'
        m2 = gpd.GeoDataFrame(m2).set_geometry(0)
        m2.crs= 'epsg:3857'
        m1['LONGITUD_1']=m1[0].x
        m1['LATITUD_1']=m1[0].y
        m2['LONGITUD_1']=m2[0].x
        m2['LATITUD_1']=m2[0].y
        m1.sort_values(['LONGITUD_1', 'LATITUD_1'], inplace=True)
        m2.sort_values(['LONGITUD_1','LATITUD_1'], inplace=True)
        try:
            m2 = m2.drop(columns=['geometry'])
        except:
            pass

        m1.rename(columns={0:'geometry'}, inplace=True)
        m2.rename(columns={0:'geometry'}, inplace=True)
        try:
            m1['dist']=m1.geometry.distance(m2.geometry, align=False)
            m1.drop(columns=['index_left'],  inplace=True)
            m2['dist']=m2.geometry.distance(m1.geometry, align=False)
            m2.drop(columns=['index_left'],  inplace=True)
            m1['ORDEN']= m1.index+1
            m2['ORDEN']= m2.index+1
            # gpd.sjoin_nearest(m1,m2 , max_distance=max_dist, how='right').plot(ax=ax)c.plot()
            m1.reset_index(drop=True, inplace=True)
            m2.reset_index(drop=True, inplace=True)
            m2['dist']=m2.geometry.distance(m1.geometry, align=False)
        except:
            pass

        m1['ORDEN']= m1.index+1
        m2['ORDEN']= m2.index+1
        m1.reset_index(drop=True, inplace=True)
        m2.reset_index(drop=True, inplace=True)
        
        tot= gpd.GeoDataFrame(pd.concat([m1, m2], axis=0), crs= 'epsg:3857',geometry='geometry')
   
        '''Distancias mínimas con respecto a la misma base para eliminar aquellos cercanos
        Si el polígono es muy pequeño,se deja un rango persivio 
        de puntos cercanos menor a 10 y mayor a 5.2 con el fin de encontrar ambos lugares en gcp'''
        tot.reset_index(drop=True, inplace=True)
        tot['ORDEN'] = tot.index +1
        try:
            nm = tot.geometry.apply(lambda g: tot.distance(g))
            for column in nm.columns:
                nm.loc[nm[column]==0,column]= float('NaN')
            zz= nm.min(axis=0)
            for orden in zz.index +1: 
                tot.loc[tot['ORDEN']== orden, 'min_dist']= zz[orden-1]
            t1= tot.loc[tot['min_dist']>=10]
            t2= tot.loc[(tot['min_dist']<10) & (tot['min_dist']>5.2) & (tot['min_dist'].duplicated())].sort_values(['ORDEN'])
            n= t2.shape[0]
            t2=t2.head(int(n/2))
            total=pd.concat([t1,t2], axis=0)
            df_final = pd.concat([total,df_final], axis=0)
        except:
            df_final = pd.concat([tot,df_final], axis=0)
            pass
        
        # break  ##Quitar esta linea, solo funcional para test
            
            
    return df_final 

def task_interpolation(n4_n):
    '''Un ciclo para generar un conjunto de puntos relacionados a la base y al id interpolados sobre el Linestring'''
    bd_f = pd.DataFrame(columns=n4_n.columns)
    bd_f[0] = float('NaN')

    for i, ids in tqdm(enumerate(n4_n['CLAVE_PREDIO'].dropna().unique()),total = len(n4_n)):
        n = n4_n.loc[n4_n['CLAVE_PREDIO'] == ids].reset_index(drop=True)
        
        n=n.to_crs(3857)

        # n=pd.DataFrame(n[:1])
        try:
            n["geometry"] = [shapely.ops.transform(_to_2d, x) for x in n["geometry"]]
        except:
            pass
        n = gpd.GeoDataFrame(n, geometry='geometry', crs='EPSG:4326')#6362
        div = len(n4_n.CVEGEO) + 1
        distance_delta = (int(n.geometry.length.unique()) /div)
        # generate the equidistant points
        distances = np.arange(0, n.geometry.length.unique(), float(distance_delta))
        
        # points = MultiPoint([n.geometry[n.geometry.index[0]].boundary.interpolate(
        #     distance) for distance in distances] + [n.geometry[n.geometry.index[0]].boundary])
        points=MultiPoint([n.geometry[n.geometry.index[0]].boundary.interpolate(distance) for distance in distances]+ [n.geometry[n.geometry.index[0]].centroid])
        list_points = gpd.GeoSeries(points).explode(index_parts=True)
        list_points.crs = 'EPSG:3857'#4326'#6362
        list_points = list_points.to_crs("EPSG:4326")
        bd = pd.concat([n4_n.loc[n4_n['CLAVE_PREDIO'] == ids].reset_index(
            drop=True), list_points.reset_index(drop=True)], axis=1)
        bd_f = pd.concat([bd_f, bd], axis=0,ignore_index=True)
    # '''Un ciclo para generar un conjunto de puntos relacionados a la base y al id interpolados sobre el Linestring'''
    # bd_f = pd.DataFrame(columns=n4_n.columns)
    # bd_f[0] = float('NaN')

    # for i, ids in tqdm(enumerate(n4_n['CLAVE_PREDIO'].unique()),total = len(n4_n)):
    #     n = n4_n.loc[n4_n['CLAVE_PREDIO'] == ids].reset_index(drop=True)
    #     n=n.to_crs(3857)
    #     try:
    #         n["geometry"] = [shapely.ops.transform(_to_2d, x) for x in n["geometry"]]
    #     except:
    #         pass
    #     n = gpd.GeoDataFrame(n, geometry='geometry', crs='EPSG:4326')#6362
    #     div = len(n4_n.CVEGEO) + 1
    #     # print(div)
    #     # print("---"*5)
    #     # print(n.geometry.length.unique())
    #     # print("____"*5)
    #     distance_delta = (int(n.geometry.length.unique()) /
    #                     div)
    #     # generate the equidistant points
    #     # print(distance_delta)
    #     # print("---"*5)
    #     # print(n.geometry)
    #     # print("____"*5)
    #     distances = np.arange(0, n.geometry.length.unique(), float(distance_delta))
    #     # print(distances)
    #     # print("____"*5)
    #     # print(n.geometry[n.geometry.index[0]])#[n.geometry[n.geometry.index[0]].interpolate(distance) for distance in distances] + [n.geometry[n.geometry.index[0]].boundary[1]])
        

    #     points = MultiPoint([n.geometry[n.geometry.index[0]].interpolate(
    #         distance) for distance in distances] + [n.geometry[n.geometry.index[0]].boundary[1]])

    #     list_points = gpd.GeoSeries(points).explode(index_parts=True)
    #     list_points.crs = 'EPSG:3857'#4326'#6362
    #     list_points = list_points.to_crs("EPSG:4326")
    #     bd = pd.concat([n4_n.loc[n4_n['CLAVE_PREDIO'] == ids].reset_index(
    #         drop=True), list_points.reset_index(drop=True)], axis=1)
    #     bd_f = pd.concat([bd_f, bd], axis=0)

    #     # break
    
    return(bd_f)

def task_chunks(chunks):    
    df_final = pd.DataFrame(columns=['index_left', 0, 'CLAVE_PREDIO'])
    # print(chunks)
    # print('chunks:  ',chunks.columns[chunks.columns.duplicated()])
    for i ,cve in tqdm(enumerate(chunks['CLAVE_PREDIO']),total = len(chunks)):
        df = chunks.loc[chunks['CLAVE_PREDIO'].str.contains(cve)]
        
        assert any(df['ESTIMADO']>=1)
        df = combinar_manzanas(df, llave='CLAVE_PREDIO')
        df.reset_index(inplace=True)
        df.drop_duplicates('CLAVE_PREDIO', inplace=True)
        m2, m1 = points_polis(df)
        
        m1 = gpd.GeoDataFrame(m1).set_geometry(0)
        m1.crs= 'epsg:3857'
        m2 = gpd.GeoDataFrame(m2).set_geometry(0)
        m2.crs= 'epsg:3857'
        m1['LONGITUD_1']=m1[0].x
        m1['LATITUD_1']=m1[0].y
        m2['LONGITUD_1']=m2[0].x
        m2['LATITUD_1']=m2[0].y
        m1.sort_values(['LONGITUD_1', 'LATITUD_1'], inplace=True)
        m2.sort_values(['LONGITUD_1','LATITUD_1'], inplace=True)
        try:
            m2 = m2.drop(columns=['geometry'])
        except:
            pass
        m1.rename(columns={0:'geometry'}, inplace=True)
        m2.rename(columns={0:'geometry'}, inplace=True)
        try:
            m1['dist']=m1.geometry.distance(m2.geometry, align=False)
            m1.drop(columns=['index_left'],  inplace=True)
            m2['dist']=m2.geometry.distance(m1.geometry, align=False)
            m2.drop(columns=['index_left'],  inplace=True)
            m1['ORDEN']= m1.index+1
            m2['ORDEN']= m2.index+1
            # gpd.sjoin_nearest(m1,m2 , max_distance=max_dist, how='right').plot(ax=ax)c.plot()
            m1.reset_index(drop=True, inplace=True)
            m2.reset_index(drop=True, inplace=True)
            m2['dist']=m2.geometry.distance(m1.geometry, align=False)
        except:
            pass

        m1['ORDEN']= m1.index+1
        m2['ORDEN']= m2.index+1
        m1.reset_index(drop=True, inplace=True)
        m2.reset_index(drop=True, inplace=True)
        
        tot= gpd.GeoDataFrame(pd.concat([m1, m2], axis=0), crs= 'epsg:3857',geometry='geometry')
   
        tot.reset_index(drop=True, inplace=True)
        tot['ORDEN'] = tot.index +1
        try:
            nm = tot.geometry.apply(lambda g: tot.distance(g))
            for column in nm.columns:
                nm.loc[nm[column]==0,column]= float('NaN')
            zz= nm.min(axis=0)
            for orden in zz.index +1: 
                tot.loc[tot['ORDEN']== orden, 'min_dist']= zz[orden-1]
            t1= tot.loc[tot['min_dist']>=10]
            t2= tot.loc[(tot['min_dist']<10) & (tot['min_dist']>5.2) & (tot['min_dist'].duplicated())].sort_values(['ORDEN'])
            n= t2.shape[0]
            t2=t2.head(int(n/2))
            total=pd.concat([t1,t2], axis=0)
            # print('dentro del try:  ',total.columns[total.columns.duplicated()])
            df_final = pd.concat([total,df_final], axis=0)
        except:
            # print('fuera del try:  ',tot.columns[tot.columns.duplicated()])
            df_final = pd.concat([tot,df_final], axis=0)
            pass
        
        # break  ##Quitar esta linea, solo funcional para test
          
    return df_final
def post_points_catastro(path_base, path_shp, funcion):
    """
        De la informacion que se obtiene, separa la lat y la lon, lo limpia y concatena, quita distancia menor entre ambos parametros.
        Se genera una base total, se hace un proceso de diatancia minima para quitar los puntos que cruzan entre si.

        Los chunks se obtienen a partir de los cores de la pc en que se ejecute este script
    """
    if path_base == float('Nan'):
        test_igecem = prep.data_prep_catastro(path_base, path_shp)
        # print('test igecem es: ',test_igecem)
    else:
        m_igecem = gpd.read_file(path_shp) ##Lee desde shp
        m_igecem.crs= 4326 #m_igecem.to_crs(4326)
    try:
        m_igecem = m_igecem.loc[~m_igecem['manz'].astype(str).str.endswith('000')]
    except: 
        pass
    test_igecem = m_igecem
    prep.replace_columns(test_igecem)
    test_igecem.drop(columns= test_igecem.columns[(test_igecem.columns.str.contains('index'))|(test_igecem.columns.duplicated())], inplace=True)
    # print('test_igecem:  ',test_igecem.columns[test_igecem.columns.duplicated()])
    test_igecem_chunks =  np.array_split(test_igecem, os.cpu_count()-1) ##Aqui se especifica si se requiere un loc y los chunks
    
    df_concat = pd.DataFrame()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for df_chunk in tqdm(zip(executor.map(funcion, test_igecem_chunks)),total = len(test_igecem_chunks)):
            df_concat = pd.concat([df_chunk[0],df_concat], axis=0)

    return df_concat


if __name__ == "__main__":
    PATH_BASE = float('Nan')
    PATH_SHP  = r"C:\Users\dlara\Downloads\final\final\test_igecem_final.shp"

    df_final_catastro = post_points_catastro(PATH_BASE, PATH_SHP, task_chunks)

    print(df_final_catastro)    

    df_final_catastro.to_csv(r'C:\Users\dlara\Atlacomulco_puntos.csv', encoding='utf-8-sig')