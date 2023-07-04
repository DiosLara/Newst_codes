import pandas as pd
import geopandas as gpd
import numpy as np
import os
from scipy.spatial import cKDTree
from shapely.geometry import Point
from preparacion_datos.src.preparacion_inter_puntos import prep
import glob
import tqdm as tqdm
from shapely.geometry import MultiPolygon, Polygon
import shapely.wkt
import shapely.geometry
import warnings
from shapely.geometry import shape
warnings.filterwarnings("ignore")
from puntos_sobre_poligono.simulacion_interpolacion import task_chunks


columnas_dict= {'columnas1': []}


def base_final(manz,curts, pred):
    corregir_predial(pred)
    pred_diff,m4=integrar_shapes_predial(manz,curts, pred)
    
    pred_final=pd.concat([pred_diff,m4])
    pred_final=pred_final.sort_values('col1')
    pred_final['LATITUD'].interpolate(method='nearest',limit_direction='backward', limit=2,inplace=True)
    pred_final['LONGITUD'].interpolate(method='nearest',limit_direction='backward', limit=2,inplace=True)
    pred_final=prep.transform_df_to_gpd(pred_final, 'LONGITUD','LATITUD')
    pred_final=pred_final[pred_final.columns[~pred_final.columns.str.contains('cve|id|mun|zona|manz')]]
    pred_final.columns=pred_final.columns.str.upper()
    df_concat=puntos_predial(pred_diff)
    return(pred_final,df_concat)

def puntos_predial(pred_diff,pred_final):
    
    igecem=pred_diff.copy()
    test_igecem=igecem[['col2', 'col3', 'col4','geometry']].groupby(['geometry','col3'], sort=False).first().reset_index()

    test_igecem=test_igecem.merge(pred_final.groupby('col2').count().reset_index().sort_values(by='col1', ascending=False)[['col1', 'CURT', 'col2']].rename(columns={'col1':'ESTIMADO'}), on ='col2')
    test_igecem=test_igecem.sort_values(by='ESTIMADO', ascending=False)
    df_concat=task_chunks(test_igecem)
    return(df_concat)

def prep_cruces(manz,curts, pred):
    pred_final,df_concat= base_final(manz,curts, pred)
    conteos=pred_final.groupby('col2').count().reset_index().sort_values(by='col1', ascending=False)[['col1', 'CURT', 'col2']].rename(columns={'col1':'ESTIMADO'})
    unicos=conteos.loc[conteos['ESTIMADO']==1]
    pred_f_unicos=pred_final.loc[pred_final['col2'].isin(unicos['col2'])]
    assert pred_final.loc[~pred_final['col1'].str.endswith('000000')].isempty
    pred_final.rename(columns={'GEOMETRY':'geometry'}, inplace=True)
    pred_cruce=pred_final.loc[~pred_final['col2'].isin(unicos['col2'])]
    pred_cruce.rename(columns={'GEOMETRY':'geometry'}, inplace=True)
    pred_cruce=gpd.GeoDataFrame(pred_cruce, geometry='geometry',crs=4326)
    df_concat=df_concat[df_concat.columns[(~df_concat.columns.isin(pred_cruce.columns)) | (df_concat.columns.astype(str).str.contains('col2|geometry'))]]
    pred_final=pred_final.to_crs(3857)
    pred_cruzado, nnm1=multillave_cruce(pred_cruce,df_concat)
    return(pred_f_unicos, pred_cruzado, nnm1)

def multillave_cruce(pred_cruce,df_concat):
    mn1=pd.merge(pred_cruce.drop(columns=['geometry','col2']),df_concat, left_on='col1',right_on='col2')
    tempn1= df_concat.loc[~df_concat['col2'].isin(mn1['col1'])]
    mn2=pd.merge(pred_cruce.drop(columns=['geometry']),tempn1, left_on='col2',right_on='col2')
    mn1=mn1.sort_values(by=['col1','LONGITUD_1','LATITUD_1']).drop_duplicates(subset='col1',keep='first')
    mn2=mn2.sort_values(by=['col1','LONGITUD_1','LATITUD_1']).drop_duplicates(subset='col1',keep='first')
    pred_cruzado=pd.concat([mn1,mn2])
    nnm1=prep.ckdnearest(pred_cruce.loc[~pred_cruce['geometry'].is_empty].to_crs(3857),df_concat[['geometry','LATITUD_1','LONGITUD_1']])
    nnm1=nnm1.sort_values(by=['col1','min_dist_2']).drop_duplicates(subset=['col1'],keep='first')
    
    pred_cruzado['CURT']=pred_cruzado['CURT'].fillna(float('Nan'))
    pred_cruzado['curt']=pred_cruzado['curt'].fillna(float('Nan'))
    if all(pred_cruzado['CURT'])==all(pred_cruzado['curt']):
        pred_cruzado.drop(columns=['CURT'], inplace=True)

    nnm1=nnm1.loc[~nnm1['col1'].isin(pred_cruzado['col1'])]

    return(pred_cruzado, nnm1)

def concat_multi(pred,manz,curts):
    
    nnm1,pred_cruzado,pred_f_unicos= prep_cruces(manz,curts, pred)
    
    pred_cruzado_min=pd.concat([nnm1,pred_cruzado,pred_f_unicos])    
    pred_cruzado_min.drop(columns=['index_left','dist',0], inplace=True)
    pred_cruzado_min=gpd.GeoDataFrame(pred_cruzado_min, geometry='geometry')
    return(pred_cruzado_min)

def copyFile(titulo=''):  
   # using the filedialog's askopenfilename() method to select the file  
   fileToCopy = fd.askopenfilename(  
      title = titulo,  
      filetypes=[("All files", "*.*")]  
      )  
   return(fileToCopy)

if __name__ == "__main__":

    for_pred= 'Selecciona el path donde se encuentra predial'
    for_curts= 'Selecciona el path donde se encuentran las curts \n *shape only*'
    for_manz= 'Selecciona el path donde se encuentran las manzanas \n *shape only*'
    path_pred=copyFile(titulo=for_pred)
    pred= pd.read_csv(path_pred, encoding='utf-8-sig')
    path_curts=copyFile(titulo=for_curts)
    if path_curts:
        curts=gpd.read_file(path_curts)
        if  curts.crs!= 3857:
            curts=curts.to_crs(3857)

    else:
        curts=float('Nan')
        curts.crs= 3857
    path_manz=copyFile(titulo=for_manz)
    manz=gpd.read_file(path_manz)
    manz=manz.to_crs(curts.crs)
    pred_cruzado_min=concat_multi(pred,manz,curts)