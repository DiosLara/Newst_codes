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



   
def integrar_shapes_predial(manz,curts, pred):
    if curts:
        m1=pd.merge(pred, curts,  left_on='ID', right_on='id_cat')
        temp1= pred.loc[~pred['ID'].isin(m1['ID'])]
        m2=pd.merge(temp1, curts,  left_on='ID2', right_on='id_cat')
        temp2= pred.loc[(~pred['ID'].isin(m1['ID']))& (~pred['ID'].isin(m2['ID']))]
    else:
        m1= pd.DataFrame()
        m2= pd.DataFrame()
        temp2=pred
        m3=pd.merge(temp2,manz,  left_on='CLAVE_MZA', right_on='cve_cat')
        temp4= pred.loc[(~pred['ID'].isin(m1['ID']))& (~pred['ID'].isin(m2['ID']))& (~pred['ID'].isin(m3['ID']))]
        m4=pd.merge(temp4,manz,  left_on='CLAVE_LOC', right_on='cve_cat')

    pred_diff=pd.concat([m1,m2,m3])
    pred_diff=pred_diff.sort_values('ID')[['ID', 'ID2', '...']]
    pred_diff=gpd.GeoDataFrame(pred_diff, geometry='geometry', crs=4326)
    pred_diff['geometry']=pred_diff.geometry.centroid
    pred_diff['LATITUD']= pred_diff.geometry.y
    pred_diff['LONGITUD']= pred_diff.geometry.x
    return(pred_diff,m4)

def base_final(manz,curts, pred):
    corregir_predial(pred)
    pred_diff,m4=integrar_shapes_predial(manz,curts, pred)
    
    pred_final=pd.concat([pred_diff,m4])
    pred_final=pred_final.sort_values('ID')
    pred_final['LATITUD'].interpolate(method='nearest',limit_direction='backward', limit=2,inplace=True)
    pred_final['LONGITUD'].interpolate(method='nearest',limit_direction='backward', limit=2,inplace=True)
    pred_final=prep.transform_df_to_gpd(pred_final, 'LONGITUD','LATITUD')
    pred_final=pred_final[pred_final.columns[~pred_final.columns.str.contains('cve|id|mun|zona|manz')]]
    pred_final.columns=pred_final.columns.str.upper()
    df_concat=puntos_predial(pred_diff)
    return(pred_final,df_concat)

def puntos_predial(pred_diff,pred_final):
    
    igecem=pred_diff.copy()
    test_igecem=igecem[['ID2', '...','geometry']].groupby(['geometry',''], sort=False).first().reset_index()

    test_igecem=test_igecem.merge(pred_final.groupby('ID2').count().reset_index().sort_values(by='ID', ascending=False)[['ID', 'CURT', 'ID2']].rename(columns={'ID':'ESTIMADO'}), on ='ID2')
    test_igecem=test_igecem.sort_values(by='ESTIMADO', ascending=False)
    df_concat=task_chunks(test_igecem)
    return(df_concat)

def prep_cruces(manz,curts, pred):
    pred_final,df_concat= base_final(manz,curts, pred)
    conteos=pred_final.groupby('ID2').count().reset_index().sort_values(by='ID', ascending=False)[['ID', 'CURT', 'ID2']].rename(columns={'ID':'ESTIMADO'})
    unicos=conteos.loc[conteos['ESTIMADO']==1]
    pred_f_unicos=pred_final.loc[pred_final['ID2'].isin(unicos['ID2'])]
    assert pred_final.loc[~pred_final['ID'].str.endswith('000000')].isempty
    pred_final.rename(columns={'GEOMETRY':'geometry'}, inplace=True)
    pred_cruce=pred_final.loc[~pred_final['ID2'].isin(unicos['ID2'])]
    pred_cruce.rename(columns={'GEOMETRY':'geometry'}, inplace=True)
    pred_cruce=gpd.GeoDataFrame(pred_cruce, geometry='geometry',crs=4326)
    df_concat=df_concat[df_concat.columns[(~df_concat.columns.isin(pred_cruce.columns)) | (df_concat.columns.astype(str).str.contains('ID2|geometry'))]]
    pred_final=pred_final.to_crs(3857)
    pred_cruzado, nnm1=multillave_cruce(pred_cruce,df_concat)
    return(pred_f_unicos, pred_cruzado, nnm1)

def multillave_cruce(pred_cruce,df_concat):
    mn1=pd.merge(pred_cruce.drop(columns=['geometry','ID2']),df_concat, left_on='ID',right_on='ID2')
    tempn1= df_concat.loc[~df_concat['ID2'].isin(mn1['ID'])]
    mn2=pd.merge(pred_cruce.drop(columns=['geometry']),tempn1, left_on='ID2',right_on='ID2')
    mn1=mn1.sort_values(by=['ID','LONGITUD_1','LATITUD_1']).drop_duplicates(subset='ID',keep='first')
    mn2=mn2.sort_values(by=['ID','LONGITUD_1','LATITUD_1']).drop_duplicates(subset='ID',keep='first')
    pred_cruzado=pd.concat([mn1,mn2])
    nnm1=prep.ckdnearest(pred_cruce.loc[~pred_cruce['geometry'].is_empty].to_crs(3857),df_concat[['geometry','LATITUD_1','LONGITUD_1']])
    nnm1=nnm1.sort_values(by=['ID','min_dist_2']).drop_duplicates(subset=['ID'],keep='first')
    
    pred_cruzado['CURT']=pred_cruzado['CURT'].fillna(float('Nan'))
    pred_cruzado['curt']=pred_cruzado['curt'].fillna(float('Nan'))
    if all(pred_cruzado['CURT'])==all(pred_cruzado['curt']):
        pred_cruzado.drop(columns=['CURT'], inplace=True)

    nnm1=nnm1.loc[~nnm1['ID'].isin(pred_cruzado['ID'])]

    return(pred_cruzado, nnm1)

def concat_multi(pred,manz,curts):
    
    nnm1,pred_cruzado,pred_f_unicos= prep_cruces(manz,curts, pred)
    
    pred_cruzado_min=pd.concat([nnm1,pred_cruzado,pred_f_unicos])    
    pred_cruzado_min.drop(columns=['index_left','dist',0,'NOTAS','PROCESO'], inplace=True)
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