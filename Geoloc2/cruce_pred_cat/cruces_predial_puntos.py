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




def corregir_predial(pred):
    pred.dropna(subset=['col1'],inplace=True)
    pred.loc[~(pred['col1'].astype(str).str.startswith('0')),'col1']=pred.loc[~(pred['col1'].astype(str).str.startswith('0'))]['col1'].astype(str).str[0:15].str.zfill(16)
    pred.loc[(pred['col1'].str.len()<16)| (pred['col1'].str.startswith('00')),'col1']=pred.loc[pred['col1'].str.len()<16]['col1'].str.ljust(16,fillchar='0')
    pred=pred.sort_values('col', ascending=False).drop_duplicates(['col1'])
    pred['col1']=pred['col1'].astype(str).str.zfill(16)
    pred['col2']=pred['col1'].str[0:10]+'000000'
    pred['col3']=pred['col1'].str[0:8]+'00000000'
    pred['col4']=pred['col1'].str[0:6]+'0000000000'
    pred['col4']=pred['col4'].str.zfill(16)
    pred['col3']=pred['col3'].str.zfill(16)
    
def integrar_shapes_predial(manz,curts, pred):
    if curts:
        m1=pd.merge(pred, curts,  left_on='col1', right_on='id_cat')
        temp1= pred.loc[~pred['col1'].isin(m1['col1'])]
        m2=pd.merge(temp1, curts,  left_on='col2', right_on='id_cat')
        temp2= pred.loc[(~pred['col1'].isin(m1['col1']))& (~pred['col1'].isin(m2['col1']))]
    else:
        m1= pd.DataFrame()
        m2= pd.DataFrame()
        temp2=pred
        m3=pd.merge(temp2,manz,  left_on='col3', right_on='opt1')
        temp4= pred.loc[(~pred['col1'].isin(m1['col1']))& (~pred['col1'].isin(m2['col1']))& (~pred['col1'].isin(m3['col1']))]
        m4=pd.merge(temp4,manz,  left_on='col4', right_on='opt1')

    pred_diff=pd.concat([m1,m2,m3])
    pred_diff=pred_diff.sort_values('col1')[[]]
    pred_diff=gpd.GeoDataFrame(pred_diff, geometry='geometry', crs=4326)
    pred_diff['geometry']=pred_diff.geometry.centroid
    pred_diff['LATITUD']= pred_diff.geometry.y
    pred_diff['LONGITUD']= pred_diff.geometry.x
    return(pred_diff,m4)

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
    test_igecem=igecem[['CLAVE_PREDIO', 'CLAVE_MZA', 'CLAVE_LOC',  'curt','geometry']].groupby(['geometry','CLAVE_MZA'], sort=False).first().reset_index()

    test_igecem=test_igecem.merge(pred_final.groupby('CLAVE_PREDIO').count().reset_index().sort_values(by='col1', ascending=False)[['col1', 'CURT', 'CLAVE_PREDIO']].rename(columns={'col1':'ESTIMADO'}), on ='CLAVE_PREDIO')
    test_igecem=test_igecem.sort_values(by='ESTIMADO', ascending=False)
    df_concat=task_chunks(test_igecem)
    return(df_concat)

def prep_cruces(manz,curts, pred):
    pred_final,df_concat= base_final(manz,curts, pred)
    conteos=pred_final.groupby('CLAVE_PREDIO').count().reset_index().sort_values(by='col1', ascending=False)[['col1', 'CURT', 'CLAVE_PREDIO']].rename(columns={'col1':'ESTIMADO'})
    unicos=conteos.loc[conteos['ESTIMADO']==1]
    pred_f_unicos=pred_final.loc[pred_final['CLAVE_PREDIO'].isin(unicos['CLAVE_PREDIO'])]
    assert pred_final.loc[~pred_final['col1'].str.endswith('000000')].isempty
    pred_final.rename(columns={'GEOMETRY':'geometry'}, inplace=True)
    pred_cruce=pred_final.loc[~pred_final['CLAVE_PREDIO'].isin(unicos['CLAVE_PREDIO'])]
    pred_cruce.rename(columns={'GEOMETRY':'geometry'}, inplace=True)
    pred_cruce=gpd.GeoDataFrame(pred_cruce, geometry='geometry',crs=4326)
    df_concat=df_concat[df_concat.columns[(~df_concat.columns.isin(pred_cruce.columns)) | (df_concat.columns.astype(str).str.contains('CLAVE_PREDIO|geometry'))]]
    pred_final=pred_final.to_crs(3857)
    pred_cruzado, nnm1=multillave_cruce(pred_cruce,df_concat)
    return(pred_f_unicos, pred_cruzado, nnm1)

def multillave_cruce(pred_cruce,df_concat):
    mn1=pd.merge(pred_cruce.drop(columns=['geometry','CLAVE_PREDIO']),df_concat, left_on='col1',right_on='CLAVE_PREDIO')
    tempn1= df_concat.loc[~df_concat['CLAVE_PREDIO'].isin(mn1['col1'])]
    mn2=pd.merge(pred_cruce.drop(columns=['geometry']),tempn1, left_on='CLAVE_PREDIO',right_on='CLAVE_PREDIO')
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