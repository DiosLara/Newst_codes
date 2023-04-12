import pandas as pd
import warnings
# from padagua_2021.Geolocalizacion.separacion_domicilios.funciones_diccionarios import diccionarios_gcp  
from PipelinesMunicipios.pipeline import Pipeline
from PipelinesMunicipios.process import Process

def read_files():
    """
        Agregamos al dict los df tomando como referencia su path y el nombre del df
    """
    for key in dict_dfs:
        if dict_dfs[key]['path'].endswith('csv') or dict_dfs[key]['path'].endswith('txt'):
            dict_dfs[key]['df'] = pd.read_csv(dict_dfs[key]['path'], encoding='utf-8-sig', low_memory=False)
        elif dict_dfs[key]['path'].endswith('xlsx'):
            dict_dfs[key]['df'] = pd.read_excel(dict_dfs[key]['path'], encoding='utf-8-sig', low_memory=False)

    return dict_dfs

def clean_info_ads(dict):
    ads = dict_dfs.get('ads')['df']; GCP0 = dict_dfs.get('GCP0')['df']; pred = dict_dfs.get('pred')['df']

    GCP0.loc[~GCP0['CLAVECATASTRAL'].isin(pred['id_cat'])]

    GCP0=GCP0[['CLAVECATASTRAL', 'Número de Unidad', 'BDINTERNA_CONT_DIRECCION',
       'CLAVEASENTAMIENTO', 'CLAVECP', 'CLAVEMANZANA', 'CLAVEZONA',
       'CLAVESTATUS', 'DIRECCION_CATS', 'VALORTERRENO', 'VALORTERRENOCOMUN',
       'VALORCONSTPROP', 'VALORCONSTCOMUN', 'SUPERFICIE',
       'SUPERFICIETERRCOMUN', 'SUPERFICIECONST', 'SUPERFICIECONSTCOMUN',
       'CLAVEANTERIOR', 'FECHAALTA', 'ASENTAMIENTO_NR', 'CLAVECP_NR',
       'CLAVE_PRED', 'curt', 'Latitude_clean', 'Longitude_clean',
       'COORDENADAS_RECAUDACION', 'CLAVEMANZANA_R', 'CLAVEZONA_R',
       'DIRECCION',
       'place_id', 'types', 'geometry.location.lat', 'geometry.location.lng',
       'geometry.location_type', 'plus_code.compound_code']]

    GCP0.rename(columns={'place_id':'INTFIS_RS_FD_PLACE_ID', 'geometry.location.lat': 'INTFIS_RS_FD_LAT',
        'geometry.location.lng': 'INTFIS_RS_FD_LON', 'plus_code.compound_code': 'INTFIS_RS_FD_PLUS_CODE', 'DIRECCION':'INTFIS_RS_FD_DIRECCION'}, inplace=True)

    ads[['CLAVECATASTRAL', 'INTFIS_RS_FD_GEO_CLAVECATASTRAL_PREDIO',
       'CLAVECATASTRAL_RECAUDACION']]
    ads=ads.loc[ads['CLAVECATASTRAL']=='0000000000000nan'].drop(columns=['CLAVECATASTRAL', 'INTFIS_RS_FD_GEO_CLAVECATASTRAL_PREDIO',
        'CLAVECATASTRAL_RECAUDACION'])
    ads=ads.loc[~ads['INTFIS_RS_FD_PLACE_ID'].isin(GCP0['INTFIS_RS_FD_PLACE_ID'])]

    dict_dfs['ads'].update({'df':ads, 'path':'/'})
    dict_dfs['GCP0'].update({'df':GCP0, 'path':'/'})

    return dict_dfs
    
def classification_cve_catastral(dict_dfs):
    CAT_N = dict_dfs.get('CAT_N')['df']; E_PC  = dict_dfs.get('E_PC')['df']; pred = dict_dfs.get('pred')['df']; DBF  = dict_dfs.get('DBF')['df']
    
    PRED_CAT_N = pd.merge(CAT_N, E_PC, left_on='Clave Catastral', right_on='CLAVECATASTRAL', how='outer')
    PRED_CAT_N['CLAVECATASTRAL'] = PRED_CAT_N.CLAVECATASTRAL.combine_first(PRED_CAT_N['Clave Catastral'])
    dict_dfs['PRED_CAT_N'] = {'df' : PRED_CAT_N, 'path' : '/'}

    PRED_CAT_CURTS = pd.merge(pred, PRED_CAT_N, left_on='id_cat', right_on='CLAVECATASTRAL', how='right')
    PRED_CAT_CURTS['CLAVECATASTRAL'] = PRED_CAT_CURTS.CLAVECATASTRAL.combine_first(PRED_CAT_CURTS['id_cat'])
    dict_dfs['PRED_CAT_CURTS'] = {'df' : PRED_CAT_CURTS, 'path' : '/'}

    PRED_CAT_CURT_DBF = pd.merge(PRED_CAT_CURTS, DBF[['USO','CLAVECATASTRAL']].drop_duplicates('CLAVECATASTRAL'), on='CLAVECATASTRAL', how='left')
    PRED_CAT_CURT_DBF = PRED_CAT_CURT_DBF.drop_duplicates('CLAVECATASTRAL')
    dict_dfs['PRED_CAT_CURT_DBF'] = {'df' : PRED_CAT_CURT_DBF, 'path' : '/'}
    PCCDF=PRED_CAT_CURT_DBF.loc[(PRED_CAT_CURT_DBF['curt'].notna())| (PRED_CAT_CURT_DBF['DIRECCION'].notna())| (PRED_CAT_CURT_DBF['Domicilio de inmueble'].notna())]
    dict_dfs['PCCDF'] = {'df' : PCCDF, 'path' : '/'}

    dict_dfs = clean_info_ads(dict_dfs)

    return dict_dfs

def new_address_catastro(list_fields,DIRS):
    for count, value in enumerate(list_fields, start=1):
        DIRS[value]=DIRS['Domicilio del inmueble construido'].str.split('COLONIA|CALLE|NUMERO EXTERIOR|EDIFICIO|DEPARTAMENTO|NUMERO INTERIOR').str[count]

    return DIRS

def generate_addres_catastro(DIR_AUX):
    DIR_AUX['MANZANA_LOTE_catastro']= DIR_AUX['Domicilio de inmueble'].str.split('MZ|MZA', expand=True)[1]
    DIR_AUX['MANZANA_LOTE_catastro']= 'MZ'+DIR_AUX['MANZANA_LOTE_catastro']
    DIR_AUX['CALLE_catastro_0']=DIR_AUX.apply(lambda row: row['Domicilio de inmueble'].replace(str(row['MANZANA_LOTE_catastro']), ""),axis=1).str.replace(',','').str.strip(' ').str.replace(r'\s+', ' ', regex=True)

    DIR_AUX['NUMERO_EXTERIOR_catastro']=DIR_AUX['NUMERO_EXTERIOR_catastro'].str.extract('(\d+)')

    DIR_AUX['N_EXT_catastro_0']=DIR_AUX['CALLE_catastro_0'].str.replace(r'\s+', ' ').str.strip().str.split(' NO.|N°|Nº').str[1].str.findall(r"(\d{1}\-?\d{0,4})").str[0]
    DIR_AUX['CALLE_catastro_0']=DIR_AUX['CALLE_catastro_0'].astype(str)
    DIR_AUX['CALLE_catastro_0']=DIR_AUX.apply(lambda row: row['CALLE_catastro_0'].replace(str(row['N_EXT_catastro_0']), ""),axis=1).str.replace(',','').str.strip(' ').str.replace(r'\s+', ' ', regex=True)
    DIR_AUX['CALLE_catastro_0']=DIR_AUX['CALLE_catastro_0'].str.replace('N°','').str.replace('Nº','')

    DIR_AUX['CALLE_catastro_0'].str.replace(r'\s+', ' ').str.strip().str.split(' NO.|N°|Nº').str[1].str.findall(r"(\d{1}\-?\d{0,4})").str[0][308275]
    # DIR_AUX['CALLE_catastro_0'].astype(str)[308275]
    DIR_AUX.apply(lambda row: row['CALLE_catastro_0'].replace(str(row['N_EXT_catastro_0']), ""),axis=1).str.replace(',','').str.strip(' ').str.replace(r'\s+', ' ', regex=True)[308275]

    DIR_AUX.loc[(DIR_AUX['CALLE_catastro_0']==' ') |(DIR_AUX['CALLE_catastro_0']=='') | (DIR_AUX['CALLE_catastro_0'].str.contains('SIN NOMBRE')),'CALLE_catastro_0']=float('NaN')
    DIR_AUX.loc[(DIR_AUX['CALLE_catastro']==' ')|(DIR_AUX['CALLE_catastro']=='') | (DIR_AUX['CALLE_catastro'].str.contains('SIN NOMBRE')),'CALLE_catastro']=float('NaN')

    DIR_AUX['CALLE_catastro_final']=DIR_AUX['CALLE_catastro'].combine_first(DIR_AUX['CALLE_catastro_0']).str.replace('CALLE','').str.replace('N°','', regex=False).str.replace('AV.','', regex=False)
    DIR_AUX['NUMERO_EXTERIOR_catastro_final']=DIR_AUX['NUMERO_EXTERIOR_catastro'].combine_first(DIR_AUX['N_EXT_catastro_0'])

    return DIR_AUX

def clean_address(dict_dfs):
    PCCDF = dict_dfs.get('PCCDF')['df']
    DIRS=PCCDF.loc[(PCCDF['Domicilio de inmueble'].notna()) & (PCCDF['curt'].isna())]
    DIRS['MANZANA_PREDIAL']= DIRS['DIRECCION'].str.split('MZ|NO.', expand=True)[1]
    DIRS['MANZANA_PREDIAL']= 'MZ'+DIRS['MANZANA_PREDIAL']
    DIRS['DIRECCION_0']=DIRS.apply(lambda row: row['DIRECCION'].replace(str(row['MANZANA_PREDIAL']), ""),axis=1).str.replace(',','').str.strip(' ').str.replace(r'\s+', ' ', regex=True)
    DIRS['LOTE_PREDIAL']= DIRS['DIRECCION_0'].str.split('LT', expand=True)[1]
    DIRS['LOTE_PREDIAL']= 'LT'+DIRS['LOTE_PREDIAL']
    DIRS['DIRECCION_0']=DIRS.apply(lambda row: row['DIRECCION_0'].replace(str(row['LOTE_PREDIAL']), ""),axis=1).str.replace(',','').str.strip(' ').str.replace(r'\s+', ' ', regex=True)
    DIRS['N_EXT_PREDIAL']=DIRS['DIRECCION_0'].str.replace(r'\s+', ' ').str.strip().str.split(' NO.|,|N°|Nª').str[1].str.findall(r"(\d{1}\-?\d{0,4})").str[0]
    DIRS=DIRS[['DIRECCION','N_EXT_PREDIAL','Domicilio de inmueble','Domicilio del inmueble construido','Clave Catastral', 'Municipio', 'Zona',
            'Manzana', 'Lote', 'Edificio', 'Departamento', 'Año de construcción',
            'Estado de la Construcción', 'Niveles de Construcción',
            'Cantidad de veces que se repite clave catastral', 'CLAVECATASTRAL',
            'CLAVEASENTAMIENTO', 'CLAVECP', 'CLAVEMUNICIPIO', 'CLAVEENTIDAD',
            'CLAVEMANZANA', 'CLAVEZONA', 'CLAVESTATUS', 'VALORTERRENO',
            'VALORTERRENOCOMUN', 'VALORCONSTPROP', 'VALORCONSTCOMUN', 'SUPERFICIE',
            'SUPERFICIETERRCOMUN', 'SUPERFICIECONST', 'SUPERFICIECONSTCOMUN',
            'CLAVEANTERIOR', 'LATITUD', 'LONGITUD', 'FECHAALTA', 'ASENTAMIENTO_NR',
            'CLAVECP_NR', 'USO']]

    DIRS = new_address_catastro(['COLONIA_catastro','CALLE_catastro','NUMERO_EXTERIOR_catastro','EDIFICIO_catastro','DEPARTAMENTO_catastro','NUMERO_INTERIOR_catastro'], DIRS)

    DIRS = generate_addres_catastro(DIRS)
    
    ## AQUI IBA EL CODIGO DE LA FN

    DIRS.loc[(DIRS['CALLE_catastro_final'].astype(str).str.strip() ==DIRS['CALLE_PRED'].str.strip().astype(str)) & (DIRS['N_EXT_PREDIAL'].astype(str).str.strip() !=DIRS['NUMERO_EXTERIOR_catastro_final'].str.strip().astype(str))&(DIRS['NUMERO_EXTERIOR_catastro_final'].notna())&(DIRS['N_EXT_PREDIAL'].notna())][['DIRECCION','DIRECCION_0','CALLE_PRED','N_EXT_PREDIAL','Domicilio de inmueble','Domicilio del inmueble construido','COLONIA_catastro','CALLE_catastro','NUMERO_EXTERIOR_catastro','NUMERO_EXTERIOR_catastro_final','CALLE_catastro_0','N_EXT_catastro_0','CALLE_catastro_final']]
    DIRS['DOMICILIO_CATASTRO']= DIRS['CALLE_catastro_0'].str.cat(DIRS[['NUMERO_EXTERIOR_catastro_final','MANZANA_LOTE_catastro', 'COLONIA_catastro','EDIFICIO_catastro','DEPARTAMENTO_catastro', 'NUMERO_INTERIOR_catastro']].astype(str), ' ').str.replace('nan','')
    DIRS.loc[DIRS['DOMICILIO_CATASTRO'].str.contains('0 0'),'DOMICILIO_CATASTRO'] =float('NaN')
    DIRS['DIRECCION_FINAL']=DIRS['DOMICILIO_CATASTRO'].combine_first(DIRS['DIRECCION'])
    DIRS.loc[(DIRS['DIRECCION_FINAL']!='nan') &(~DIRS['DIRECCION_FINAL'].str.contains('0 0'))][['CLAVECATASTRAL',  'DIRECCION_FINAL']].to_csv('BUSQUEDA_ECATEPEC.csv', encoding='utf-8-sig')
    dict_dfs['DIRS'] = {'df' : DIRS, 'path' : '/'}

    DIRS_CALLES_0=DIRS.loc[(DIRS['DIRECCION_FINAL']!='nan') &(~DIRS['DIRECCION_FINAL'].str.contains('0 0'))][['CLAVECATASTRAL',  'DIRECCION_FINAL']]
    DIRS_CALLES_0['INTFIS_RS_FD_KEY']=DIRS_CALLES_0.reset_index().index
    dict_dfs['DIRS_CALLES_0'] = {'df' : DIRS_CALLES_0, 'path' : '/'}
    
    return dict_dfs

def street_adjustments(dict_dfs):
    DIRS_CALLES = dict_dfs.get('DIRS_CALLES')['df']; DIRS_CALLES_0 = dict_dfs.get('DIRS_CALLES_0')['df']; CAT_N = dict_dfs.get('CAT_N')['df']; 
    ads = dict_dfs.get('ads')['df']
    DIRS_CALLES.columns=DIRS_CALLES.columns.str.upper()
    DIRS_CALLES.rename(columns={'FORMATTED_ADDRESS': 'DIRECCION', 'GEOMETRY.LOCATION.LNG':'LON', 'GEOMETRY.LOCATION.LAT':'LAT', 'PLUS_CODE.COMPOUND_CODE':'PLUS_CODE', 'GEOMETRY.LOCATION_TYPE':'GEOM_TYPE'}, inplace=True)
    DIRS_CALLES.columns=str('INTFIS_RS_FD_')+DIRS_CALLES.columns
    DIRS_CALLES=DIRS_CALLES.drop_duplicates('INTFIS_RS_FD_KEY', keep='last')
    DIRS_CALLES_N=pd.merge(DIRS_CALLES,DIRS_CALLES_0, on='INTFIS_RS_FD_KEY')
    dict_dfs['DIRS_CALLES_N'] = {'df' : DIRS_CALLES_N, 'path' : '/'}

    PRED_CAT_FINAL = None ##Checar de donde esta saliendo este df

    DIRS_2=CAT_N.loc[~CAT_N['Clave Catastral'].isin(PRED_CAT_FINAL['CLAVECATASTRAL'])]

    DIRS_2 = new_address_catastro(['COLONIA_catastro','CALLE_catastro','NUMERO_EXTERIOR_catastro','EDIFICIO_catastro','DEPARTAMENTO_catastro','NUMERO_INTERIOR_catastro'], DIRS_2)

    DIRS_2 = generate_addres_catastro(DIRS_2)

    DIRS_2['DOMICILIO_CATASTRO']= DIRS_2['CALLE_catastro_final'].fillna('').str.cat(DIRS_2[['NUMERO_EXTERIOR_catastro_final','MANZANA_LOTE_catastro', 'COLONIA_catastro','EDIFICIO_catastro','DEPARTAMENTO_catastro', 'NUMERO_INTERIOR_catastro']].astype(str), ' ').str.replace('nan','').str.replace('SIN','').str.replace('S/N','')

    ads.drop_duplicates('INTFIS_RS_FD_DIRECCION').to_csv('FULL_ADICIONALES.csv', encoding='utf-8-sig') ## Checar si es correcto que esto vaya aqui

    DIRS_2.rename(columns={'Clave Catastral':'CLAVECATASTRAL'}, inplace=True)
    DIRS_2[['CLAVECATASTRAL','DOMICILIO_CATASTRO']].to_csv('BUSQUEDA_CATASTRO2.csv', encoding='utf-8-sig')

    dict_dfs['DIRS_2'] = {'df' : DIRS_2, 'path' : '/'}

    return dict_dfs

def streets_catastro(dict_dfs):
    DIRS_2_Q = dict_dfs.get('DIRS_2_Q')['df']; DIRS_2 = dict_dfs.get('DIRS_2')['df']; DIRS_CALLES_N = dict_dfs.get('DIRS_CALLES_N')['df'];  
    DIRS_2_Q.rename(columns={'place_id':'PLACE_ID', 'geometry.location.lat': 'LAT',
        'geometry.location.lng': 'LON', 'plus_code.compound_code': 'PLUS_CODE', 'formatted_address':'DIRECCION'}, inplace=True)
    DIRS_2_Q.columns=str('INTFIS_RS_FD_')+DIRS_2_Q.columns.str.upper()

    DIRS_2.columns=str('CATASTRO_')+DIRS_2.columns.str.upper()
    DIRS_2['INTFIS_RS_FD_KEY']=DIRS_2.reset_index(drop=True).index

    DIRS_2_Q=DIRS_2_Q.drop_duplicates('INTFIS_RS_FD_KEY')

    DIRS_2.rename(columns={'CATASTRO_CLAVECATASTRAL':'CLAVECATASTRAL','CATASTRO_ESTADO DE LA CONSTRUCCIÓN':'CATASTRO_ESTADO_CONSTRUCCION', 'CATASTRO_AÑO DE CONSTRUCCIÓN':  'CATASTRO_ANIO_CONSTRUCCION', 'CATASTRO_ESTADO DE LA CONSTRUCCIÓN':'CATASTRO_ESTADO_CONSTRUCCION',
       'CATASTRO_NIVELES DE CONSTRUCCIÓN':'CATASTRO_NIVELES_CONSTRUCCION', 'CATASTRO_DOMICILIO DE INMUEBLE':'CATASTRO_DOMICILIO_INMUEBLE',
       'CATASTRO_DOMICILIO DEL INMUEBLE CONSTRUIDO':'CATASTRO_DOMICILIO_INMUEBLE_CONSTRUIDO',
       'CATASTRO_CANTIDAD DE VECES QUE SE REPITE CLAVE CATASTRAL':'CATASTRO_CANTIDAD_REPITE_CVECATASTRAL'}, inplace=True)
    DIRS_2_Q.rename(columns={'INTFIS_RS_FD_GEOMETRY.LOCATION_TYPE':'INTFIS_RS_FD_GEOM_TYPE'}, inplace=True)

    dict_dfs['DIRS_2_Q'].update({'df':DIRS_2_Q, 'path':'/'})
    dict_dfs['DIRS_2'].update({'df':DIRS_2, 'path':'/'})

    DIRS_2_N=pd.merge(DIRS_2_Q, DIRS_2, on='INTFIS_RS_FD_KEY')
    DIRS_2_N=DIRS_2_N[['INTFIS_RS_FD_DIRECCION', 'INTFIS_RS_FD_PARTIAL_MATCH',
       'INTFIS_RS_FD_PLACE_ID', 'INTFIS_RS_FD_TYPES', 'INTFIS_RS_FD_KEY',
       'INTFIS_RS_FD_LAT',
       'INTFIS_RS_FD_LON', 'INTFIS_RS_FD_GEOM_TYPE',
       'INTFIS_RS_FD_PLUS_CODE',
       'CLAVECATASTRAL', 'CATASTRO_MUNICIPIO',
       'CATASTRO_ZONA', 'CATASTRO_MANZANA', 'CATASTRO_LOTE',
       'CATASTRO_EDIFICIO', 'CATASTRO_DEPARTAMENTO',
       'CATASTRO_ANIO_CONSTRUCCION', 'CATASTRO_ESTADO_CONSTRUCCION',
       'CATASTRO_NIVELES_CONSTRUCCION', 'CATASTRO_DOMICILIO_INMUEBLE',
       'CATASTRO_DOMICILIO_INMUEBLE_CONSTRUIDO',
       'CATASTRO_CANTIDAD_REPITE_CVECATASTRAL',
       'CATASTRO_DOMICILIO_CATASTRO']]

    ###AQUIIIIIIIII

if __name__ == "__main__":
    pipeline = Pipeline(cache=True)
    
    dict_dfs = {'CAT_N'      : {'df':None, 'path':'Downloads/Ecatepec claves  1 a 1 depurado.xlsx'},
                'E_PC'       : {'df':None, 'path':'Downloads/Ecatepec de Morelos_PREDIAL (1).xlsx'},
                'pred'       : {'df':None, 'path':'Downloads/prediosEcatepec_shapes.csv'},
                'DBF'        : {'df':None, 'path':'Downloads/ECATEPEC GC203T06 (nov. 2021).xlsx'},
                'ads'        : {'df':None, 'path':'Ecatepec_parcial.csv'},
                'DIRS_2_Q'   : {'df':None, 'path':'Downloads/ECATEPEC_CATASTRO_CALLES.txt'}, #sep='\t'
                'CLASS_E'    : {'df':None, 'path':'Downloads/ECATEPEC_CLASIFICADO.csv'}, #Creo que este no se usa (Validar)
                'MUESTRAS_F' : {'df':None, 'path':'Downloads/MUESTRAS_FINAL.csv'},
                'MUESTRAS_0' : {'df':None, 'path':'Downloads/Muestra_Ecatepec.xlsx'},
                'revs'       : {'df':None, 'path':'Downloads/Revisiones Ecatepec - Predial revisiones 3er dia.csv'},
                'GCP0'       : {'df':None, 'path':'Downloads/ECATEPEC_PARA_REV_GEO.txt'}, #sep='\t
                'DIRS_CALLES': {'df':None, 'path':'Downloads/ECATEPEC_CAT_QUERY_CALLES (1).txt'} #sep='\t', index=False
                }

    procesos = [
        Process(
                "Leyendo Archivos",
                read_files,
                should_cache=True,
        ),
        Process(
                "Clasificar Por Clave Catastral",
                classification_cve_catastral,
                should_cache=False,
        ),
        Process(
                "Limpiando Domiclios",
                clean_address,
                should_cache=False,
        ),
        Process(
                "Ajustes a Calles",
                street_adjustments,
                should_cache=False,
        ),
        Process(
                "Trabajando con Calles Catastro",
                streets_catastro,
                should_cache=False,
        )
    ]

    pipeline.add(procesos)
    pipeline.execute()


