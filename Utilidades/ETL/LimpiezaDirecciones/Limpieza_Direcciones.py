#Librerías para conexión a Oracle
from OracleIntfis.db_connection import OracleDB
from telnetlib import PRAGMA_HEARTBEAT
import os
from dotenv import load_dotenv
#Librerías para conexión a PostgreSQL
from sqlalchemy import create_engine
import psycopg2
import pandas as pd
#Librerías para utilizar pandas y las funciones de csv
import csv
import pandas as pd
#Librería para normalizar String y utilizar la herramienta regex
import unidecode
import unicodedata 
from unicodedata import normalize
import re
#Librerías para funciones de distancias
from scipy.spatial import cKDTree
from shapely.geometry import Point


def get_data_oracle():
    #Conexión a la base de datos 

    load_dotenv(dotenv_path='') ##Aqui va la ruta donde tenemos nuestro env, solo la carpeta contenedora
    BASE_FOLDER = os.getenv("BASE_FOLDER") ##Aqui toma la ruta BASE_FOLDER que tenemos en el .env
    db = OracleDB("C:/.env") ## Poner ruta completa del archivo .env
    db_dom=db.query_to_df("SELECT * FROM DESA.DOMICILIOS_PRUEBA") #FETCH FIRST 100 ROWS ONLY")
    print(db_dom)
    return db_dom

def get_data_postgres():
    #Conexión a la base de datos 
    #postgresql://user:password@localhost:5432/Schema'

    engine = create_engine('')
    df = pd.read_sql("SELECT * FROM cursos", engine)
    print(df)
    return df

def get_data_doc(path, sheet_name):
    datos = pd.read_excel(path, sheet_name)
    #datos=pd.read_csv(path, sep="\t", encoding="utf8",low_memory=False, encoding_errors='ignore')
    #datos = pd.read_csv("prediosCoyotepec.txt", sep="\t", encoding="utf8",low_memory=False, encoding_errors='ignore')
    print(datos)
    return datos

def Data_Upper(datU):
    '''
    Función que se encarga de convertir un string en mayúsculas
    '''
    
    datU=str(datU)
    datU = unidecode.unidecode(datU.upper())

    return datU

def remove_diacritics_DIR(diac):  
    """"
        --Función de limpieza Direcciones--

    Se hace una homologación de nombres.

    * IMPORTANTE: Los datos que recibe deben ser de tipo String.

    Ejemplo: 

    Entrada:        CLUB ALPINO DE MEXICO  S.A. DE C.V.  
    Salida:         CLUB ALPINO DE MEXICO  

    """
    diac=str(diac)
    #strip quita espacios en blanco
    diac = re.sub(r"\s+",' ', diac)

    #A partir de aquí se utiliza Regex para cambiar abreviaturas por nombres completos.
    
    #diac = re.sub(r'(LC|L-|LT.)','LOTE ',diac)
    diac = re.sub(r'(MZ.|M-)','MANZANA ',diac)
    diac = re.sub(r'(CDA.|CDA)','CERRADA ',diac)
    diac = re.sub(r'(COND\.|CONDO\.)','CONDOMINIO ',diac)
    diac = re.sub(r'(SECC|SECC.)','SECCION ',diac)
    diac = re.sub(r'(S/N|SIN NUMERO|SIN NÚMERO)','SN',diac)
    diac = re.sub(r'(PROLG|PROL.|PROL)','PROLONGACION ',diac)
    diac = re.sub(r'(TERR)','TERRENO ',diac)
    diac = re.sub(r'(AV\.| AVE\.)','AVENIDA ',diac)
    diac = re.sub(r'(NO\.| NMERO)','NUMERO ',diac)
    diac = re.sub(r'(PBLICO)','PUBLICO ',diac)
    diac = re.sub(r'(SIST MUN|SIST MUNICIPAL)','SISTEMA MUNICIPAL ',diac)
    diac = diac.replace("PRESTC","PRESTACION")
    diac = diac.replace("MPAL", 'MUNICIPAL ')
    diac = re.sub(r'(ORG PUB|ORGPUBLICO)','ORGANISMO PUBLICO',diac)
    diac = diac.replace("C/",'CALLE ')
    diac = re.sub(r'S.A. DE C.V.|S.A DE C.V|S.A.|S.A|SA DE CV|S A|C V|CV|C.V|C.V.|SA', '',diac)
    diac = re.sub(r'COP.','COLECTIVO',diac)
    diac = re.sub(r'C. PROP.|C PROPIETARIO', 'PROPIETARIO', diac)
    diac = diac.replace("MTPLE.", "MULTIPLE")
    diac = diac.replace("GPO.", "GRUPO")
    diac = diac.replace("FINAN", "FINANCIERO")
    diac = diac.replace("INST.", "INSTITUTO")
    diac = diac.replace("ARREND.", "ARRENDADORA")
    diac = diac.replace("INMOB.", "INMOBILIARIA")
    diac = diac.replace("ELEC", "ELECTRONICA")
    diac = diac.replace("Y/O",'')
    
    #diac = diac.replace("AV.", 'AVENIDA')    
    #diac = diac.replace("NO.",'NUMERO')
    
    return diac

def remove_signs(sig):

    '''
    En esta función se recibe un dato de tipo String y si no es string se convierte a uno, quita todos los signos como puntos, comas y más.
    
    Ejemplo:
    Entrada:    CLUB ALPINO DE MEXICO  S.A. DE +++++++C.V.  
    Salida:     CLUB ALPINO DE MEXICO  SA DE CV  
    '''
    sig=str(sig)
    sig = sig.replace(".", '')
    sig = sig.replace(",", '')
    sig = sig.replace("-", '')
    #sig = sig.replace("\*", '')
    sig = sig.replace("`", '')
    sig = sig.replace("?", '')
    sig = sig.replace("#", '')
    sig = re.sub(r'(\*|\+|\/|\:|\¿)','',sig)
    sig = re.sub(r"\s+",' ', sig)

    #Quita los acentos y signos evitando la ñ
    sig = re.sub(r"([^cn\u0300-\u036f]|n(?!\u0303(?![\u0300-\u036f]))|c(?!\u0327(?![\u0300-\u036f])))[\u0300-\u036f]+", r"\1", 
        normalize( "NFD", sig), 0, re.I)
    # -> NFC
    sig = normalize('NFC', sig)
    

    return sig

#Función para filtrar por municipio

def transform_df_to_gpd(df: pd.DataFrame,lon_col: str = 'INTFIS_RS_FD_LON',lat_col: str = 'INTFIS_RS_FD_LAT', crs= 'EPSG:4326'):
        '''Transforma base a partir de dos columnas de latitud y longitud'''
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(
                df[lon_col], df[lat_col]
            ),
            crs=crs,
        )
        return gdf

#Funciones de limpieza para GEO

def distancia(C1, C2, score=80) -> pd.DataFrame:
    # Score mide el % de similitud entre c1 y c2
    score_sort = [
        (x,) + i
        for x in C1
        for i in process.extract(x, C2, scorer=fuzz.token_sort_ratio)
    ]

    # Create a dataframe from the tuples
    similarity_sort = pd.DataFrame(
        score_sort, columns=["P_B1_DIR", "B1DIR", "score_sort", "dunno"]
    )

    similarity_sort = similarity_sort.loc[
        similarity_sort["score_sort"] > score
    ].sort_values("score_sort")

    #similarity_sort.drop(columns=["score_sort", "dunno"], inplace=True)


    return similarity_sort

def base_dicts(base, dicc, tipo_dir="INTFIS_RS_FD_", tipo_dict="CALLE", score=80):
    diccionarios = distancia(
        dicc["nom_via"].astype(str),
        base[str(tipo_dir) + str(tipo_dict) + "_NORM"]
        .astype(str)
        .fillna("SIN DATO")
        .unique(),
        score,
    )
    
    assert isinstance(diccionarios, pd.DataFrame)

    diccionarios.rename(
        columns={"P_B1_DIR": tipo_dict, "B1DIR": str(tipo_dict) + str(tipo_dir)},
        inplace=True,
    )
    print(diccionarios.columns)
    diccionarios = diccionarios.drop_duplicates(str(tipo_dict) + str(tipo_dir))
    base_dict = pd.merge(
        base,
        diccionarios,
        left_on=str(tipo_dir) + str(tipo_dict) + "_NORM",
        right_on=str(tipo_dict) + str(tipo_dir),
        how="left",
    )
    base_dict[str(tipo_dict) + str(tipo_dir)] = base_dict[str(tipo_dict)].combine_first(
        base_dict[str(tipo_dir) + str(tipo_dict) + "_NORM"]
    )
    print(base_dict.shape)
    print(base_dict.columns)
    # print(base_dict.CLAVECATASTRAL.value_counts())
    
    return base_dict


def load_dict(
    base, base_folder: str, tipo_dir="INTFIS_RS_FD_", tipo_dict="CALLE", score=80
) -> pd.DataFrame:
    print(base_folder)
    path = base_folder + "\\INEGI\\" + str(tipo_dict) + ".csv"
    dicts = pd.read_csv(path, encoding="utf-8-sig")
    dicts["nom_via"] = normalize_names(dicts["nom_via"])
    base_dicc = base_dicts(base, dicts, tipo_dir, tipo_dict, score)

    return base_dicc


def full_doms(base_dict, tipo_dir="INTFIS_RS_FD_", n_ext_col= 'N_EXT', n_int_col='None') -> pd.DataFrame:
    base_dict[str(tipo_dir) + "FULL_DOM"] = base_dict["CALLE" + str(tipo_dir)].str.cat(
        base_dict[[str(tipo_dir) +  n_ext_col, "COLONIA" + str(tipo_dir)]].astype(str), " "
    )

    base_dict[str(tipo_dir) + "FULL_DOM"] = (
        base_dict[str(tipo_dir) + "FULL_DOM"]
        .astype(str)
        .str.replace("0.0", "", regex=False)
    )
    # tipo_dir='NOMINA_'
    if n_int_col!='None': 
        base_dict[str(tipo_dir) + n_int_col] = base_dict["CALLE" + str(tipo_dir)].str.cat(
            base_dict[[str(tipo_dir) +  n_ext_col,str(tipo_dir) + n_int_col, "COLONIA" + str(tipo_dir)]].astype(str), " "
        )

        base_dict[str(tipo_dir) + n_int_col] = (
        base_dict[str(tipo_dir) + n_int_col]
        .astype(str)
        .str.replace("0.0", "", regex=False)
        )

    return base_dict


def no_validos(base: pd.DataFrame) -> pd.DataFrame:
    base.loc[
        (base["col"] == 0)
        | (base["col"].isna())
        | (base["col"] == "0"),
        "ETIQUETA",
    ] = "NO VALIDO"

    return base


def renombrar_columnas_gcp(base_gcp: pd.DataFrame) -> pd.DataFrame:
    base_gcp.rename(
        columns={
            "geometry.location.lat": "geo_lat",
            "geometry.location.lng": "geo_lon",
            "plus_code.compound_code": "plus_code",
        },
        inplace=True,
    )

    base_gcp.columns = base_gcp.columns.str.upper()
   
    base_gcp.drop(
        columns=base_gcp.columns[base_gcp.columns.str.contains("_X|_Y")], inplace=True
    )

    base_gcp.columns = [
        str("INTFIS_RS_FD_") + column
        if not column.startswith("INTFIS_RS_FD_")
        else column
        for column in base_gcp.columns
    ]
 

    return base_gcp


def domicilios_intervalos(base_gcp: pd.DataFrame) -> pd.DataFrame:
    return base_gcp.loc[
        ~base_gcp["INTFIS_RS_FD_DIRECCION"].astype(str).str.contains("-", regex=False)
]

def procesamiento_con_diccionarios(base: pd.DataFrame, base_folder: str, tipo_dir: str) -> pd.DataFrame:
    return (
        base
        .pipe(limpieza_norm_dir, tipo_dir, debug=False)
        .pipe(load_dict, base_folder, tipo_dir, tipo_dict="CALLE", score=80)
        .pipe(load_dict, base_folder, tipo_dir, tipo_dict="COLONIA", score=80)
        .pipe(full_doms, tipo_dir)
    )

def diccionarios_gcp(base_gcp: pd.DataFrame, base_folder: str) -> pd.DataFrame:
    return (
        base_gcp
        .pipe(domicilios_intervalos)
        .pipe(procesamiento_con_diccionarios, base_folder, "INTFIS_RS_FD_")
        .pipe(renombrar_columnas_gcp)
    )

def nomina_diccionarios(base: pd.DataFrame, base_folder:str)-> pd.DataFrame:
    return (
        base
        .pipe(procesamiento_con_diccionarios, base_folder, "NOMINA_")
    )

def limpieza_base_externa(
    base_externa: pd.DataFrame, base_gcp: pd.DataFrame, base_folder: str
):
    # TODO: Mover la limpieza de GCP y base externa a una sección del repo que tenga sentido

    # Convertir columnas a mayusculas
    renombre_columnas = {columna: columna.upper() for columna in base_externa.columns}
    base_externa = base_externa.rename(columns=renombre_columnas)
    base_externa.replace("SIN DATO", np.nan, inplace=True)
    base_externa.replace("SIN_DATO", np.nan, inplace=True)

    # Convertimos la columna de fuente de origen a categorica
    base_externa = base_externa.astype({"BD_INTERNA_FUENTE_ORIGEN": "category"})

    logging.info(
        f"Numero de claves catastrales: {sum(~base_externa.CLAVECATASTRAL.isna())}"
    )

    base_externa = procesamiento_con_diccionarios(base_externa, base_folder, "BDINTERNA_CONT_")

    # Domicilios no válidos
    base_externa = no_validos(base_externa)
    base_externa["CLAVECATASTRAL_RECAUDACION"] = str("CAT_") + base_externa[
        "CLAVECATASTRAL"
    ].astype(str)

    logging.info(
        f"Numero de claves catastrales: {sum(~base_externa.CLAVECATASTRAL_RECAUDACION.isna())}"
    )

    return base_externa, base_gcp

# Normalización de los nombres

def normalize_string(string: str) -> str:
    try:
        string = unidecode.unidecode(string.strip().lower())
        string = string.replace(".", "")
        string = string.replace(",", " ")
    except Exception as e:
        raise e

    return string

def normalize_personal_names(names):
    names = names[~names.isna()].apply(lambda name: normalize_string(name))

    to_remove = [
        "S.A.P.I.",
        "SA DE CV",
        "DE TOLUCA",
        "\*",
        "-",
    ]

    dict_to_remove = {normalize_string(item): "" for item in to_remove}
    dict_to_remove[normalize_string("C. PROPIETARIO")] = "propietario"
    dict_to_remove[normalize_string("C.PROPIETARIO")] = "propietario"
    dict_to_remove[r"\ss\s*a\s*d\s*e\s*c\s*v"] = " " # Regex para remover combinaciones de 'SA de CV'
    dict_to_remove[r"\ss\s*a($|\s)"] = " " # Remover combinaciones de 'SA'
    dict_to_remove['"'] = ""
    dict_to_remove[r"y\s*(\/|-)\s*(o|)"] = ""
    dict_to_remove["de cv"] = ""
    dict_to_remove["cv"] = ""
    dict_to_remove[r"\s+"] = " " # Regex para convertir espacios multiples a simples

    names = names.replace(dict_to_remove, regex=True)
    names = names[~names.isna()].apply(lambda name: normalize_string(name))

    return names

def normalize_names(names):
    names = names[~names.isna()].apply(lambda name: normalize_string(name))

    to_remove = [
        "'",
        "\[",
        " - ",
    ]

    dict_to_remove = {normalize_string(item): "" for item in to_remove}
    dict_to_remove[normalize_string("S/N")] = "SN"
    dict_to_remove[normalize_string("/C")] = "CALLE"
    dict_to_remove[normalize_string("\*")] = ""
    dict_to_remove[normalize_string("PRIVADA")] = "PRIV"
    dict_to_remove[normalize_string("PVADA.")] = "PRIV"
    dict_to_remove[normalize_string("PVADA")] = "PRIV"
    dict_to_remove[normalize_string("PVAD.")] = "PRIV"
    dict_to_remove[normalize_string("PRIV.")] = "PRIV"
    dict_to_remove[normalize_string("PVDA.")] = "PRIV"
    dict_to_remove[normalize_string("PVDA")] = "PRIV"
    dict_to_remove[normalize_string("PVA.")] = "PRIV"
    dict_to_remove[normalize_string("PVA")] = "PRIV"
    dict_to_remove[normalize_string("BOULEVARD.")] = "BLVR"
    dict_to_remove[normalize_string("BOULEVARD")] = "BLVR"
    dict_to_remove[normalize_string("BOULEVERD")] = "BLVR"
    dict_to_remove[normalize_string("BOULEBARD")] = "BLVR"
    dict_to_remove[normalize_string("BOUVEVARD")] = "BLVR"
    dict_to_remove[normalize_string("BAULEVAR")] = "BLVR"
    dict_to_remove[normalize_string("BOULEVAR")] = "BLVR"
    dict_to_remove[normalize_string("BOULEVAD")] = "BLVR"
    dict_to_remove[normalize_string("BOULVERD")] = "BLVR"
    dict_to_remove[normalize_string("BOULVARD")] = "BLVR"
    dict_to_remove[normalize_string("BOULBD.")] = "BLVR"
    dict_to_remove[normalize_string("BOULEV")] = "BLVR"
    dict_to_remove[normalize_string("BOULD")] = "BLVR"
    dict_to_remove[normalize_string("BOULBD")] = "BLVR"
    dict_to_remove[normalize_string("BOULVD")] = "BLVR"
    dict_to_remove[normalize_string("BOULV.")] = "BLVR"
    dict_to_remove[normalize_string("BOULV")] = "BLVR"

    dict_to_remove[normalize_string("1ERO")] = "1RO"
    dict_to_remove[normalize_string("1O.")] = "1RO"
    dict_to_remove[normalize_string("1O")] = "1RO"
    dict_to_remove[normalize_string("2NDA")] = "2A"
    dict_to_remove[normalize_string("2DA")] = "2A"
    
    dict_to_remove[normalize_string("AVENIDA")] = "AV"
    dict_to_remove[r"\s+"] = " " # Regex para convertir espacios multiples a simples

    names = names.replace(dict_to_remove, regex=True)
    names = names[~names.isna()].apply(lambda name: normalize_string(name))

    return names

def simplify_names(names):

    replace_table = str.maketrans(dict.fromkeys('aeiou. ')) # Definimos los caracteres que queremos quitar
    return names[~names.isna()].apply(lambda name: name.translate(replace_table))
 
def distancia(columna1, columna2, score=85) -> pd.DataFrame:
    '''funcion para saber las coincidencia que existe entre dos palabras de diferentes columnas (todos contra todos)
    Ejemplo:
    columna1=df1["columna_comparar1"]
    columna2=df2["columna_comparar2"]
    score => porcentaje de confiabilidad de igualdad de palabras

    Similar,nosimilar=distancias(columna1,columna2)
    ó
    Similar,nosimilar=distancias(columna1,columna2,score=75)

    output:
    dos dataframe excluyentes
    el primero donde las palabras coincinden >= al score
    y el segundo donde las palabras estan por debajo del score pero sin considerar las palabras ya contenidas en el primero 
    '''
    # Score mide el % de similitud entre c1 y c2
#    score_sort = [(x,) + i 
 #                 for x in C1 
  #                for i in process.extract(x, C2, scorer=fuzz.token_sort_ratio)
   #              ]
    C1=columna1.str.upper()
    C1=C1.dropna().unique()
    C1=C1.astype(str)
    C2=columna2.str.upper()
    C2=C2.dropna().unique()
    C2=C2.astype(str)
    score_sort = []
    for x in C1:
        var = process.extract(x, C2, scorer=fuzz.token_sort_ratio)
        for i in var:
            if i[1] > score:
                score_sort.append((x,) + i)
    df1 = pd.DataFrame(score_sort).drop_duplicates(subset=[0])
    
    similarity_sort = pd.DataFrame(
        score_sort, columns=[columna1.name, columna2.name, "score_sort", "dunno"]
    )
    
    similarity_sort = similarity_sort.loc[
        similarity_sort["score_sort"] > score
    ].sort_values("score_sort", ascending=False)
    try:
        C1 = list(C1)
        for i in df1[0]:
            for j in C1:
                if i == j:
                    del C1[C1.index(j)]
    except:
        pass
    # Create a dataframe from the tuples
    score_sort = [(x,) + i 
                  for x in C1 
                  for i in process.extract(x, C2, scorer=fuzz.token_sort_ratio)
                 ]
########################################################################
    ########################################################################
    non_similarity_sort = pd.DataFrame(
        score_sort, columns=[columna1.name, columna2.name, "score_sort", "dunno"]
    )
    non_similarity_sort = non_similarity_sort.loc[
        non_similarity_sort["score_sort"] < score
    ].sort_values("score_sort", ascending=False)
    #similarity_sort.drop(columns=["score_sort", "dunno"], inplace=True)
    return similarity_sort, non_similarity_sort

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
            pd.Series(dist, name='min_dist')
        ], 
        axis=1)

    return gdf