import pandas as pd
from typing import Union
from descripcion_grl import decribe_df
from Razones_sociales import Validador_len_rs
from estandarizacion_columnas import corregir_tipo_dato
from rapidfuzz import process, fuzz

class Validador:
    '''
    Esta clase, será para la validación de datos de un DataFrame

    Aun no esta definido, bien los metodos, pero como propuesta inicial se tiene:
    - Validador de RFCs; Un metodó para revisar, la estructura de cada RFC, y sea capaz de detectar
                         alguna anomalia.
    - Validador de Fechas; Un metodo que revise todas las fechas y verifique que sean reales y esten en tiempo
                           cronologico, de preferencia que imprima el rango de datos en el que se encuentra.
    - Suma de totales; Un metodo que sume los totales de las columnas numericas, para tenerlo como cotejo para
                        un cruce (o tratamiento de informacion) posterior
    - Validador tipo variables; Un metodo que te imprima por pantalla los campos que se toman como numericos
                           y en caso de ser necesario, los no numericos tambien.
    - Descripcion_grl; Un metodo que cuenta los valores unicos de cada variable, asi como la longitud de caracteres
    '''
    def __init__(self, base1:pd.DataFrame, base_name:str='Base1') -> None:
        self.df1 = corregir_tipo_dato(base1)
        #self.df1 = base1
        self.base_name = base_name
    

    def Quita_caracteres(self,columns:list,  caracteres:list=[",", ".","(",")",";",'[',"]",'"'],otros=None ): # -> pd.DataFrame:
        '''
        (Method)
            Este atributo quita los caracteres mas populares a este momento, puede agregar caracteres o bien 
            puede sustituir completamente los caracteres. El proposito de este atributo es quitar los signos de 
            puntuacion.
        (Paramaters)
            - columns:      Tambien puede ser un string, indicando una columna. Son las columnas a aplicar el 
                            metodo
            - caracteres:   Basicamente un iterable donde se alojan los caracteres a quitar
            - otros:        Se usa en caso de querer agregar caracteres a los que ya tiene, forzosamente una lista.
        (Returns)
            pd.DataFrame
        '''
        if type(otros)==list:
            caracteres.extend(otros)
            print('Agrego elementos a la lista: ', caracteres)
            

        if type(columns)== str:
            for car in caracteres:
                self.df1[columns] = self.df1[columns].astype(str).str.replace(car,'',regex=False).str.strip()
        else:
            try:
                for col in columns:
                    for car in caracteres:
                        self.df1[col] = self.df1[col].astype(str).str.replace(car,'',regex=False) 
            except Exception as e:
                raise e
            
        return self.df1

    def Quitar_acentos(self, columns:list) :
        '''
        (Method)
            Este metodo quita los acentos de las columnas seleccionadas.
        (Paramaters)
            - columns: Tambien puede ser un string, indicando una columna. Son las columnas a aplicar el 
                        metodo
        (Returns)
            pd.DataFrame
        '''
        try:
            for col in columns:
                self.df1[col] = self.df1[col].map(lambda x: self.__quitar_acentos(str(x)))
        except Exception as e:
            if type(columns)==str:
                col = columns
                self.df1[col] = self.df1[col].map(lambda x: self.__quitar_acentos(str(x)))
            else:
                raise e

        return self.df1

    def __quitar_acentos(self, s: str) :
        """
        (Attribute)
        Función que se encarga de remplazar en un string caracteres con acentos a caracteres sin acentuar que este en mayúsculas.

        (Parameters)
        s: String a procesar.

        (Returns)
        String sin acentuar en sus caracteres.
        """
        replacements = (('Á', 'A'), ('É', 'E'), ('Í', 'I'), ('Ó', 'O'), ('Ú', 'U'))
        s = s.upper()
        for a, b in replacements:
            s = s.replace(a, b)
        return s



    def panorama_grl(self, tipo:str=False, save_folder:str=''):
        '''
        (Method)
            Este metodo devuelve la informacion general del DataFrame
        (Parameters)
            - tipo: En caso de querer un arcvhio salida especifique "Excel" o "Csv" 
        (Returns)
            DataFrame
        '''
        resultado, rfc_invalidos = decribe_df(self.df1)
        #print(resultado)
        if tipo==False:
            pass
        elif tipo.upper() == 'EXCEL':
            resultado.to_excel(save_folder+'/Descripcion_general_'+self.base_name+'.xlsx', index=True)
        elif tipo.upper() == 'CSV':
            resultado.to_csv(save_folder+'/Descripcion_general_'+self.base_name+'.csv', index=True, encoding='utf-8-sig')
        return rfc_invalidos

    def validador_razon_social(self,columns: Union[list, tuple, str]):
        '''
        (Method)
            Este metodo Valida la razon social, toma los siguientes rubros en cuenta.
            - Cantidad de palabras: En mexico la cantida de palabras minimas en un nombre es de 3, Ambos 
              apeidos mas el nombre.
        (Paramaters)
            columns: Columnas que se consideran como Razon Social

        '''
        resultado = pd.DataFrame()
        try:
            if type(columns)==list:
                for col in columns:
                    res_aux = Validador_len_rs(self.df1[col])
                    resultado = pd.concat([resultado,res_aux])
                resultado.reset_index(drop=True, inplace=True)
        except Exception as e:
            if type(columns)==str:
                resultado = Validador_len_rs(self.df1[col])
            else:
                raise e
        return resultado

    def quitar_duplicados_x_columnas(self,df:pd.DataFrame):
        '''
        (Function)
            Esta Funcion es para quitar las columnas duplicadas, en caso de haber se quedara con la mas a la izquierda.
        (Parameters)
            - df: El DataFrame el cual, se cree que tiene 
        (Returns)
            DataFrame
        (Examples)
            df = pd.DataFrame({'A':[1,2,3],
                   'B':[4,5,6],
                   'C':[4,5,6],
                   'D':[9,6,3],
                   'E':[7,8,9],
                   })
            df.columns = ['A','B','A','B','C']
        '''
        cols_unica = []
        cols_rename = []
        columnas = df.columns

        for col in columnas:
            if col in cols_unica:
                cols_rename.append(str(col)+'_duplicado')
            else:
                cols_rename.append(col)
                cols_unica.append(col)
        df.columns = cols_rename
        df_clean = df[cols_unica]
        return df_clean
    
    

def quitar_duplicados_x_columnas(df:pd.DataFrame):
    '''
    (Function)
        Esta Funcion es para quitar las columnas duplicadas, en caso de haber se quedara con la mas a la izquierda.
    (Parameters)
        - df: El DataFrame el cual, se cree que tiene 
    (Returns)
        DataFrame
    (Examples)
        df = pd.DataFrame({'A':[1,2,3],
                'B':[4,5,6],
                'C':[4,5,6],
                'D':[9,6,3],
                'E':[7,8,9],
                })
        df.columns = ['A','B','A','B','C']
    '''
    cols_unica = []
    cols_rename = []
    columnas = df.columns

    for col in columnas:
        if col in cols_unica:
            cols_rename.append(str(col)+'_duplicado')
        else:
            cols_rename.append(col)
            cols_unica.append(col)
    df.columns = cols_rename
    df_clean = df[cols_unica]
    return df_clean

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


def verificaciones_sumas(df_cfdi_agg,df_issemym_agg,cruce0,col_verificaciones_cfdi:list=['Subtotal'],ancho_tabla = 47,
col_verificaciones = []):

    '''
    (Function)
        Funcion que imprime una tabla con el nombre de la columna y un Validador, en caso de coincidir los montos saldra True.
    (Parameters)
        -df_cfdi_agg: Agrupacion del DataFrame df_cfdi
        -df_issemym_agg: Agrupacion del DataFrame df_issemym
        -cruce0: Cruce de los 2 DataFrame anteriores (La agrupacion)
        -col_verificaciones_cfdi: Columnas para verificar Entre Cfdi vs Cruce
        -ancho_tabla: Parametro para el ancho de la tabla que se imprime
        -col_verificaciones_issemym: Columnas para verificar entre Issemym vs Cruce
    (Returns)
        None
    '''
    print('|','COLUMNA'.center(ancho_tabla,' '),'|', 'VALIDADOR'.center(10,' '),'|')
    print('|',''.center(ancho_tabla,'-'), '|', ''.center(10,'-'),'|')
    for col in col_verificaciones_cfdi:
        ver = round(df_cfdi_agg[col].sum(),0) == round(cruce0[col].sum(),0)
        print('|',col.ljust(ancho_tabla,' '),'|', str(ver).ljust(10,' '),'|')
        print('|',''.center(ancho_tabla,'-'), '|', ''.center(10,'-'),'|')
    
    omitir = ['Importe de Cuota de SaludImporte de Aportación de Salud','Conteo_issemym']

    #print('COLUMNA'.center(ancho_tabla,' '),'|', 'VALIDADOR'.center(10,' '),'|')
    #print('|',''.center(ancho_tabla,'-'), '|', ''.center(10,'-'),'|')
    for col in col_verificaciones_issemym:
        if col in omitir:
            continue
        ver = round(df_issemym_agg[col].sum(),0) == round(cruce0[col].sum(),0)
        print('|',col.ljust(ancho_tabla,' '),'|', str(ver).ljust(10,' '),'|')
        print('|',''.center(ancho_tabla,'-'), '|', ''.center(10,'-'),'|')
    return None

def agrupacion_de_bases(df_cfdi:pd.DataFrame, df_issemym:pd.DataFrame, cols_cfdi_agg=['NombreRazonSocialReceptor',"RfcReceptor","key"],
cols_issemym_agg:list=['CURP',"Full name","Clave ISSSEMYM","key"], dict_issemym_agg:dict={"Conteo_issemym":"count",'Sueldo Neto':sum,"Percepciones Cotizables":sum, 'Importe de Cuota SCI':sum, 
        'Importe Aportación SCI':sum,   'Importe de Cuota SCI Voluntaria':sum,'Importe de Aportación SCI Voluntaria':sum,
        'Importe de Cuota de Salud':sum,  'Importe de Aportación de Salud':sum, 'Importe de Cuota Solidario':sum,
        'Importe de Aportación Solidario':sum, 'Importe de Retención Institucional':sum,'Importe de Retención':sum, 'Importe de Cuota Extraordinaria':sum,
        'Importe de Aportación Extraordinaria':sum,'Importe de Aportación Gastos de Administración':sum,'Importe de Aportación Prima Básica':sum,
        'Importe de Aportación Prima de Servicios':sum,'Importe de Aportación de Prima Por Siniestros ':sum,
        'Importe de Aportación de Prima por Riesgos':sum}, dict_cfdi_agg:dict={"Subtotal":"sum","RfcEmisor":"count"},
        nombre_cfdi:str='NombreRazonSocialReceptor' , nombre_issemym:str='Full name' ):
    '''
    (Function)
        Funcion que hace una agrupacion por las columnas correspondientes
    (Paramaters)
        - df_cfdi: DataFrame correspondiente a GEM
        - df_issemym: DataFrame correspondiente a Issemym
        - cols_issemym_agg: Columnas que fungen como columnas para la agrupacion en la base Issemym
        - cols_cfdi_agg: Columnas que fungen como columnas para la agrupacion en la base GEM
        - dict_cfdi_agg: Diccionario que indica las columnas que se agruparan y el tipo de operacion que hace para GEM
        - dict_issemym_agg: Diccionario que indica las columnas que se agruparan y el tipo de operacion que hace para Issemym
        - nombre_cfdi: Nombre de la columna que tiene el nombre completo del contribuyente
        - nombre_issemym: Nombre de la columna que tiene el nombre completo del contribuyent
    (Returns)
        Dos DataFrames agrupados
    '''
    df_issemym_agg = df_issemym.groupby(by=cols_issemym_agg,as_index=False).agg(dict_issemym_agg).copy()
    df_cfdi_agg = df_cfdi.groupby(cols_cfdi_agg,as_index=False).agg(dict_cfdi_agg).copy()
    df_cfdi_agg['Full name'] = df_cfdi_agg[nombre_cfdi]
    df_issemym_agg['Nombre Completo Issemym'] = df_issemym_agg[nombre_issemym]
    return df_cfdi_agg, df_issemym_agg

def score_str_vs_str(comparar:str,busqueda:str,rapido:bool=True)->int:
    '''
    (Function)
        Funcion para comparar dos strings y dar una puntacion de similitud letra a letra con 2 variantes
        rapido=True
            La funcion compara letra a letra en el orden de ambas palabra letra[1]_a ==letra[1]_b
        rapido=False
            La funcion permite considerar que la letra buscada esta en una posicion anterior, en la misma o una posicion posterior 
            usada en casos de que la palabra tenga errores por omision o adicion de letra dentro del texto
            Ejemplo:
                #"Rehabilitacion" la palabra puede ser mal escrita sin la "h" 
                #y al seguir un orden exacto por posicion las letras siguientes pueden estar bien escritas pero el score saldra muy bajo 
                #por esta razon se aumento el rango de busqueda

                score_str_vs_str("Rehabilitacion","Reabilitacion",rapido=True) ##output 15.38%

                score_str_vs_str("Rehabilitacion","Reabilitacion",rapido=False) ##output 107.69%

    (Parameters)
        argumentos: "comparar" es un string que se tiene en un dataframe
                    "busqueda" es el string que se quiere encontrar una coincidencia
                    "rapido" es un boolean inicializado en True
    (Returns)
        Devuelve un valor porcentrual de similutud
    '''
    score=0
    if rapido:
        for x in range(len(comparar)):
            try:
                if busqueda[x]==comparar[x]:
                    score+=1
            except:
                pass
        return score/x*100
    else:
        for x in range(len(comparar)):
            for y in range(-1,2):
                try:
                    if busqueda[x+y]==comparar[x]:
                        score+=1
                except:
                    pass
        return score/min(len(comparar),len(busqueda))*100