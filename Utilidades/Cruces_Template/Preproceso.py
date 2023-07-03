import datetime
import re

import pandas as pd
import warnings
from Postproceso import distancia
warnings.filterwarnings('ignore')


numero=re.compile("\d\,\d|\d\.\d|\d")
fecha=re.compile("^(?P<day>\d\d?)[\-|\/|\s](?P<month>\d\d?)[\-|\/|\s](?P<year>\d\d\d\d)|^(?P<year1>\d\d\d\d)[\-|\/|\s](?P<month1>\d\d?)[\-|\/|\s](?P<day1>\d\d?)")


def seleccion_cols_principales (df:pd.DataFrame):
    '''
    (Function)
        Esta funcion imprime las columnas disponibles con un indice, para que el usuario seleccione cual columna usara 
        para cada caso (llave y Nombre)
    (Parameters)
        df: Dataframe que se desea saber las columna(s) principales
    '''
    len_cols = df.columns.map(lambda x: len(str(x)))
    ancho_tabla = len_cols.max() + 1
    if ancho_tabla >60:
        ancho_tabla = 62
    cols_filtrar = 'Las columnas disponibles  son: '.center(ancho_tabla+10,'*') + '\n'
    i = 0
    cols_filtrar += '|'+'Indice'.center(8,' ')+'|'+ 'COLUMNA'.center(ancho_tabla,' ')+'|' + '\n'
    cols_filtrar += '|'+''.center(8,'-')+ '|'+ ''.center(ancho_tabla,'-') + '|' + '\n'
    for columna in df.columns:
        cols_filtrar += '|' + str(i).center(8,' ') + '|' +  str(columna)[:61].center(ancho_tabla,' ') + '|' + '\n'
        cols_filtrar += '|' + ''.center(8,'-') + '|' + ''.center(ancho_tabla,'-') + '|' + '\n'
        i += 1
    print(cols_filtrar)
    indicadora = True
    while indicadora:
        ans1 = input('Seleccione el indice de la columna que contiene el Nombre o Razon Social:  ')
        try:
            cols_nombre = df.columns[int(ans1)]
            indicadora = False
        except:
            print('Indice invalido, intentalo otra vez')

    indicadora = True
    while indicadora:
        ans1 = input('Seleccione el indice de la columna que contiene la key:  ')
        try:
            cols_key = df.columns[int(ans1)]
            indicadora = False
        except:
            print('Indice invalido, intentalo otra vez')
    ans = input('Desea tomar los 10 primeros digitos de la columna que selecciono como llave?:  ')
    if ans.lower() in ['si', 'yes', 'claro', '1', 'sip', 'sipi', 'afirmativo', 'por supesto', 'ok','porsupollo', 'sincho']:
        df['key'] = df[cols_key].map(lambda x: str(x)[:10])
    else:
        df['key'] = df[cols_key]
    
    return df, cols_nombre, cols_key



def corregir_tipo_dato(df:pd.DataFrame,corregir=False)->pd.DataFrame:
    """Funcion cambia los tipos de datos al dato identificado automaticamente y retorna el dataframe"""
    datos=pd.DataFrame(df.dtypes,columns=["Anterior"])
    validador_date=0
    for colum in df.columns:
        if str(datos.loc[colum,"Anterior"])=="object":
            valores=df[colum].unique()
            a=0
            df[colum]=df[colum].astype("str")
            for val in valores[:100]:
                if val=="":
                    print(colum+" con valores nulos")
                try:
                    m=fecha.match(val)
                    if m.group("year"):
                        date=datetime.date(int(m.group("year")),int(m.group("month")),int(m.group("day")))
                    else:
                        date=datetime.date(int(m.group("year1")),int(m.group("month1")),int(m.group("day1")))
                    df.loc[df[colum]==val,colum]=date
                    validador_date=1
                except:
                    validador_date=0
                    try:
                        df[colum]=df[colum].astype("int64")
                    except:
                        try:
                            validador_numero=numero.findall(val)
                            if len(validador_numero)>0:
                                a+=1
                            else:
#                                 print(val)
                                if a>0:
                                    if corregir:
                                       # print("Columna: "+colum+" Tiene dato invalido: "+val+" corregido a 0")
                                        df.loc[df[colum]==val,colum]="0"
                                    else:
                                       # print("Columna: "+colum+" Tiene dato invalido: "+val)
                                        pass
                        except:
                            pass


            if a>0:
                df[colum]=df[colum].str.replace(",","")
                try:
                    df=df.astype({colum:"float64"})
                except:
                    pass
            if validador_date==1:
                df=df.astype({colum:"datetime64[ns]"})
    datos=pd.concat([datos,pd.DataFrame(df.dtypes,columns=["Nueva"])],axis=1)
    #print(datos)
    return df


    import re
import pandas as pd


vocales=re.compile("[A|E|I|O|U]")

def preRFC(rz):
    """Funcion genera dos posibles combinaciones de RFC siendo que el input sea un nombre en cualquier orden y genere ambas combinaciones"""
    rz=str(rz).upper().split()
    try:
        c1=rz[0][0]+vocales.findall(rz[0][1:])[0]+rz[1][0]
        #c2=rz[-2][0]+vocales.findall(rz[-2][1:])[0]+rz[-1][0]
        return c1 #"|".join([c1,c2])
    except:
        return "|".join(["",""])

def preprocesing(df_concentrado1:pd.DataFrame,
                 col_nombre:str, 
                 col_key:str='key',
                 cols_agrupacion:list=[], 
                 tipo_agrupacion:str='max',
                 score:int = 88):
    '''
    (Function)
        Esta funcion hace un preprocesado para detectar defectuosos en la columna llave, es indispensable contar 
        con la columna Nombre, ya que se hace un discriminante para validar el RFC, solo funciona si la col_key es el 
        RFC o el CURP. Luego de haber identificado los duplicados y haber pasado el discrimanente, se hara una agrupacion
        de para deshacer los duplicados, por ello debe tener cuidado en como selecciona el tipo de agrupacion.
    (Parameters)
        - df_concentrado1: DataFrame que se desea cruzar el cual se le hara el preprocesado
        - col_key: Nombre de la columna que tiene el RFC o el CURP
        - col_nombre: Nombre de la columna que tiene la Razon Social
        - cols_agrupacion: Una lista de todas las otras columnas que se desa mantener ya que en caso de encontrar 
                            duplicados, estos se pasan por el discriminante y en caso, se agruparan rescantando las columnas que 
                            agregue, No incluya las dos que ya indico; "col_key" y "col_nombre"
        - preRFC: Funcion que genera el discriminante
        - tipo_agrupacion: Inidique si es suma o promedio.
    (Returns)
        Retorna el df_concentrado1, que entro inicialmente con las correcciones pertinentes.
        Ademas retorna otro DataFrame, que es "Defectuoso", pero sencillamente puede reutilizarlo para cruzarlo pero esta vez
        utilize dos llaves; key & Razon Social
    '''
    # Revisamos el col_key
    if isinstance(col_key,list):
        key = col_key[0]
    else:
        key = col_key

    # Encuentro todas aquellas llaves que este duplicadas
    key_concentrado_duplicada = df_concentrado1.loc[df_concentrado1.key.duplicated()]['key']
    df_defect_concentrado = df_concentrado1[df_concentrado1[key].isin(key_concentrado_duplicada)].sort_values(col_key)#['key']
    
    # Los borramos del original
    df_concentrado1.drop(index=df_defect_concentrado.index, inplace=True)
    print('Pre del duplicated_key')

    # Como df_defect_concentrado tienen keys duplicadas, podria deberse a un error de tipado al capturarlos. Asi que estandarizamos los nombres
    if isinstance(col_key, list):
        df_defect_concentrado = duplicated_key(df_defect_concentrado,col_key=col_key[0], col_nombre=col_nombre, score=score)
    else:
        df_defect_concentrado = duplicated_key(df_defect_concentrado,col_key=col_key, col_nombre=col_nombre, score=score)

    # Generamos discriminantes
    print('Llegamos al discriminante')
    df_defect_concentrado['Discriminante_fn'] = df_defect_concentrado[col_nombre].map(lambda x: preRFC(x) )
    df_defect_concentrado['Discriminante_rfc'] = df_defect_concentrado[key].map(lambda x: str(x)[:3] )

    # Reciclamos los utiles segun los discriminantes
    df_reciclado_concentrado = df_defect_concentrado[df_defect_concentrado['Discriminante_rfc'] == df_defect_concentrado['Discriminante_fn']]

    # Borramos de los defectuosos los reciclados
    df_defect_concentrado.drop(index=df_reciclado_concentrado.index, inplace=True)

    # Apendamos las dos cols en la agrupacion
    try:
        if isinstance(col_key, list):
            col_key.append(col_nombre)
            cols_agrupacion.extend(col_key)
        else:
            cols_agrupacion.extend([col_nombre, col_key])
    except:
        print('El parametro cols_agrupacion, debe ser una lista')

    ### Agrupamos los reciclados, ya que se duplica la key
    if  tipo_agrupacion.lower() in ['max', 'maximo', 'mas grande']:
        df_reciclado_concentrado1 = df_reciclado_concentrado.groupby(by=cols_agrupacion, as_index=False).max()

    elif tipo_agrupacion.lower() in ['sum', 'suma']:
        df_reciclado_concentrado1 = df_reciclado_concentrado.groupby(by=cols_agrupacion, as_index=False).sum()
    
    elif tipo_agrupacion.lower() in ['mean', 'promedio', 'average']:
        df_reciclado_concentrado1 = df_reciclado_concentrado.groupby(by=cols_agrupacion, as_index=False).mean()
    
    elif tipo_agrupacion.lower() in ['min', 'minimo', 'mas pequeño']:
        df_reciclado_concentrado1 = df_reciclado_concentrado.groupby(by=cols_agrupacion, as_index=False).min()
    
    else:
        raise ValueError('Debe indicar adecuademente la variable "tipo_agrupacion"')
    

    # Concateno el concentrado original con el reciclado
    df_concentrado1 = pd.concat([df_concentrado1, df_reciclado_concentrado1])
    df_concentrado1.pop('Discriminante_fn')
    df_concentrado1.pop('Discriminante_rfc')
    df_concentrado1.reset_index(drop=True, inplace=True)
    
    return df_concentrado1, df_defect_concentrado

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



def Quita_caracteres(df1, columns:list,  caracteres:list=[",", ".","(",")",";",'[',"]",'"'],otros=None ): # -> pd.DataFrame:
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
            df1[columns] = df1[columns].astype(str).str.replace(car,'',regex=False).str.strip()
    else:
        try:
            for col in columns:
                for car in caracteres:
                    df1[col] = df1[col].astype(str).str.replace(car,'',regex=False) 
        except Exception as e:
            raise e
        
    return df1

def Quitar_acentos(df1:pd.DataFrame, columns:list) :
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
        if isinstance(columns, list):
           # print('Ingreo listas')
            for col in columns:
                df1[col] = df1[col].map(lambda x: __quitar_acentos(str(x)))

        elif isinstance(columns, str):
           # print('Ingreso str')
            col = columns
            df1[col] = df1[col].map(lambda x: __quitar_acentos(str(x)))

    except Exception as e:
        print('Ingreso columnas invalidas para Quitar los acentos', columns, '\n',e)

    return df1

def __quitar_acentos( s: str) :
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
    return s.strip()

def Estandarisacion_str_col(df1:pd.DataFrame, columns:list, otros):
    '''
    (Function)
        Esta Funcion Estandariza las columnas string que se especifican en el parametro "columns", quita caracteres como:
        [",", ".","(",")",";",'[',"]",'"'], pero puede agregar mas si lo desea.
        Ademas quita los acentos y hace una limpieza de espacios iniciales y finales.
    (Paramaters)

        - columns:      Tambien puede ser un string, indicando una columna. Son las columnas a aplicar el 
                        metodo
        - caracteres:   Basicamente un iterable donde se alojan los caracteres a quitar
        
    '''
    df1 = Quita_caracteres(df1=df1, columns=columns, otros=otros)
    df1 = Quitar_acentos(df1=df1,columns=columns)
    return df1

def duplicated_key(df_duplicated:pd.DataFrame, col_key:str, col_nombre:str, score:int=88):
    '''
    (Function)
        Esta función se enfoca en los valores duplicados por llave, hace una homologación de nombres para cada llave duplicada.
    (Parameters)
        - df_duplicated: DataFrame que se desea procesar para la homologación de nombres
        
    (Returns)
        Retorna el df_duplicated, que entro inicialmente con las correcciones pertinentes.
    '''

    for claves in df_duplicated[col_key].unique():
        C1 = df_duplicated.loc[df_duplicated[col_key]==claves, col_nombre]
        C1.reset_index(drop=True, inplace=True)
        key1 = claves
        dist=distancia([C1[0]], [C1[1]], score=80)
        
        if dist['score_sort'].values >= score:      
            df_duplicated.loc[df_duplicated['key']==key1 , col_nombre] = dist['P_B1_DIR'][0]
        
     
    return df_duplicated
