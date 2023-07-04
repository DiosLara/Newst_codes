import pandas as pd 
import re
from rapidfuzz import process, fuzz

def verificaciones_sumas(df_cfdi_agg,df_issemym_agg,cruce0,
                        col_verificaciones_cfdi:list=[],
                        col_verificaciones_issemym = []):

    '''
    (Function)
        Funcion que imprime una tabla con el nombre de la columna y un Validador, en caso de coincidir los montos saldra True.
    (Parameters)
        - df_cfdi_agg: Un elemento del cruce
        - df_issemym_agg: Otro elemento del cruce
        - cruce0: Cruce de los 2 DataFrame anteriores 
        - col_verificaciones_cfdi: Columnas para verificar Entre Cfdi vs Cruce
        - col_verificaciones_issemym: Columnas para verificar entre Issemym vs Cruce
        - ancho_tabla: Parametro para el ancho de la tabla que se imprime
    (Returns)
        None
    '''
    ancho_tabla = 0
    cols = col_verificaciones_cfdi + col_verificaciones_issemym
    for i in range(len(cols)):
        if len(cols[i])>60:
            cols[i] = cols[i][:60]
    ancho_tabla = len(max(cols, key=len))
    if ancho_tabla == 60:
        ancho_tabla = 62
    print('|','COLUMNA'.center(ancho_tabla,' '),'|', 'VALIDADOR'.center(10,' '),'|')
    print('|',''.center(ancho_tabla,'-'), '|', ''.center(10,'-'),'|')
    for col in col_verificaciones_cfdi:
        try:
            ver = round(df_cfdi_agg[col].sum(),0) == round(cruce0[col].sum(),0)
            print('|',col.ljust(ancho_tabla,' '),'|', str(ver).ljust(10,' '),'|')
            print('|',''.center(ancho_tabla,'-'), '|', ''.center(10,'-'),'|')
        except:
            pass

    
    for col in col_verificaciones_issemym:
        try:
            ver = round(df_issemym_agg[col].sum(),0) == round(cruce0[col].sum(),0)
            print('|',col.ljust(ancho_tabla,' '),'|', str(ver).ljust(10,' '),'|')
            print('|',''.center(ancho_tabla,'-'), '|', ''.center(10,'-'),'|')
        except:
            pass

def Buscar_cols_numericas(df:pd.DataFrame):
    '''
    (Function)
        Esta funcion busca todas las columnas numericas que tenga el DataFrame, se consideran todas aquellas que tiene como dtype igual
        a int o float, en caso de ser numerica y no tener el tipo de dato adecuado, pues no la considera.
    (Parameteres)
        df: DataFrame del cual se busca extraer las col numericas.
    '''
    datos=pd.DataFrame(df.dtypes,columns=["Anterior"])
    cols_numericas = datos.loc[(datos['Anterior'].astype(str).str.contains('int', na=False)) | (datos['Anterior'].astype(str).str.contains('float', na=False))].index.values
    return list(cols_numericas)

def distancia(C1, C2, score=80) :
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

def solucion_key_duplicada(cruce0:pd.DataFrame, df_cfdi:pd.DataFrame, 
                            col_key:str='key',
                           col_nombrerazonsocial:str='NombreRazonSocialReceptor',
                           col_nombreissemym:str='Nombre Completo Issemym'):
    '''
    (Function)
        Esta funcion revisa los nombres de 
    (Parameters)
        - col_key: Nombre de la columna que funge como key
        - cruce0: DataFrame que tiene el cruce de dos tablas
        - df_cfdi: Es un DataFrame de los cuales salio cruce0, es basicamente el DataFrame en el cual se haran los cambios y el mismo que saldra en Return.
        - col_nombrerazonsocial: Nombre de la columna que tiene el Nombre completo en el DataFrame df_cfdi
        - col_nombreissemym: Nombre de la columna que tiene el Nombre completo en el DataFrame df_issemym, puedes pensarlo como la columna con el nombre correcto que se sustituira en 
                             col_nombrerazonsocial
    '''
    
    key_cruce_duplicada = cruce0[cruce0[col_key].duplicated()][col_key]
    cruce0_key_repetida = cruce0[((cruce0[col_key].isin(key_cruce_duplicada)))].sort_values(col_key)
    keyss=[]
    cont = 0

    for key in cruce0_key_repetida[col_key].unique():
        try:
            rs_cfdi = cruce0_key_repetida.loc[(cruce0_key_repetida[col_key]==key)&(cruce0_key_repetida[col_nombrerazonsocial].fillna('Vacio')!='Vacio'), col_nombrerazonsocial].values
            rs_issemym = cruce0_key_repetida.loc[(cruce0_key_repetida[col_key]==key)&(cruce0_key_repetida[col_nombrerazonsocial].fillna('Vacio')=='Vacio'), col_nombreissemym].values

            rs_cfdi1=[rs_cfdi[0][:40]]
            rs_cfdi1.append(str(rs_cfdi1[0]).replace("Ñ","N"))
            rs_issemym1=[rs_issemym[0][:40]]
            
            score_d = distancia(rs_issemym1,rs_cfdi1)
            if score_d['score_sort'].values[0] > 95 or score_d['score_sort'].values[1] > 95:
                print('Cambiar nombre del cfdi por el de Issemym')
                print(score_d['score_sort'])
                print(rs_cfdi[0], ' --> ',rs_issemym[0] )
                df_cfdi.loc[df_cfdi[col_nombrerazonsocial]==rs_cfdi[0], col_nombrerazonsocial] = rs_issemym[0]
                cont += 1

            else:
                ans = input(f'Se comparo {rs_cfdi[0]} vs {rs_issemym[0]}, asumiendo llaves iguales ({score_d["score_sort"]}). Asumismo que es la misma persona? (1:si, 0:no)')
                if ans == '1':
                    print('Cambiar nombre del cfdi por el de Issemym')
                    print(score_d['score_sort'])
                    print(rs_cfdi[0], ' --> ',rs_issemym[0] )
                    df_cfdi.loc[df_cfdi[col_nombrerazonsocial]==rs_cfdi[0], col_nombrerazonsocial] = rs_issemym[0]
                    cont += 1
                elif ans.lower() == 'break':
                    break
                else:
                    print('Se omitio')

        except Exception as e:
            keyss.append(key)
            print(key, ' No valido  ', e)
            continue
        print('--'.center(45,'-'))
    print('Se hicieron ', cont, ' cambios') 
    return df_cfdi 


def solucion_key_duplicada(cruce0:pd.DataFrame, df_cfdi:pd.DataFrame, 
                           col_key:str='key',
                           col_nombrerazonsocial:str='NombreRazonSocialReceptor',
                           col_nombreissemym:str='Nombre Completo Issemym'):
    '''
    (Function)
        Esta funcion revisa los keys duplicadas en el cruce, en caso de hacer un cruce por dos llaves (rfc y razon social),
        y corrige los nombres en df_cfdi, para que al rehacer el cruce estos hagan match.
    (Parameters)
        - col_key: Nombre de la columna que funge como key
        - cruce0: DataFrame que tiene el cruce de dos tablas
        - df_cfdi: Es un DataFrame de los cuales salio cruce0, es basicamente el DataFrame en el cual se haran los cambios y el mismo que saldra en Return.
        - col_nombrerazonsocial: Nombre de la columna que tiene el Nombre completo en el DataFrame df_cfdi
        - col_nombreissemym: Nombre de la columna que tiene el Nombre completo en el DataFrame df_issemym, puedes pensarlo como la columna con el nombre correcto que se sustituira en 
                             col_nombrerazonsocial
    '''
    
    key_cruce_duplicada = cruce0[cruce0[col_key].duplicated()][col_key]
    cruce0_key_repetida = cruce0[((cruce0[col_key].isin(key_cruce_duplicada)))].sort_values(col_key)
    keyss=[]
    cont = 0

    for key in cruce0_key_repetida[col_key].unique():
        try:
            rs_cfdi = cruce0_key_repetida.loc[(cruce0_key_repetida[col_key]==key)&(cruce0_key_repetida[col_nombrerazonsocial].fillna('Vacio')!='Vacio'), col_nombrerazonsocial].values
            rs_issemym = cruce0_key_repetida.loc[(cruce0_key_repetida[col_key]==key)&(cruce0_key_repetida[col_nombrerazonsocial].fillna('Vacio')=='Vacio'), col_nombreissemym].values

            rs_cfdi1=[rs_cfdi[0][:40]]
            rs_cfdi1.append(str(rs_cfdi1[0]).replace("Ñ","N"))
            rs_issemym1=[rs_issemym[0][:40]]
            
            score_d = distancia(rs_issemym1,rs_cfdi1)
            if score_d['score_sort'].values[0] > 95 or score_d['score_sort'].values[1] > 95:
                print('Cambiar nombre del cfdi por el de Issemym')
               # print(score_d['score_sort'])
                print(rs_cfdi[0], ' --> ',rs_issemym[0] )
                df_cfdi.loc[df_cfdi[col_nombrerazonsocial]==rs_cfdi[0], col_nombrerazonsocial] = rs_issemym[0]
                cont += 1

            else:
                ans = input(f'Se comparo {rs_cfdi[0]} vs {rs_issemym[0]}, asumiendo llaves iguales ({score_d["score_sort"]}). Asumismo que es la misma persona? (1:si, 0:no)')
                if ans == '1':
                    print('Cambiar nombre del cfdi por el de Issemym')
                   # print(score_d['score_sort'])
                    print(rs_cfdi[0], ' --> ',rs_issemym[0] )
                    df_cfdi.loc[df_cfdi[col_nombrerazonsocial]==rs_cfdi[0], col_nombrerazonsocial] = rs_issemym[0]
                    cont += 1
                elif ans.lower() in ['break', 'quitar', 'romper', 'salir', '28']:
                    break
                else:
                    pass
                    #print('Se omitio')

        except Exception as e:
            keyss.append(key)
            print(key, ' No valido  ', e)
            continue
        print('--'.center(45,'-'))
    print('Se hicieron ', cont, ' cambios') 
    return df_cfdi 

def score_str_vs_str(comparar:str,busqueda:str,rapido:bool=True): 
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

def solucion_nombre_duplicado(cruce0:pd.DataFrame, df_cfdi:pd.DataFrame, 
                           col_key:str='key',
                           col_nombrerazonsocial:str='NombreRazonSocialReceptor',
                           col_nombreissemym:str='Nombre Completo Issemym'):
    '''
    (Function)
        Esta funcion revisa los nombres del cruce por duplicados, solo sirve cuando la key es el RFC o el CURP
        ya que considera la fechas y en caso de haber una variacion de 5 dias en la fecha, corrige la key en df_cfdi, para 
        que al rehacer el cruce, las que salian duplicadas por key diferente, ahora haran match.
    (Parameters)
        - col_key: Nombre de la columna que funge como key
        - cruce0: DataFrame que tiene el cruce de dos tablas
        - df_cfdi: Es un DataFrame de los cuales salio cruce0, es basicamente el DataFrame en el cual se haran los cambios y el mismo que saldra en Return.
        - col_nombrerazonsocial: Nombre de la columna que tiene el Nombre completo en el DataFrame df_cfdi
        - col_nombreissemym: Nombre de la columna que tiene el Nombre completo en el DataFrame df_issemym, puedes pensarlo como la columna con el nombre correcto que se sustituira en 
                             col_nombrerazonsocial
    '''
    rs_cruce_duplicada = cruce0[cruce0['Full name'].duplicated()]['Full name'] 
    cruce0_rs_repetida = cruce0[(cruce0['Full name'].isin(rs_cruce_duplicada)) & (cruce0['_merge']!='both') ].sort_values('Full name')
    fecha_regex=re.compile("\d\d\d\d\d\d")
    cont = 0
    for rs in cruce0_rs_repetida['Full name'].unique():
        try:
            key_cfdi    = cruce0_rs_repetida.loc[(cruce0_rs_repetida['Full name']==rs)&(cruce0_rs_repetida[col_nombrerazonsocial].fillna('Vacio')!='Vacio'), 'key'].values[0]
            key_issemym = cruce0_rs_repetida.loc[(cruce0_rs_repetida['Full name']==rs)&(cruce0_rs_repetida[col_nombrerazonsocial].fillna('Vacio')=='Vacio'), 'key'].values[0]
            key_cfdi_n    = fecha_regex.findall(key_cfdi)[0]
            key_issemym_n = fecha_regex.findall(key_issemym)[0]
            print(key_cfdi, key_issemym)
            
            score_d = score_str_vs_str(key_cfdi_n,key_issemym_n, rapido=False)
            if score_d >95:
                print('Cambiar key del cfdi por el de Issemym')
                print(key_cfdi, ' --> ',key_issemym )
                #print(score_d)
                df_cfdi.loc[df_cfdi['key']==key_cfdi, 'key'] = key_issemym
                cont += 1
            elif abs( int(str(key_cfdi_n)[-2:]) - int(str(key_issemym_n)[-2:]))<=5:
                print('Cambiar key del cfdi por el de Issemym')
                print(key_cfdi, ' --> ',key_issemym )
                #print(score_d)
                df_cfdi.loc[df_cfdi['key']==key_cfdi, 'key'] = key_issemym
                cont += 1
                
        except:
            print(rs, ' No valido')
    print('Se hicieron ', cont, ' cambios')  
    return df_cfdi