import pandas as pd
import numpy as np
import datetime
def filtrar_df(df:pd.DataFrame,vup=10):
    '''
    (Function)
        Esta funcion pregunta al usuario si desea filtrar por alguna columna,en caso de querer filtrar despliega un menu
         mostrando las columnas disponibles para filtro, considera que puede tomar mas de una columna, Y  ademas muestra los valores disponibles
         para filtro.
    (Parameters)
        - df: El DataFrame que se desea filtrar
        - vup: valor unico permisible indica la cantidad de registros unicos por columna que se mostraran en el input
    (Return)
        Regresa el mismo DataFrame con los filtros correspondientes, en caso de haber querido el usuario
    '''
    df.columns=[str(x).replace(" ","_") for x in df.columns]
    for columna in df.columns:
        texto=[]
        if str(df[columna].dtype)=="object":
            unicos=list(df[columna].unique())
            try:
                unicos.remove(np.nan)
            except:
                pass
            if len(unicos)<vup and len(unicos)>1:
                for indice,valor in enumerate(unicos):
                    texto.append(str(indice+1)+": "+ str(valor))
                seleccion=int(input("Columna: "+columna+" Seleciona el indice del filtro "+", ".join(texto)+", 0: Ninguna "))
                if seleccion==0:
                    df=df
                else:
                    filtro=texto[seleccion-1].split(": ")[1]
                    df=df[df[columna]==filtro]
        elif str(df[columna].dtype)=="int64" or str(df[columna].dtype)=="float64":
            unicos=df[columna].unique()
            if len(unicos)!=1:
                if len(unicos)<vup:
                    for indice,valor in enumerate(unicos):
                        texto.append(str(indice+1)+": "+ str(valor))
                    print("valores: "+", ".join(texto))
                    seleccion2=input("Indicar el filtro sobre columna '"+columna+"' que numerico requerido sobre la columna (ejemplo: '>100') o 0 omitir columna ")
                else:
                    seleccion2=input("Indicar el filtro sobre columna '"+columna+"' que numerico requerido sobre la columna (ejemplo: '>100') o 0 omitir columna ")
                if seleccion2=="0" or seleccion2==0:
                    df=df
                else:
                    df=df.query(columna+seleccion2)
        if str(columna).upper().find("FECHA")>=0:
            fecha_f=int(input("Filtro de fecha 1, 0 omitir"))
            if fecha_f==1:
                seleccion3=datetime.datetime(int(input("año: ")),int(input("mes: ")),int(input("día: ")))
                df

    return df
 
