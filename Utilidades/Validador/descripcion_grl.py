import re
import pandas as pd
import numpy as np

# Blindar codigo para que no se rompa al sumar los tipos de datos (ej; datetime)

def mode(dataset):
    """Funcion para detectar la moda en una serie de datos"""
    frequency = {}
    for value in dataset:
        frequency[value] = frequency.get(value, 0) + 1
    most_frequent = max(frequency.values())
    modes = [key for key, value in frequency.items()
                      if value == most_frequent]
    return modes

data_regex=re.compile("^\w[\w|&]\w\w?\d{6}[A-Z0-9]{3}$")
def decribe_df(df:pd.DataFrame) :
    """Funcion para generar una descripcion de un DataFrame"""
    resultado=pd.DataFrame()
    data_invalidos=[]
    for i in df.columns:
        try:
            resultado.loc[i,"Registros unicos"]=len(df[i].unique())
            if df.fillna(0)[i].dtype!="O":
                resultado.loc[i,"Suma"]=df[i].sum()
                resultado.loc[i,"Promedio"]=df[i].mean()
                resultado.loc[i,"Maximo"]=df[i].max()
                resultado.loc[i,"Minimo"]=df[i].min()
            else:
                ###Fechas
                ##conteo de elementos por elemento
                resultado.loc[i,"Longitud de Caracteres"]="|".join([str(y) for y in list(set([len(str(x)) for x in df[i]]))])
            if i.upper().find("data")>=0:
                validos=[]
                for x in df[i].unique():
                    try:
                        if str(x)[-3:]!="000" or (str(x)[-3]!=str(x)[-2]!=str(x)[-1]):
                            if len(data_regex.findall(str(x).upper().strip().replace(" ","")))>0:
                                validos.append(1)
                            else:
                                data_invalidos.append(x)
                        else:
                            data_invalidos.append(x)
                    except:
                        pass
                
                resultado.loc[i,"data_invalidos"]=len(df[i].unique())-np.sum(validos)
        except:
            print(i, ':  Columna no agregada')
    
        resultado.loc[i,"Cantidad de elementos por valores unicos"]="|".join([str(x) for x in df[i].value_counts().unique()])
        try:
            a=pd.DataFrame(df[i].value_counts())
            resultado.loc[i,"repeticion mayoritaria"]=mode(a[i])
        except:
            pass
        if len(data_invalidos)>0:
            print("Hay data invalidos")

    return resultado.fillna(""),data_invalidos
