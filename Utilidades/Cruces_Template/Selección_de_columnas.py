import pandas as pd
def seleccion_columnas(df: pd.DataFrame):
    cols = list(df.columns)
    i = 0
    print('Columnas del dataframe ingresado:')
    for col in cols:
        print(str(i),": ",str(col))
        i+=1
    df['key'] = df[cols[int(input('Índice de la columna que se usará como key: '))]]
    df['Full name'] = df[cols[int(input('Índice de la columna que se usará como Nombre: '))]]
    return df


        
    