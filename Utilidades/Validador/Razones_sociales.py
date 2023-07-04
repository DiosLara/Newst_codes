import pandas as pd

def cuenta_palabras(rs:str):
    '''
    (Function)
        Esta funcion cuenta las palabras que tiene el input (rs)
    (Paramaters)
        -rs: Cadena de texto, a la cual se quieren contar las palabras
    (Return)
        str
    '''
    rs = rs.split()
    return len(rs)

def Validador_len_rs(col:pd.Series):
    '''
    (Method)
        Esta funcion crea un DataFrame con uno de los campos la Serie que entra y ademas otro campo
        cuyo nombre es "Valido", el cual es 1 si es valido, 0 lo contario
    (Paramaters)
        col: Es una serie a la cual se hara la validacion de los datos
    (Return)
        DataFrame
    '''
    df = pd.DataFrame(col)
    df['total_word'] = col.map(lambda x: cuenta_palabras(x))
    df['Valido'] = df['total_word'].map(lambda x: 1 if x>=3 else 0)
    return df[[col.name,'Valido']]