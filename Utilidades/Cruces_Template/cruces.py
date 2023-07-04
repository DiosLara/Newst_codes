import pandas as pd

def cruce_instantaneo(df1:pd.DataFrame, df2:pd.DataFrame, 
                      nombre1:str, nombre2:str, 
                      tipo_cruce:str, key_cruce:str='key' ):
    '''
    (Function)
        Esta funcion realiza un cruce segun el tipo de cruce
    (Parameters)
        - df1: Base 1
        - df2: Base 2
        - nombre1: Nombre de la base 1
        - nombre2: Nombre de la base 2
        - tipo_cruce: inner, rigth, left, outer
        - key_cruce: Nombre de la columna de cruce, debe existir en ambas Bases
    '''
    # Hacemos el cruce, segun el parametro
    if tipo_cruce.lower() in ['left','solo '+nombre1,'solo_'+nombre1 ]:
        df_cruce = df1.merge(df2, how='left', on=key_cruce,
                                        indicator=True, 
                                        suffixes=('_'+str(nombre1), '_'+nombre2))
                                     #   validate='one_to_one')

    elif tipo_cruce.lower() in ['rigth', 'both', 'ambas']:
        df_cruce = df2.merge(df1, how='left', on=key_cruce,
                                        indicator=True, 
                                        suffixes=('_'+str(nombre1), '_'+nombre2))
                                        # validate='one_to_one')

    elif tipo_cruce.lower() in ['inner', 'both']:
        df_cruce = df1.merge(df2, how='inner', on=key_cruce,
                                        indicator=True, 
                                        suffixes=('_'+str(nombre1), '_'+nombre2))
                                        # validate='one_to_one')
    
    elif tipo_cruce.lower() in ['outer', 'all','todo', 'toda']:
        df_cruce = df1.merge(df2, how='outer', on=key_cruce,
                                        indicator=True, 
                                        suffixes=('_'+nombre1, '_'+nombre2))
                                        # validate='one_to_one')
    else:
        print('tipo de curce no valido')
        df_cruce = None
    
    return df_cruce