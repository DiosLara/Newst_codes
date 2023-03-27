import pandas as pd

# PREDIAL
def calculo_predial(valor_catastral: float, tabla = pd.read_csv('Predial.csv')):
    '''
    (Function)
        Función que calcula el impuesto predial con el valor catastral del predio y la tabla de rangos vigente
    (Parameters)
        valor_catastral: Valor catastral del predio, previamente calculado
        tabla: Tabla de rangos vigente    
        '''  
    for i in range(len(tabla)):
        if i<len(tabla)-1:
            if valor_catastral>=tabla.loc[i,'LIM_INF'] and valor_catastral<=tabla.loc[i,'LIM_SUP']:
                predial = tabla.loc[i,'CUOTA_FIJA'] + (valor_catastral-tabla.loc[i,'LIM_INF'])*tabla.loc[i,'FACTOR']
                break
        else:
            predial = tabla.loc[len(tabla)-1,'CUOTA_FIJA'] + (valor_catastral-tabla.loc[len(tabla)-1,'LIM_INF'])*tabla.loc[len(tabla)-1,'FACTOR']           
    return predial










