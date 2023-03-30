import pandas as pd

# PREDIAL
def calculo_predial(base , tabla = pd.read_csv(r'C:\Users\dlara\Downloads\Predial.csv')):
    '''
    (Function)
        Función que calcula el impuesto predial con el valor catastral del predio y la tabla de rangos vigente
    (Parameters)
        valor_catastral: Valor catastral del predio, previamente calculado
        tabla: Tabla de rangos vigente    
        '''  
    for i in tabla.index: 
        base.loc[((base['VALORCATASTRAL']<=tabla['LIM_SUP'][i])) & ((base['VALORCATASTRAL']>tabla['LIM_INF'][i])), 'LIM_INF'] = tabla['LIM_INF'][i]
        base.loc[((base['VALORCATASTRAL']<=tabla['LIM_SUP'][i])) & ((base['VALORCATASTRAL']>tabla['LIM_INF'][i])), 'CUOTA_FIJA'] = tabla['CUOTA_FIJA'][i]
        base.loc[((base['VALORCATASTRAL']<=tabla['LIM_SUP'][i])) & ((base['VALORCATASTRAL']>tabla['LIM_INF'][i])), 'FACTOR'] = tabla['FACTOR'][i]
    base['PREDIAL']=base['CUOTA_FIJA'] + (base['VALORCATASTRAL'] - base['LIM_INF'])*base['FACTOR']











