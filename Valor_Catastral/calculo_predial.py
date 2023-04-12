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




def Categorizar_por_intervalos(x:int, li:int, ls:int):
    '''
(Function)
    Esta funcion retorna un 1 si el valor ingresado esta en el intervalo definido
    de  lo contrario regresa un 0, en caso de ser un tipo de dato no valido
    returna un 9
(Parameters)
    - x: Valor que se desea clasificar
    - li: Limite inferior
    - ls: Limite superior
(Author)
    Hector Limon
    '''
    if isinstance(x,int) or isinstance(x,float):
        if x<=ls and x>= li:
            return 1
        else:
            return 0
    else:
        raise KeyError(f'{x} no tiene el formato numerico adecuado') 






