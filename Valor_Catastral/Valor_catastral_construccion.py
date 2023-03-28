# VALOR CATASTRAL DE CONSTRUCCION

def factor_edad(ac, cd):
    '''
    (Function)  
        Esta funcion calcula el factor de edad de construccipon (FEC) para cuestiones del calculo del valor catastral
    (Parameters)
        ac: Años transcurridos desde la construcción o desde la última remodelación en enteros
        cd: Coeficiente de demérito anual obtenido de la tabla con el mismo nombre
        '''
    if isinstance(ac, int) or (isinstance(cd,float) or isinstance(cd,int)):
        return round(1-(ac*cd),5)
    else:
        prin('Tipos de datos de no valido, verifique que sean numericos (int or float)')

def factor_grado_conservacion(g):
    '''
    (Function)  
        Esta funcion calcula el factor de conservación de construcción (FGC) para cuestiones del calculo del valor catastral
    (Parameters)
        g: Cadena de texto del grado de conservación entre Bueno, Normal, Regular, Malo y Ruinoso
        '''
    grados = {'Bueno':1, 
              'Normal':0.90, 
              'Regular':0.75,
              'Malo':0.40,
              'Ruinoso':0.08}
    if isinstance(g, str):
        return round(grados[g],5)
    else:
        print('Tipo de dato no valido, verifique que sea una cadena de texto')

def factor_numero_niveles(nn):
    '''
    (Function)  
        Esta funcion calcula el factor de numero de niveles (FNN) para cuestiones del calculo del valor catastral
    (Parameters)
        nn: Número de niveles de la construcción expresado en enteros
        '''
    if isinstance(nn, int):
        return round(1+(nn-2)*0.002,5)
    else:
        print('Tipos de datos de no valido, verifique que sean numericos (int or float)')