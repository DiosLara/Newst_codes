# VALOR CATASTRAL DE CONSTRUCCION
import pandas as pd
import random



def get_factor_edad_va(x=''):
    return  round(random.uniform(0.6,1),5)

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

def factor_numero_niveles(nn,g):
    '''
    (Function)  
        Esta funcion calcula el factor de numero de niveles (FNN) para cuestiones del calculo del valor catastral
    (Parameters)
        nn: Número de niveles de la construcción expresado en enteros
        g: Cadena de texto del grado de conservación entre Bueno, Normal, Regular, Malo y Ruinoso
        '''
    if isinstance(nn, int):
        if nn==1 or nn==2 or g=='Ruinoso': return 1
        else: return round(1+(nn-2)*0.002,5)
    else:
        print('Tipos de datos de no valido, verifique que sean numericos (int or float)')
        
# def factor_edad(ac,t,g,cd=pd.read_csv('./Data/Factor_de_demerito_naucalpan.csv')):#vu):
#     '''
#     (Function)  
#         Esta funcion calcula el factor de edad de construccipon (FEC) para cuestiones del calculo del valor catastral
#     (Parameters)
#         ac: Años transcurridos desde la construcción o desde la última remodelación en enteros
#         cd: Tabla correspondiente al factor de demérito anual vigente
#         t: cadena de texto de la tipología de construcción
#         g: Cadena de texto del grado de conservación entre Bueno, Normal, Regular, Malo y Ruinoso
#         '''
#     if isinstance(ac, int) and ((isinstance(cd,float) or isinstance(cd,int))):
#         #if ac>vu: return: 0.6
#         if g=='Ruinoso': return 1
#         else: return round(1-(ac*cd[cd['Tipo']==t]['CDA']),5)
#     else:
#         print('Tipos de datos de no valido, verifique que sean numericos (int or float)')
