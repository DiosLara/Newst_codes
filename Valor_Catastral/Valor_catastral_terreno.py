# VALOR CATASTRAL DEL TERENO
import random 


def factor_frente(x,p):
    '''
    (Function)  
        Esta funcion calcula el factor de frente (FFe) para cuestiones del calculo del valor catastral
    (Parameters)
        x: Longitud del frente expresada en metros lineales
        p: cadena de texto del tipo de posición del terreno
    '''
    if p=='Interior':
        return 1
    if isinstance(x,float) or isinstance(x, int):
        if x>=3.5 : return 1
        else: 
            valor = round(x/3.5,5)
            if valor <= 0.5:return 0.5
            else: return valor
    else:
        print('Tipo de dato de no valido, verifique que sea numerico (int or float)')

def factor_fondo(f, fb, p):
    '''
    (Function)  
        Esta funcion calcula el factor de fondo (FFo) para cuestiones del calculo del valor catastral
    (Parameters)
        f: Longitud del fondo expresada en metros lineales
        fb: Fondo base determinado según la tabla de valores unitarios
        p: cadena de texto del tipo de posición del terreno
    '''
    if p=='Interior' : return 1
    if (isinstance(f,float) or isinstance(f, int)) and (isinstance(fb,float) or isinstance(fb, int)):
        if p in ['Manzanero','Cabecero','Frentes_no_contiguos']:
            f=f/2
        if  f<=fb: return 1        
        else: 
            valor = round(0.6 + ((fb/f)*0.4),5)  
            if valor <= 0.6: return 0.6
            else: return valor
    else:
        print('Tipos de datos de no validos, verifique que sean numericos (int or float)')

def factor_irregularidad(s,ai,p,r=''):
    '''
    (Function)  
        Esta funcion calcula el factor de irregularidad (FI) para cuestiones del calculo del valor catastral
    (Parameters)
        s: Superficie del terreno expresada en metros cuadrados
        ai: Área inscrita definida como la mayor superficie del terreno cubierta por como máximo dos rectángulos
        r: Indicador de regularidad del poligono (0 para no regular, 1 para regular)
        p: cadena de texto del tipo de posición del terreno
        '''
    if p=='Interior': return 1
    if (isinstance(ai,float) or isinstance(ai, int)) and (isinstance(s,float) or isinstance(s, int)):
        if  p!='Interior': 
            valor = round(0.5 + ((ai/2)/s),5) 
            if valor < 0.5: return 0.5
            else: return valor
        else: return 1
    else:
        print('Tipos de datos de no valido, verifique que sean numericos (int or float)')

def factor_area(s,ab,p):
    '''
    (Function)  
        Esta funcion calcula el factor de área (FA) para cuestiones del calculo del valor catastral
    (Parameters)
        s: Superficie del terreno expresada en metros cuadrados
        ab: Área base determinada según la tabla de valores unitarios
        p: cadena de texto del tipo de posición del terreno
    '''
    if p=='Interior' : return 1 
    if (isinstance(s,float) or isinstance(s, int)) and (isinstance(ab,float) or isinstance(ab, int)):
        if s<=ab or p=='Interior': return 1
        else: 
            valor = round(0.7 + ((ab/s)*0.3),5)         
            if valor < 0.7: return 0.7
            else: return valor
    else:
        print('Tipos de datos de no validos, verifique que sean numericos (int or float)')

def factor_topografia(h,f,p,i=''):
    '''
    (Function)  
        Esta funcion calcula el factor de topografía (FT) para cuestiones del calculo del valor catastral
    (Parameters)
        h: Longitud de la altura 
        f: Longitud del fondo expresada en metros lineales
        i: indicador de inclinación (0 a nivel de la banqueta, 1 inclinado)
        p: cadena de texto del tipo de posición del terreno
        '''
    if p=='Interior' : return 1
    if (isinstance(f,float) or isinstance(f, int)) and (isinstance(h,float) or isinstance(h, int)):
        if  p=='Interior': return 1   # i==0 or
        else: 
            if round(1 - ((h/2)/f),5) >= 0.5  : return round(1 - ((h/2)/f),5)
            else: return 0.5
    else:
        print('Tipos de datos de no validos, verifique que sean numericos (int or float)')

def factor_posicion(p:str):
    '''
    (Function)  
        Esta funcion calcula el factor de posicion (FP) para cuestiones del calculo del valor catastral
    (Parameters)
        p: cadena de texto del tipo de posición del terreno
        '''
    posiciones = {'Interior':0.5, 
              'Intermedio':1, 
              'Esquinero':1.1,
              'Frentes_no_contiguos':1.1,
              'Cabecero':1.2,
              'Manzanero':1.3}
    if isinstance(p, str):
        return round(posiciones[p],5)
    else:
        print('Tipo de dato no valido, verifique que sea una cadena de texto')

def factor_restriccion(s,aa,p):
    '''
    (Function)  
        Esta funcion calcula el factor de restriccion (FR) para cuestiones del calculo del valor catastral
    (Parameters)
        s: Superficie del terreno expresada en metros cuadrados
        aa: Área aprovechable del terreno expresada en metros cuadrados
        p: cadena de texto del tipo de posición del terreno
        '''
    if (isinstance(aa,float) or isinstance(aa, int)) and (isinstance(s,float) or isinstance(s, int)):
        if p.capitalize()=='Interior': return 1
        else: 
            valor = round(0.5 + ((aa/2)/s),5)  
            if valor < 0.5: return 0.5
            else: return valor
    else:
        print('Tipos de datos de no validos, verifique que sean numericos (int or float)')

def get_factor_posicion(x:int):
    '''
(Function)
    Esta funcion genera el factor de posicion en base a un preanalisis geoespacial, de manera que se identifican 
    los preidos que estan en el borde de una calle. Debido a la distribucion se generan variables aleatorias en base 
    al parametro de entrada, ya que si x es 1, determinamos que es un predio que esta en el contorno de la manzana,
    es decir puede ser un predio Intermedio, Esquinero, Frentes no contiguos, Cabecero o Manzanero. Pero si es 0, sencillamente
    decimos que es Interior. Para ver mas detalles de esto consule el Manual Catastral del Estado de Mexico en la pagina 79.
(Parameters)
    - x: Valor entero que pertenece a una categoria, si x es 1, determinamos que es un predio que esta en el contorno de la manzana,
    es decir puede ser un predio Intermedio, Esquinero, Frentes no contiguos, Cabecero o Manzanero. Pero si es 0, sencillamente
    decimos que es Interior
(Author)
    Hector Limon
    '''
    if x==1:
        va = random.random()
        # print(va)
        if va < 0.04:
            return 1.3
        elif va < 0.1:
            return 1.2
        elif va < 0.35:
            return 1.1
        else:
            return 1
    elif x == 0:
        return 0.5
    else:
        raise ValueError('El valor ingresado debe ser un numero entero')