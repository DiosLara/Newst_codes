# VALOR CATASTRAL DEL TERENO
def factor_frente(x,p):
    '''
    (Function)  
        Esta funcion calcula el factor de frente (FFe) para cuestiones del calculo del valor catastral
    (Parameters)
        x: Longitud del frente expresada en metros lineales
        p: cadena de texto del tipo de posición del terreno
    '''
    if isinstance(x,float) or isinstance(x, int):
        if p=='Interior' or x>=3.5 : return 1
        else: return round(x/3.5,5)
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
    if (isinstance(f,float) or isinstance(f, int)) and (isinstance(fb,float) or isinstance(fb, int)):
        if p in ['Manzanero','Cabecero','Frentes_no_contiguos']:
            f=f/2
        if p=='Interior' or f<=fb: return 1        
        else: return round(0.6 + ((fb/f)*0.4),5)  
    else:
        print('Tipos de datos de no validos, verifique que sean numericos (int or float)')

def factor_irregularidad(s,ai,r,p):
    '''
    (Function)  
        Esta funcion calcula el factor de irregularidad (FI) para cuestiones del calculo del valor catastral
    (Parameters)
        s: Superficie del terreno expresada en metros cuadrados
        ai: Área inscrita definida como la mayor superficie del terreno cubierta por como máximo dos rectángulos
        r: Indicador de regularidad del poligono (0 para no regular, 1 para regular)
        p: cadena de texto del tipo de posición del terreno
        '''
    if (isinstance(ai,float) or isinstance(ai, int)) and (isinstance(s,float) or isinstance(s, int)):
        if r==0 or p!='Interior': return round(0.5 + ((ai/2)/s),5)
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
    if (isinstance(s,float) or isinstance(s, int)) and (isinstance(ab,float) or isinstance(ab, int)):
        if s<=ab or p=='Interior': return 1
        else: return round(0.7 + ((ab/s)*0.3),5)         
    else:
        print('Tipos de datos de no validos, verifique que sean numericos (int or float)')

def factor_topografia(h,f,i,p):
    '''
    (Function)  
        Esta funcion calcula el factor de topografía (FT) para cuestiones del calculo del valor catastral
    (Parameters)
        h: Longitud de la altura del desnivel expresada en metros lineales
        f: Longitud del fondo expresada en metros lineales
        i: indicador de inclinación (0 a nivel de la banqueta, 1 inclinado)
        p: cadena de texto del tipo de posición del terreno
        '''
    if (isinstance(f,float) or isinstance(f, int)) and (isinstance(h,float) or isinstance(h, int)):
        if i==0 or p=='Interior': return 1
        else: return round(1 - ((h/2)/f),5)  
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
        if p=='Interior': return 1
        else: return round(0.5 + ((aa/2)/s),5)  
    else:
        print('Tipos de datos de no validos, verifique que sean numericos (int or float)')

