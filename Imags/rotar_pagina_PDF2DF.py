'''
Rotar Página PDF
por Héctor Limón

Recomendable correrlo en Google Collab con GPU.
Tener estas librerias previo a su ejecución:
!pip install PyPDF2

Ejemplo de llamada al programa:

#Main
nombre_archivo = ''
nombre_salida = ''
Rotar_pagina_right (nombre_archivo, nombre_salida, grados=270, pagina=4)
'''

from os import listdir
from PyPDF2 import PdfReader, PdfWriter


def Rotar_pagina_right (nombre_archivo:str, nombre_salida:str, grados=90, pagina=None ):
    '''
    (Function)
        Esta función rota todas las paginas de un archivo la cantidad de grados especificados a
        la derecha, creando un segundo archivo, con el especificado en nombre_salida, en caso de
        ser  necesario, puede especificarse la pagina.
    (Parameters)
        - nombre_archvio: La ruta del archivo del cual se desea rotar la(s) pagina(s).
        - nombre_salida: La ruta y el nombre del archivo de salida (output), debe agregar
                         ".pdf" forzosamente.
        - grados: [int] Cantidad de grados a girar la pagina a la derecha.
        - pagina: [int] Por default voltea todas las paginas, si desea solo 1 pagina, debe especificar
                        aqui que numero, la primer pagina se asigna al numero 1.
    (Returns)
        None
    (Notes)
        Recomendable correrlo en Google Collab con GPU.
        Tener estas librerias previo a su ejecución:
        !pip install PyPDF2
    (Example)
        nombre_archivo = ''
        nombre_salida = 'LO2.pdf'
        Rotar_pagina_right (nombre_archivo, nombre_salida, grados=270, pagina=4)

    '''
    reader = PdfReader(nombre_archivo)
    writer = PdfWriter()
    if pagina:
        i = 1
        for page in reader.pages:
            if i == pagina:
                page.rotate_clockwise(grados)
            writer.add_page(page)
            i += 1
        
    else:
        for page in reader.pages:
            # Rota a la derecha
            page.rotate_clockwise(grados)
            writer.add_page(page)
    with open(nombre_salida , "wb") as pdf_out:
        writer.write(pdf_out)