import easyocr
from pdf2image import convert_from_path
from PyPDF2 import PdfReader, PdfWriter
import pandas as pd 
import numpy as np
import sqlite3 

reader=easyocr.Reader(["es"],gpu=True)
#Funciones

def PDF2DF(images, file_name: str="", route: str="", name: str="", name_table:str="", hoja: int = 0,to_sql=True) -> None:
    """
        (Function)
        Esta función es para convertir un PDF a imágenes y después guarda a un sql.

        (Parameters)
        images: La imagén procesada de pdf2images que utiliza PILLOW por detrás.
        file_name: El nombre del archivo pdf.
        route: El nombre de la ruta donde hay que guardar la base de datos.
        name: El nombre dela base de datos a utilizar.
        name_table: El nombre de la tabla de la base de datos.
        hoja: El número de la hoja que esta siendo procesado.
        to_sql: inicializado en True para almacenar en sql la salida, False para solo obtener el proceso
        (Returns)
        None
    """
    img1=images.convert("L")
    threshold = 180
    im = img1.point(lambda p: p < threshold and 255)
    image = np.asarray(im)
    result=reader.readtext(image)
    res=pd.DataFrame(result)
    del img1,images,im,image
    res["p1x"]=list(map(lambda res:res[0][0][0],result))
    res["p1y"]=list(map(lambda res:res[0][0][1],result))
    x=res["p1x"]
    y=res["p1y"]
    res["x"]=(x-np.min(x))/(np.max(x)- np.min(x))
    res["y"]=(y-np.min(y))/(np.max(y)- np.min(y))
    res["origen"]=file_name.replace("\\","/").split("/")[-1]
   
    res=res[[1,"x","y","origen"]]
    res["hoja"]=hoja
    if to_sql:
        conn = sqlite3.connect(route + '/' + name) ## abrir una sesion sqlite3
        res.to_sql(name_table, conn,if_exists='append', index = False) ##exportar la base a sqlite3
        conn.close()
    else:
        return res


def run_script_PDF2DF(files: list, route: str, namedb: str, name_table: str, pages_end: int, pages_start: int =0) -> None:
  """
    (Function)
      Esta función es para correr de manera iterativa la función PDF2DF.
    
    (Parameters)
    files: una lista de archivos a utilizar.
    route: ruta a donde guardar los archivos.
    namedb: nombre de la base de datos.
    name_table: nombre de la tabla de la base de datos.
    pages_end: Fin de las páginas a procesar.
    pages_star: Inicio de las páginas a procesar.

    (Returns)
     None
  """
  for file_name in files:
    images = convert_from_path(file_name,175)[pages_start:pages_end]
    for i,image in enumerate(images):
      PDF2DF(image,file_name,route,namedb,name_table,i)
    
def Rotar_pagina_right (nombre_archivo:str, nombre_salida:str, grados=90, pagina=None ):
    '''
    (Function)
        Esta funciÃ³n rota todas las paginas de un archivo la cantidad de grados especificados a
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
        Tener estas librerias previo a su ejecuciÃ³n:
        !pip install PyPDF2
    (Example)
        nombre_archivo = '/content/drive/MyDrive/Equipo_Agua/Adjudicaciones/salidas/Compranet/Pdfs_solo_acta_fallo/SFP/LO-815074882-E1-2020-ACTA DE NOTIFICACION FALLO LP-060.pdf'
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