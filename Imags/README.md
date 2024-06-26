# Procesamiento de imagen2022

Códigos de para extracción de información de los archivos de adjudicaciones.


**REQUISITOS**

!pip install pdf2image

!pip install PyPDF2

!pip install easyocr

## Descarga de archivos.

Lo primero que se llevo a cabo fue la descarga de archivos. Esta descarga se hizo mediante web-scrapping. Esto con la finalidad de tenerlos en nuestras máquinas para posteriormente procesarlos.
La descarga se hizo con ayuda de la extension de [Batch Link Downloader](https://chrome.google.com/webstore/detail/batch-link-downloader/aiahkbnnpafepcgnhhecilboebmmolnn) de archivos de google Chrome, ya que la cantidad de archivos era muy grande para hacerlo manual.

![alt text](https://lh3.googleusercontent.com/C57vV21wg_LU_oNc_IY1wfti-r2b5zqkEEQltEWOb4tNNjDD0pRY92sKGhNWkIv3WTiKV2I107UdfKw2SAWJ_EcA_g=w640-h400-e365-rj-sc0x00ffffff)
Ejemplo del gestor de descarga Batch Link Downloader

## Clasificación de archivos

La clasificación de archivos de varios se hizo manualmente, ya que la era la única manera de poder clasificarlos, pero otra parte se hizo con código pues se podía reconocer un patrón con dicho nombre.

Aquí un ejemplo de cómo se hizo aplicando código, este proceso necesita ser iterativo manualmente.

```python
import shutil
import glob
import os


def move_pull_data(route,key_file,folder):
      for file_name in glob.glob(route+key_file):
        destination = route+folder+file_name[len(route):]
        shutil.move(file_name,destination)

os.mkdir('/content/drive/MyDrive/Pdfs_solo_acta_fallo/SCEM')
move_pull_data('/content/drive/MyDrive/Pdfs_solo_acta_fallo/','SCEM*.pdf','SCEM/')
```

## Implementación de OCR
Una vez descargados los archivos, es necesario pasarlos a una Base SQL para poderles extraer la información importante de manera más sencilla y en masa.
Para esto usamos la función **PDF2DF** del módulo de **PDF2SQL**  esto lo hacemos:

```python
from PDF2SQL import PDF2DF
```
Luego, se busca obtener la base SQL con los archivos ya procesados con OCR, ya que la función que se manda a llamar es la que corre de manera iterativa el código de PDF2SQL.

```python
path="E:\pdf\*/"                        # Ruta carpeta 
filenames=glob.glob(path+"*.pdf")       # Lista de archivos
PDF2SQL.run_script_PDF2DF(filenames,path,"Archivospdf","pdf")
```
Debe saber que esta función tiene otros parametros que pueden ayudar a mejorar la calidad de la base, siempre puede leer la documentación de la función.


En este punto se habrán procesado tantos archivos como fueron necesarios; Posicionándolos en múltiples carpetas, según sea el caso.

## Extracción de la información.
Ahora que ya tenemos una base SQL 

![alt img](https://i.postimg.cc/xdDvZrFz/Whats-App-Image-2022-12-14-at-1-05-04-PM.jpg)

ejemplo de base SQL..


Una vez que se obtiene la base SQL correspondiente al Tipo de archivo que se procesó, para poder acceder a la base e iniciar la extracción de datos se hace la siguiente conexión:

```python
conect=sqlite3.connect("E:/Archivospdf")
base=pd.read_sql_query("SELECT * FROM  pdf",con=conect)
conect.close()
```
Es necesario la extracción de la información, para ello es necesario que importemos el módulo **Procesamiento_Imagenes** el cual contiene diversas funciones que son algoritmos especializados según el tipo de archivo, los cuales son:
    1. Archivos de tipo FAM
    2. Archivos de tipo FP
    3. Archivos de tipo JC 
    4. Archivos de tipo PAD
    5. Archivos de tipo Compranet(Contratos)

De ellos, se describe su aplicación particularmente en el archivo ***main.ipynb***

Lamentablemente la presición de dichos algoritmos no es del 100% lo que nos llevó a generar alternativas para la extracción, el módulo **Acta_Fallo_JC** se diseñó para archivos de Compranet que son del tipo Junta de caminos pero son Actas de fallo (Fue necesario revisar actas de fallo, para los links que no tenían contratos). Este el único módulo que tiene una clase, la cual es bastante conveniente importar como:

```python
from Acta_Fallo_JC import Junta_Caminos_compranet
```

La clase esta diseñada para seguir aprendiendo, por lo que es necesario estar alimentando (Cuando se necesite) constantemente con nuevas *keywords* para sus archivos que lo necesiten.

```
