import cv2
import numpy as np
from tkinter import filedialog
from tkinter import *
import shutil
import rasterio
import tqdm
import pandas as pd 
import rasterio
import os
import rasterio.mask



btn_down = False

def get_points(im):
    # Set up data to send to mouse handler
    data = {}
    data['im'] = im.copy()
    data['lines'] = []

    # Set the callback function for any mouse event
    cv2.imshow("Image", im)
    cv2.setMouseCallback("Image", mouse_handler, data)
    cv2.waitKey(0)

    # Convert array to np.array in shape n,2,2
    points = np.uint16(data['lines'])

    return points, data['im']

def mouse_handler(event, x, y, flags, data):
    global btn_down

    if event == cv2.EVENT_LBUTTONUP and btn_down:
        #if you release the button, finish the line
        btn_down = False
        data['lines'][0].append((x, y)) #append the seconf point
        cv2.circle(data['im'], (x, y), 3, (0, 0, 255),5)
        cv2.line(data['im'], data['lines'][0][0], data['lines'][0][1], (0,0,255), 2)
        cv2.rectangle(data['im'], data['lines'][0][0], data['lines'][0][1], (0,0,255), 2)
        cv2.imshow("Image", data['im'])

    elif event == cv2.EVENT_MOUSEMOVE and btn_down:
        #thi is just for a ine visualization
        image = data['im'].copy()
        cv2.line(image, data['lines'][0][0], (x, y), (0,0,0), 1)
        cv2.rectangle(image, data['lines'][0][0], (x, y), (0,0,255), 2)
        cv2.imshow("Image", image)

    elif event == cv2.EVENT_LBUTTONDOWN: #and len(data['lines']) < 10:
        btn_down = True
        data['lines'].insert(0,[(x, y)]) #prepend the point
        cv2.circle(data['im'], (x, y), 3, (0, 0, 255), 5, 16)
        cv2.imshow("Image", data['im'])
        
        
# Running the code
# mode="train"
# filename="E:\geoshapes\TOLUCA\imagenes1\zona_4_0_0.png"
# path_init = r"Users\ASUS\Desktop\zerodayssoftware\deteccion de terrenos"
# path_save = "C:\Users\52551\Downloads"

def etiquetar_imagenes(
                        path_init:str,
                        path_save:str
                        ):
    '''
    (Function)
        Funcion que ayuda a etiquetar imagenes en cajas, con la finalidad de identificar objetos
    (Parameters)
        - path_init: Ruta de acceso al directorio (carpeta) donde tiene las imagenes
        - path_save: Ruta de acceso al directorio (carpeta) donde se almacenara las imagenes.
        
    (Return)
        Archivo txt con numero de etiqueta y coordenadas referenciales al objeto señalado
    
    '''
    root=Tk()
    root.filename =  filedialog.askopenfilename(initialdir = path_init,
                                                title = "Select file",
                                                filetypes = (("image",["*.png","*.jpg","*.jpeg","*.tiff","*.txt"]),("all files","*.*")),
                                                multiple=True)

    filenames=root.filename
    filenames
    root.destroy()
    for filename in filenames:
        img = cv2.imread(filename, 1)
        h,w,_=img.shape
        print(h,w)
        if w>h:
            img=cv2.resize(img,(int((h*1000)/w),1000))
        else:
            img=cv2.resize(img,(1000,int(w*1000/h)))
        h,w,_=img.shape
        pts, final_image = get_points(img)
        cv2.imshow('Image', final_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        coordenadas=[]
        yolo_F=[]   
        for rec in pts:
            x1=np.min([rec[0][0],rec[1][0]])
            y1=np.min([rec[0][1],rec[1][1]])
            x2=np.max([rec[0][0],rec[1][0]])
            y2=np.max([rec[0][1],rec[1][1]])
            coordenadas.append('{"label": "casas", "coordinates": {"x":'+str(float(x1*2))+', "y": '+str(float(y1*2))+', "width": '+str((x2*2)-(x1*2))+', "height": '+str((y2*2)-(y1*2))+'}}')
            v1=(x1+(x2-x1)/2)/w
            v2=(y1+(y2-y1)/2)/h
            v3=(x2-x1)/w
            v4=(y2-y1)/h
            yolo_F.append(" ".join(["0",str(v1),str(v2),str(v3),str(v4)]))
        yolo_F="\n".join(yolo_F)
        coordenadas=", ".join(coordenadas)
        cve=filename.replace("\\","/").split("/")[-1].split('.')[0]
        print(cve)
        # with open(str(cve)+".json","w+") as file:
        #         file.write('[{"image": "'+str(cve)+'.png", "verified": false, "annotations":['+coordenadas+"]}]")
        #         file.close()
        #         print("archivo creado: "+cve+".json")
        cve=cve.split("/")[-1]
        with open(path_save+"/"+str(cve)+".txt","w+") as file:
                file.write(yolo_F)
                file.close()
                print(f"archivo creado: {path_save}/+{str(cve)}.txt")

def particionar_imagen (fp:str, parametro:int, ruta_salida:str, nombre_salida:str):
    '''
    (Function)
        Esta funciona toma el archivo "fp" y lo particiona en n-elementos (imagenes) para tener fragmentos de la original
        con un zoom siendo cuadradas
    (Paramateres)
        - fp: Ruta de acceso al archivo .tif (Obligatoriamente) 
        - parametro: cantidad de pixeles que tomara para formar un cuadrado (fragmento)
    '''
    img = rasterio.open(fp)
    print('Imagen leida.')
    array = img.read()
    four_images=[array[2],array[1],array[0],array[3]]
    stacked_images = np.stack(four_images, axis=-1)
    H,W,D=stacked_images.shape
    z=0
    for i in tqdm.tqdm(range(int(H/parametro))):
        for j in range(int(W/parametro)):
            array1=stacked_images[parametro*i:parametro*(i+1),parametro*j:parametro*(j+1)]
            cv2.imwrite(ruta_salida+"/"+nombre_salida+str(i)+"_"+str(j)+".png",array1)



def Generar_txt(vector:list, save_folder:str='/content/drive/MyDrive/Equipo_Agua/Geo/Data/Carpeta_prueba/Hector/txt'):
    '''
    (Function)
        Esta funcion genera archivos txt, que es lo que harias con la app "labelImg", que contiene la etiqueta y coordenadas
        correspondientemente. 
    (Parameters)
        vector: Esta variable sale despues de hacer la prediccion, utilizando el modelo de entrenado con la arquitectura
                Yolov7.
        save_folder: Ruta al directorio, donde desea guardar los txt generados.
    (Example)
        Antes de llamar a la funcion es necesario correr el siguiente codigo en una celda, ya que el codigo
        genera una variable que se llama vector, no se ve explicitamente la declaracion, pero al correr el codigo
        se crea (tras bambalinas)
        
        %run detect_1.py --device 0 --weights best_Fer.pt --conf 0.1 --img-size 256 --source ruta/carpeta/con/imagenes/para/deteccion
        
        luego simplemente llama a la funcion Generar_txt(), si quiere puede revisar en una celda que existe la variable vector
        que es una lista.
    
    '''
    vector=[str(x).replace("tensor(","").replace("=","").replace(", device","").replace("[","").replace("'cuda:0'","").replace("]","").replace(".)","").replace(")","").replace(" ","").replace("(","").replace("']","").replace("'","").strip().split(",") for x in vector]
    df_vector = pd.DataFrame(vector)
    df_vector['Etiqueta'] = df_vector[0].map(lambda x: str(x))
    df_vector.rename(columns={6:'Nombre'}, inplace=True)
    for nombre in tqdm.tqdm(df_vector['Nombre'].unique()):
        df_aux = df_vector[df_vector['Nombre'] == nombre][['Etiqueta', 1,2,3,4]]
        df_aux.reset_index(drop=True, inplace=True)
        try:
            os.mkdir(save_folder)
        except:
            pass
        with open(save_folder + '/' + nombre + '.txt', 'w') as archivo:
            for i in df_aux.index:
                linea = df_aux.iloc[i].astype(str).values
                archivo.writelines(' '.join(linea) + '\n')

def generar_mosaico_v2(raster:str,output_path:str,mode:bool=False, dim:int=1024 ):
    '''
    (Function)
        Esta funcion particiona un archivo .tif en multiples, con la intencion de tener fragmentos del mismo.
    (Parameters)
        - raster: Ruta del archivo a particionar
        - output_path: Ruta a la carpeta donde se guardaran los fragmentos
        - mode: Si es True genera archivos .tif de lo contrario genera archivos .png (Por default False)
        - dim: Numero de pixeles a considerar por particion
    (Example)
    '''
    from osgeo import gdal
    
    gdal_interpeter = gdal.Open(raster)
    width = gdal_interpeter.RasterXSize
    height = gdal_interpeter.RasterYSize
    coordenadas_gdal = gdal_interpeter.GetGeoTransform()
    minx = coordenadas_gdal[0]
    miny = coordenadas_gdal[3] + width*coordenadas_gdal[4] + height*coordenadas_gdal[5] 
    maxx = coordenadas_gdal[0] + width*coordenadas_gdal[1] + height*coordenadas_gdal[2]
    maxy = coordenadas_gdal[3] 
    minx,maxx,miny,maxy,"W",maxx-minx,"H",maxy-miny
    src_raster_path = raster
    src=rasterio.open(src_raster_path)
    H,W=src.shape
    alto=int(np.floor(H/dim))
    ancho=int(np.floor(W/dim))
    for j in tqdm.tqdm(2,range(ancho)):#ancho
        for i in (range(alto)):#alto
            # j=1
            label=raster.replace("\\","/").split("/")[-1][:-4]+"_"
            nameimg=label.lower()+str(i)+"_"+str(j)
            cuadro=[]
            for k in range(2): 
                for l in range(2):
                    cuadro.append((minx+(maxx-minx)/ancho*(j+k),
                                maxy-(maxy-miny)/alto*(i+l),
                                0.0))
            cuadro=[cuadro[0],cuadro[1],cuadro[3],cuadro[2],cuadro[0]]
            shapes=[{"type":'Polygon','coordinates':[cuadro]}]
            vector=[]
            if mode==False:
                array, out_transform = rasterio.mask.mask(src, shapes, crop=True)
                array=array.copy()
                four_images=[array[2],array[1],array[0]]
                stacked_images = np.stack(four_images, axis=-1)
                imagen_n=0
                imagen_n=stacked_images.copy()
                cv2.imwrite(output_path+"/"+nameimg+'.png',imagen_n)
            if mode==True:
                out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True) # setting all pixels outside of the feature zone to zero
                out_meta = src.meta

                out_meta.update({"driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform})

                output_file = output_path+"/"+nameimg+".tif"

                with rasterio.open(output_file, "w", **out_meta) as dest:
                    dest.write(out_image)