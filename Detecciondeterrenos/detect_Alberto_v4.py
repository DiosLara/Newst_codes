# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 10:45:35 2023
@author: Alberto
"""
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import math
from shapely.geometry import Polygon
import cv2
import time
from pathlib import Path
import torch
import tqdm
import numpy as np
from models.experimental import attempt_load
from utils.datasets import LoadImages , letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh, set_logging
from utils.torch_utils import select_device, time_synchronized, TracedModel
import os
from osgeo import gdal
import rasterio
import geopandas as gpd
import rasterio.mask
from rasterio.windows import Window
import sys
from shapely.geometry import mapping
from scipy.ndimage import rotate as rotate_image
from shapely import geometry
import cv2
import torch,torchvision
import numpy as np
import torch.nn as nn
from PIL import Image
from torchvision import models
from torchsummary import summary
from shapely.ops import cascaded_union
from clasificacion_chinchetas import *
opt_img_size=256
class modelo():
    def __init__(self,weights=["yolov7.pt"]):
        """inicializa el modelo con los pesos"""
        self.device = torch.device(0 if torch.cuda.is_available() else "cpu")
        model = attempt_load(weights, map_location=self.device)
        model = TracedModel(model, self.device, opt_img_size)
        self.model= model
        
    def detect(self,opt_conf_thres,opt_source="train/images",display=False,imagen_s=np.array([1,1])):
        """Genera las deteccion de objetos por imagen precargada o por carpeta/archivo"""
        vector=[]
        opt_no_trace=False
        opt_iou_thres=0.45
        opt_save_conf=True
        opt_classes=None
        opt_agnostic_nms=False
        opt_augment=False
        opt_no_trace=False
        source,  imgsz, trace = opt_source, opt_img_size, not opt_no_trace
        set_logging()
        half = self.device.type != 'cpu'  # half precision only supported on CUDA
    # load FP32 model
        stride = int(self.model.stride.max())  # self.model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
        if half:
            self.model.half()  # to FP16
        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        # Set Dataloader
        if imagen_s.shape[0]==2:
            dataset = LoadImages(source, img_size=imgsz, stride=stride)
            t0 = time.time()
            for path, img, im0s, vid_cap in tqdm.tqdm(dataset):
                img = torch.from_numpy(img).to(self.device)
                img = img.half() if half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                t1 = time_synchronized()
                with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                    pred = self.model(img, augment=opt_augment)[0]
                t2 = time_synchronized()
                pred = non_max_suppression(pred, opt_conf_thres, opt_iou_thres, classes=opt_classes, agnostic=opt_agnostic_nms)
                t3 = time_synchronized()
                for i, det in enumerate(pred):  # detections per image
                    p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
                    p = Path(p)  # to Path
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    if len(det):
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                        for *xyxy, conf, cls in reversed(det):
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if opt_save_conf else (cls, *xywh)  # label format
                            vector.append(list(line)+[p.name[:-4]])  
                if display:
                    print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
        else:
            img0=imagen_s  
            img = letterbox(img0, imgsz, stride)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if half else img.float()
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            t1 = time_synchronized()
            with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                pred = self.model(img, augment=opt_augment)[0]
                # pred_o=pred
            t2 = time_synchronized()
            pred = non_max_suppression(pred, opt_conf_thres, opt_iou_thres, classes=opt_classes, agnostic=opt_agnostic_nms)
            t3 = time_synchronized()
            for i, det in enumerate(pred):  # detections per image
                s, im0 =  '', img0
#                 p = Path(p)  # to Path
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    for *xyxy, conf, cls in reversed(det):
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt_save_conf else (cls, *xywh)  # label format
                        vector.append(line)
        return vector
    
def correct_orientation(img_rgb,dim,pattern_path="pattern1.png"):
    """Determina el angulo donde existe la mayor deteccion de lineas rectas"""
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    template = cv2.imread(pattern_path,0)
    w, h = template.shape[::-1]
    template  = cv2.resize(template,(int(w/4),int(h/4)))
    image_ro=img_gray.copy()
    angulo=0
    an=[]
    le=0
    angulo_f=0
    imagen_final=img_rgb.copy()
    for i in range(-45,45,1):
        angulo=i
        M = cv2.getRotationMatrix2D((dim//2,dim//2), angulo, 1)
        image_ro = cv2.warpAffine(img_gray, M, (dim,dim))
        res = cv2.matchTemplate(image_ro,template,cv2.TM_CCOEFF_NORMED)
        threshold =.5
        loc = np.where( res >= threshold)
        com=len(loc[0])
        if com>0:
            an.append(angulo)
            if le<com:
                le=com
                angulo_f=angulo
                M = cv2.getRotationMatrix2D((dim//2,dim//2), angulo, 1)
                imagen_final = cv2.warpAffine(img_rgb, M, (dim,dim))
    return angulo_f,imagen_final

def verificacion(im):
    """filtro para determinar cuanta area verde existe en la imagen"""
    hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv,(12,30,0),(160,232,160))
    verde=int((np.sum(mask)/im.shape[0]**2/255)*100)
    return verde 

def vector2xy(vector,w,h,dim=700,nameimg="image",angle=0):
    """Transforma el vector de deteccion en coordendas porcentuales y enteras de la imagen"""
    s=[]
    for v in vector:
        str_v=(str(v).replace("tensor(","").replace("=","").replace(", device","").replace("[","").replace("'cuda:0'","").replace("]","").replace(".)","").replace(")","").replace("(","").replace("']","").replace("'","").strip().split(","))
#         h,w=dim,dim
        x1 = int( float(str_v[1]) * w )
        y1 = int( float(str_v[2]) * h )
        xw = int( float(str_v[3]) * w /2)
        yw = int( float(str_v[4]) * h /2)
        start_point_im = ((x1 - xw), (y1 - yw))
        end_point_im   = ((x1 + xw), (y1 + yw))
        start_point_100 = ((x1 - xw)/w, (y1 - yw)/h)
        end_point_100   = ((x1 + xw)/w, (y1 + yw)/h)
        area=xw*yw
        conf=str_v[5]
        try:
            nameimg=str_v[6]
        except:
            pass
        if str(str_v[0])=="0":
            tipo="casa"
        else:
            tipo="terreno"
        if int(xw)!=0 and int(yw)!=0:# and (xw/yw<=3.2 and yw/xw<=3.2):
            s.append([tipo,start_point_im,end_point_im,start_point_100,end_point_100,area,conf,nameimg," ".join(str_v[:5])])
    df_cache=pd.DataFrame(s,columns=["Tipo","start_point_im","end_point_im","start_point_100","end_point_100","area","conf","imagen","vector_o"])
    df_cache.drop_duplicates().reset_index(drop=True,inplace=True)
    return df_cache
    
def imshow_detect(df_cache,imagen_n,nameimg="image"):
    """muestra la imagen con las detecciones"""
    for i in range(len(df_cache)):
            if df_cache["Tipo"][i]=="casa":
                x,y=df_cache["start_point_im"][i]
                cv2.rectangle(imagen_n,df_cache["start_point_im"][i],df_cache["end_point_im"][i],(0,0,255),2)
#                 cv2.putText(imagen_n, str(df_cache["conf"][i]), (x+50,y+50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
            else:
                x,y=df_cache["start_point_im"][i]
                cv2.rectangle(imagen_n,df_cache["start_point_im"][i],df_cache["end_point_im"][i],(0,255,0),2)
#                 cv2.putText(imagen_n, str(int(float(df_cache["conf"][i])*100)/100),(x+50,y+50) , cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    cv2.namedWindow(nameimg, cv2.WINDOW_NORMAL)
    cv2.imshow(nameimg,imagen_n)
    cv2.waitKey()
    cv2.destroyAllWindows()

def rotacion_detect(startpoint,endpoint,angle,proyecciones,w,h,dim):
    """Rotata el cuadro detectado en el sistema de coordenadas inicial"""
    point1=np.min((proyecciones,proyecciones),axis=1)[0]
    min_y,min_x=point1[0],point1[1]
    point2=np.max((proyecciones,proyecciones),axis=1)[0]
    max_y,max_x=point2[0],point2[1]
    min_y,min_x,max_y,max_x,proyecciones
    tipos=["casa","terreno"]
    y1,x1=startpoint
    y2,x2=endpoint
    x1,y1=(x1)*2-1,(y1)*2-1
    x2,y2=(x2)*2-1,(y2)*2-1
    angle=angle*math.pi/180
    x1,y1=x1*(w/dim),y1*(h/dim)
    x2,y2=x2*(w/dim),y2*(h/dim)
    #x_p, y_p son los puntos de un rectangulo en el orden inverso al manecillas del reloj
    x1p=max_x-((x1*math.cos(angle)-y1*math.sin(angle)+1)/2)*(max_x-min_x)
    y1p=min_y+((x1*math.sin(angle)+y1*math.cos(angle)+1)/2)*(max_y-min_y)
    x2p=max_x-((x2*math.cos(angle)-y1*math.sin(angle)+1)/2)*(max_x-min_x)
    y2p=min_y+((x2*math.sin(angle)+y1*math.cos(angle)+1)/2)*(max_y-min_y)
    x3p=max_x-((x2*math.cos(angle)-y2*math.sin(angle)+1)/2)*(max_x-min_x)
    y3p=min_y+((x2*math.sin(angle)+y2*math.cos(angle)+1)/2)*(max_y-min_y)
    x4p=max_x-((x1*math.cos(angle)-y2*math.sin(angle)+1)/2)*(max_x-min_x)
    y4p=min_y+((x1*math.sin(angle)+y2*math.cos(angle)+1)/2)*(max_y-min_y)
    return Polygon(((y1p,x1p),(y2p,x2p),(y3p,x3p),(y4p,x4p),(y1p,x1p)))

def map_d(x, in_min, in_max, out_min, out_max):
    """Genera una interpolacion para pasar de un rango a otro"""
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def postproceso(Modelo,model_class,casas,conf_casas,clase_casas,                
                terreno,conf_terreno,clase_terreno,
                clase_chinchetas,umbrales,sizes,
                raster,ancho,alto,
                dim,minx,maxx,miny,maxy,shape,angulo_get=0,opt_conf_thres=0.05,
                imshow=False,imsave=False,path="",clasificar_casas:bool=True, cutline_factor:int=1,
                clasificar_chinchetas:bool=True, dict_gris={}):
    with rasterio.open(raster) as src:
        with tqdm.tqdm(total=alto*ancho*cutline_factor**2) as pbar:
            for j in range(ancho*cutline_factor):#ancho
                for i in range(alto*cutline_factor):#alto
                    generar=0
                    label=raster.replace('\\','/').split('/')[-1][:-4]+'_'
                    nameimg=label.lower()+str(i)+'_'+str(j)
                    cuadro=[]
                    for k in range(2):
                        for l in range(2):
                            cuadro.append((minx+(maxx-minx)/ancho*(j/cutline_factor+k),maxy-(maxy-miny)/alto*(i/cutline_factor+l)))
                    cuadro=[cuadro[0],cuadro[1],cuadro[3],cuadro[2],cuadro[0]]
                    cvees=[]
                    for punto in cuadro:
                        x=float(punto[0])
                        y=float(punto[1])
                        mini_df=shape[(shape[0]<=x)&(shape[2]>=x)&(shape[1]<=y)&(shape[3]>=y)]
                        if len(mini_df)>0:
                            generar=1
                            #cvees.append(mini_df["cve_cat"].values)### traer todas las cve_catastrales del punto sobre el raster ...Pendiente
#                             print(cvees)
                    if generar==1:
                        shapes=[{'type':'Polygon','coordinates':[cuadro]}]
                        vector=[]
                        array, out_transform = rasterio.mask.mask(src, shapes, crop=True)
                        if np.sum(array)<100:
                            pbar.update(1)
                            continue
                        four_images=[array[2],array[1],array[0]]
                        imagen_n = np.stack(four_images, axis=-1)
                        if imsave:
                            cv2.imwrite(path+"/"+nameimg+".png",imagen_n)
                            # print(path+"/"+nameimg+".png")
                        if angulo_get!=0:
                            angulo=-angulo_get
                        else:
                            angulo=correct_orientation(imagen_n,dim=dim)[0]
                        image_ro=imagen_n.copy()
                        image_ro=rotate_image(image_ro,angulo,reshape=True)
                        w=image_ro.shape[0]
                        h=image_ro.shape[1]
                        with torch.no_grad():
                            vector=Modelo.detect(opt_source="cache1.png",opt_conf_thres=opt_conf_thres,imagen_s=image_ro)
                        proyecciones=shapes[0].get('coordinates')[0][:-1]
                        # Vector de deteccion de yolov
                        df_cache=vector2xy(vector,w,h,dim=dim,nameimg=nameimg)
                        for cs_1 in (range(len(df_cache))):
    #                         x1,y1=df_cache.loc[cs_1,'start_point_im']
    #                         x2,y2=df_cache.loc[cs_1,'end_point_im']
    #                         df_aux=image_ro.copy()
    #                         df_aux=df_aux[y1:y2,x1:x2]
    #                         clase,imagen=model_class.predict_iamge(df_aux)
    #                         if np.sum(df_aux)>500:
    #                             if crear:
    #                                 name=r"C:\Users\ASUS\Inteligencia_Artificial\calsificador\salida/"+clase+"/"+str(datetime.datetime.now()).replace(" ","_").replace(":","_").replace(".","_").replace("-","_")+".png"
    #                                 cv2.imwrite(name,imagen)
                            x1,y1=df_cache.loc[cs_1,'start_point_im']
                            x2,y2=df_cache.loc[cs_1,'end_point_im']
                            df_aux=image_ro.copy()
                            df_aux=df_aux[y1:y2,x1:x2]
                            if clasificar_casas:
                                clase,imagen=model_class.predict_image(df_aux)
                            else:
                                clase = 'No_Aplica'
                            if clasificar_chinchetas:
                                print('entra')                                
                                clase_aux, n_aux, umbral_aux  = iter_umbral_fn(df_aux, dict_gris,n=100,
                                                                               salto_n=5,
                                                                               umbral=0.94,min_umbral=0.75)
                                cv2.imshow('df_aux',df_aux)
                                cv2.waitKey()
                                cv2.destroyAllWindows()
                                print(clase_aux)              
                                if clase_aux in ['a_verdes','establecimiento_google']:
                                    clase_aux = areasv_vs_estabgoogle(df_aux, dict_gris)                    
                                # clase_chinchetas.append(clase_aux)
                                clase_chinchetas += [clase_aux]
                                print(clase_chinchetas)
                                umbrales.append(umbral_aux)
                                sizes.append(n_aux)                            

                            if df_cache['Tipo'][cs_1]=='casa':
                                if np.sum(df_aux)>500:
                                    casas.append(rotacion_detect(df_cache.loc[cs_1,'start_point_100'], df_cache.loc[cs_1,'end_point_100'],-angulo,proyecciones,w,h,dim))
                                    conf_casas.append(df_cache.loc[cs_1,'conf'])
                                    clase_casas.append(clase)

                            else:
                                terreno.append(rotacion_detect(df_cache.loc[cs_1,'start_point_100'], df_cache.loc[cs_1,'end_point_100'],-angulo,proyecciones,w,h,dim))
                                conf_terreno.append(df_cache.loc[cs_1,'conf'])
                                clase_terreno.append(clase)
                        if imshow:
                            print(angulo)
                            imshow_detect(df_cache,image_ro)

                    pbar.update(1)
                    
def Parametro_raster(raster,metros=120):
    """Se obtienen las dimensiones del raster y del cuadro de corte"""
    gdal_interpeter = gdal.Open(raster)
    width = gdal_interpeter.RasterXSize
    height = gdal_interpeter.RasterYSize
    coordenadas_gdal = gdal_interpeter.GetGeoTransform()
    minx = coordenadas_gdal[0]
    miny = coordenadas_gdal[3] + width*coordenadas_gdal[4] + height*coordenadas_gdal[5] 
    maxx = coordenadas_gdal[0] + width*coordenadas_gdal[1] + height*coordenadas_gdal[2]
    maxy = coordenadas_gdal[3]
    src_raster_path = raster
    src=rasterio.open(src_raster_path)
    H,W=src.shape
    dim=int(np.ceil(map_d(minx+metros,minx,maxx,0,W)))
    alto=np.max([1,int(np.floor(H/dim))])
    ancho=np.max([1,int(np.floor(W/dim))])
    return alto,ancho,dim,src.crs,H,W,minx,maxx,miny,maxy


def shape_transform(shape):
    """Convierte el shape en dataframe de coordenadas que engloba el polygon del shape para delimitar el raster"""
    c=[]
    angulo_manzana=[]

    # for manzana in range(len(shape)):
    #     proyecciones1=mapping(shape['geometry'][manzana]).get('coordinates')
    #     angulos=[]
    #     d=[]
    #     poly=pd.DataFrame(proyecciones1[0])
    #     for point in range(1,len(poly)):
    #         d.append(((poly[1][point]-poly[1][point-1])**2+(poly[0][point]-poly[0][point-1])**2))
    #         angulos.append(math.atan(((poly[1][point]-poly[1][point-1])/(poly[0][point]-poly[0][point-1])))*180/math.pi)
    #     angulo_manzana.append(angulos[d.index(max(d))])
    shape["angulo_manzana"]=0
    shape["geometry"]=shape["geometry"].envelope
    for manzana in range(len(shape)):
        proyecciones1=mapping(shape['geometry'][manzana]).get('coordinates')
        proyecciones=proyecciones1[0]
        point1=np.min((proyecciones,proyecciones),axis=1)[0]
        min_y,min_x=point1[0],point1[1]
        point2=np.max((proyecciones,proyecciones),axis=1)[0]
        max_y,max_x=point2[0],point2[1]
        c.append(','.join([str(min_y),str(min_x),str(max_y),str(max_x)]))
    shape1=pd.DataFrame()
    shape1['points']=c
    shape1=shape1['points'].str.split(',',expand=True)
    shape1=shape1.astype({0:'float64',1:'float64',2:'float64',3:'float64'})
    #shape1["cve_cat"]=shape["cve_cat"]
    #shape1['zona'] = shape['zona']
    #shape1['manz'] = shape['manz']
    shape1["angulo_manzana"]=shape["angulo_manzana"]
    shape=shape1
    return shape
    
def ampliar_shape(shape,factor_ampliacion=2):
    """Amplifica el polygon de cada manzana con el fin de extrar imagenes sin perder informacion de la manzana"""
    shape["geometry"]=shape["geometry"]
    shape['centroid']=shape.centroid
    geometry=[]
    clase=[]
    for i,polygon in enumerate(shape['geometry']):
        try:
            point=mapping(shape['centroid'][i]).get('coordinates')
        except:
            continue
        x=point[0]
        y=point[1]
        go=[]
        coodinates=mapping(polygon).get('coordinates')[0]
        for a in coodinates:
            x1=a[0]
            y1=a[1]
            x2=x+(x1-x)*factor_ampliacion
            y2=y+(y1-y)*factor_ampliacion
            go.append((x2,y2))
        geometry.append(Polygon(go))
        clase.append(shape.loc[i,"clase_dete"])
    return gpd.GeoDataFrame({"clase_dete":clase},geometry=geometry)

# idx_to_class={0: 'area_verde', 1: 'carros', 2: 'casas', 3: 'en_construccion', 4: 'establecimiento', 5: 'multivivienda', 6: 'terreno_baldio'}
class alexnet():
    def __init__(self,weights,num_classes,idx_to_class):
        """inicializa el model, con los pesos entrenados"""
        alexnet=models.alexnet(pretrained=True)
        self.device = torch.device(0 if torch.cuda.is_available() else "cpu")
        checkpoint=torch.load(weights,map_location=self.device)
        # 
        alexnet.features[1]= nn.Hardtanh()
        alexnet.classifier[6] = nn.Linear(4096, 4096)
        alexnet.classifier.add_module("7",nn.Softplus())
        alexnet.classifier.add_module("8", nn.Linear(4096, 4096))
        alexnet.classifier.add_module("9",nn.Softplus())
        alexnet.classifier.add_module("10", nn.Linear(4096, 2048))
        alexnet.classifier.add_module("11", nn.Softplus())
        alexnet.classifier.add_module("12", nn.Linear(2048, num_classes))
        alexnet.classifier.add_module("13", nn.Softplus())
        alexnet.classifier.add_module("14",  nn.Softmax(dim = 1))
        # for param in alexnet.parameters():
        #     param.requires_grad = False
        # alexnet.classifier[6] = nn.Linear(4096, num_classes)
        # alexnet.classifier.add_module("7", nn.LogSoftmax(dim = 1))
        alexnet.load_state_dict(checkpoint['model_state_dict'])
        summary(alexnet, (3, 224, 224))
        self.model=alexnet
        self.idx_to_class=idx_to_class
    
    def predict_file(self,file,pad=True):
        """Genera prediccion sobre archivo"""
        # x = Image.open(file)
        # x = np.asarray(x)
        # x=np.stack([x[:,:,0],x[:,:,1],x[:,:,2]], axis=-1)
        x=cv2.imread(file)
        if pad:
            x=padding(x)
        x=cv2.resize(x,(224,224))
        x=x.astype("float32")
        x=x/255*2-1
        x=np.moveaxis(x,-1,0)
        x = np.expand_dims(x, axis=0)
        with torch.no_grad():
            img = torch.from_numpy(x).to(self.device)
            res=list(self.model(img).cpu().detach().numpy()[0])
            indice=res.index(max(res))
            clase=self.idx_to_class.get(indice)
        return clase 

    def predict_image(self,image,pad=True):
        """Generar predeccion de clase sobre imagen precargada"""
        x = np.array(image)
        # x=np.stack([x[:,:,0],x[:,:,1],x[:,:,2]], axis=-1)
        if pad:
            x=padding(x)
        imagen=x.copy()
        x=cv2.resize(x,(224,224))
        x=x.astype("float32")
        x=x/255*2-1
        x=np.moveaxis(x,-1,0)
        x = np.expand_dims(x, axis=0)
        with torch.no_grad():
            img = torch.from_numpy(x).to(self.device)
            res=list(self.model(img).cpu().detach().numpy()[0])
            indice=res.index(max(res))
            clase=self.idx_to_class.get(indice)
        return clase, imagen
    
def padding(img):
    """Escala la imagen y completa el sobrante con franjas negras, para no perder proporciones"""                
    old_image_height, old_image_width, channels = img.shape
    new_image_width = 224
    new_image_height = 224
    color = (0,0,0)
    if old_image_height<=old_image_width:
        f=new_image_width/old_image_width
    else:
        f=new_image_height/old_image_height
    img=cv2.resize(img,(int(f*old_image_width),int(f*old_image_height)))
    old_image_height, old_image_width, channels = img.shape
    result = np.full((new_image_height,new_image_width, channels), color, dtype=np.uint8)
    x_center = (new_image_width - old_image_width) // 2
    y_center = (new_image_height - old_image_height) // 2
    try:
        result[y_center:y_center+old_image_height, 
            x_center:x_center+old_image_width] = img
    except:
        result=img 
    return result  

def cortar_deteccion_municipio(shp_municipio,shp_deteccion,output_path,buffer=20):
    """
    Funcion para cortar las detecciones unicamente dentro de municipo 
    """
    municipio=gpd.read_file(shp_municipio)
    municipio=municipio.to_crs(3857)
    municipio_r=cascaded_union(municipio["geometry"].buffer(buffer))
    deteccion=gpd.read_file(shp_deteccion)
    deteccion=deteccion.to_crs(3857)
    deteccionsobremun=deteccion[[municipio_r.contains(x) for x in deteccion["geometry"].centroid]]
    deteccionsobremun.to_file(output_path)      