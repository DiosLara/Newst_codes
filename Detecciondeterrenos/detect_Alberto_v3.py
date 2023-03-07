import argparse
import time
from pathlib import Path
import tqdm
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages , letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel



opt_img_size=256
device = select_device("cpu")
class modelo():
    def __init__(self,weights=["yolov7.pt"]):
        model = attempt_load(weights, map_location=device)
        model = TracedModel(model, device, opt_img_size)
        self.model= model

    def detect(self,opt_conf_thres,opt_source="train/images",display=False,imagen_s=np.array([1,1])):
        vector=[]
        opt_no_trace=False
        opt_iou_thres=0.45
        opt_save_conf=False
        opt_classes=None
        opt_agnostic_nms=False
        opt_augment=False
        opt_no_trace=False
        source,  imgsz, trace = opt_source, opt_img_size, not opt_no_trace
        set_logging()
        stride = int(self.model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        if imagen_s.shape[0]==2:
            dataset = LoadImages(source, img_size=imgsz, stride=stride)
            t0 = time.time()
            for path, img, im0s, _ in dataset:
                # print(img.shape)
                img = torch.from_numpy(img).to(device)
                img = img.float()  # uint8 to fp16/32
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
                            vector.append(line)
                if display:
                    print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
        else:
            img0=imagen_s  
            img = letterbox(img0, imgsz, stride)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(device)
            img = img.float()  # uint8 to fp16/32
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

        

    # detect(opt_source='inference/images/',opt_weights=['best25.pt'],opt_img_size=256,opt_conf_thres=0.1,opt_no_trace=False)
