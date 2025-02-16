import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('./ultralytics/cfg/models/v8/yolov8n.yaml')#select your model yaml path
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data='./dataset/all-3.yaml', #select your data yaml path
                cache=False,
                imgsz=640,
                epochs=300,
                batch=24,
                close_mosaic=0,
                workers=4,
                # device='0',
                optimizer='SGD', # using SGD
                # patience=0, # close earlystop
                # resume=True, # 断点续训,YOLO初始化时选择last.pt
                # amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                name='yolov8n_aug',
                )