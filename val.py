import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('./best.pt')#select your model path
    model.val(data='./.yaml',#select your data yaml path
            split='val',
            imgsz=640,
            batch=32,
            # iou=0.7,
            # rect=False,
            # save_json=True, # if you need to cal coco metrice
            project='aug-val2',
            name='night',
            )