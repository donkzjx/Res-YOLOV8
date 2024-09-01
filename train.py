import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/home/featurize/work/yolov8-main/ultralytics/cfg/models/v8/yolov8-EMA.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data='/home/featurize/work/yolov8-main/dataset/data.yaml',
                cache=False,
                imgsz=640,
                epochs=200,
                batch=16,
                workers=4,
                device='0',
                optimizer='SGD', # using SGD
                # resume='', # last.pt path
                # amp=False # close amp
                # fraction=0.2,
                project='runs/train',
                name='exp',
                )
    