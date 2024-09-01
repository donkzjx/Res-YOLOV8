import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/home/featurize/work/yolov8-main/yolov8n.pt') # select your model.pt path
    model.track(source='/home/featurize/work/yolov8-main/testmp4/j2548.mp4',
                imgsz=640,
                project='runs/track',
                name='exp',
                save=True
                )