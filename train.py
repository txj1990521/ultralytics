import torch
import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 下面老是报错 shape 不一致

if __name__ == '__main__':
    # 方案1：使用配置文件而不是预训练模型，让模型根据您的数据集重新初始化
    model = YOLO(model=r'yolo11n.pt')

    # 方案2：如果坚持使用预训练模型，可以添加下面的参数来避免验证阶段的问题
    # model = YOLO(model=r'E:\py cs\data\ultralytics-main\yolov11-seg-pt\yolo11n-seg.pt')

    model.train(data=r'ultralytics/cfg/datasets/shengshui.yaml',
                imgsz=640,
                epochs=3000,
                batch=2,
                workers=0,
                device='0',
                optimizer='SGD',
                resume=False,
                project='runs/shengshui',
                name='exp',
                single_cls=False,  # 设置为True，确保模型知道这是单类别任务
                cache=False,
                )