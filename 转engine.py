from ultralytics import YOLO
# 调用.pt文件
model = YOLO(model=r'D:\MyPythonProject\ultralytics\runs\shengshui\exp4\weights\best.pt')
# 格式转换
model.export(
    format="engine",
    imgsz=640,
    device=0,
    half=True,        # FP16
    simplify=True
)

if __name__ == '__main__':
    # 调用转换后的engine文件
    model = YOLO(model=r'D:\MyPythonProject\ultralytics\runs\shengshui\exp4\weights\best.engine')
    # 进行推理
    model.predict(source=r'Z:\伊化煤矿\井筒壁漏水渗水检测\images\val',
                  save=True,
                  show=True)

