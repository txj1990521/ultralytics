import json
import os
import random
import shutil

# ========= 配置区 =========
labelme_dir = r"D:\zhanlan\segment_data\花色随机拍摄照片"   # json所在目录
image_dir   = r"D:\zhanlan\segment_data\花色随机拍摄照片"   # 图片所在目录
output_dir  = r"D:\zhanlan\yolo_segment_data"              # 输出根目录

CLASSES = ['fabric', 'part_box', 'long_fabric']            # 类别顺序(=类别id)
train_ratio = 0.8                                          # 训练集比例(可调)
seed = 42                                                  # 随机种子(保证可复现)
split_names = ("train", "val")                             # 你也可以改成 ("train","test")
# =========================

CLASS2ID = {c: i for i, c in enumerate(CLASSES)}

def load_labelme_json(json_file):
    # 避免中文文件名乱码
    with open(json_file, 'r', encoding='utf-8-sig') as f:
        return json.load(f)

def norm_points(points, w, h):
    # LabelMe points: [[x,y], [x,y], ...] -> YOLO seg needs normalized x y pairs
    out = []
    for x, y in points:
        out.append(x / w)
        out.append(y / h)
    return out

def ensure_dirs(root):
    for top in ("images", "labels"):
        for sp in split_names:
            os.makedirs(os.path.join(root, top, sp), exist_ok=True)

def convert_one(json_path, split):
    data = load_labelme_json(json_path)

    # 处理 imagePath：只取文件名（防止里面带相对路径）
    img_name = os.path.basename(data.get("imagePath", ""))
    src_img = os.path.join(image_dir, img_name)

    if not os.path.exists(src_img):
        print(f"[跳过] 找不到图片: {src_img}")
        return

    # 图像尺寸：优先用json里的 imageWidth/imageHeight
    w = data.get("imageWidth", None)
    h = data.get("imageHeight", None)
    if not w or not h:
        # 如果json里没有，就用PIL读一次
        from PIL import Image
        with Image.open(src_img) as im:
            w, h = im.size

    lines = []
    for shape in data.get("shapes", []):
        label = shape.get("label")
        if label not in CLASS2ID:
            print(f"[忽略] 未在CLASSES中定义的label: {label}  (json: {json_path})")
            continue

        pts = shape.get("points", [])
        if len(pts) < 3:
            # 分割多边形至少3个点
            continue

        cls_id = CLASS2ID[label]
        seg = norm_points(pts, w, h)

        # YOLO seg: class x1 y1 x2 y2 ...
        line = str(cls_id) + " " + " ".join(f"{v:.6f}" for v in seg)
        lines.append(line)

    # 输出路径
    img_out = os.path.join(output_dir, "images", split, img_name)
    txt_name = os.path.splitext(img_name)[0] + ".txt"
    label_out = os.path.join(output_dir, "labels", split, txt_name)

    # 复制图片
    if not os.path.exists(img_out):
        shutil.copy2(src_img, img_out)

    # 写标签（即使没有目标也写空文件，ultralytics可接受）
    with open(label_out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def main():
    ensure_dirs(output_dir)

    # 收集json列表
    json_files = [os.path.join(labelme_dir, f) for f in os.listdir(labelme_dir) if f.lower().endswith(".json")]
    json_files.sort()

    random.seed(seed)
    random.shuffle(json_files)

    n = len(json_files)
    n_train = int(n * train_ratio)

    train_list = json_files[:n_train]
    val_list   = json_files[n_train:]

    print(f"总数: {n} | train: {len(train_list)} | {split_names[1]}: {len(val_list)} | train_ratio={train_ratio}")

    for jp in train_list:
        convert_one(jp, split_names[0])
    for jp in val_list:
        convert_one(jp, split_names[1])

    print("Done.")

if __name__ == "__main__":
    main()
