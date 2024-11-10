import json
import numpy as np
import cv2
from PIL import Image, ImageDraw

with open('../data/labels/0854.json', 'r') as f:
    label_data = json.load(f)
# 创建一个与原图像大小相同的空掩码
image_height = label_data['imageHeight']
image_width = label_data['imageWidth']
mask = np.zeros((image_height, image_width), dtype=np.uint8)

# 将标注的多边形区域绘制到掩码上
for shape in label_data['shapes']:
    label = shape['label']
    points = shape['points']
    # 确保所有坐标都是整数
    points = [(int(x), int(y)) for x, y in points]

    # 创建一个空白图像，用于绘制多边形
    img = Image.new('L', (image_width, image_height), 0)
    draw = ImageDraw.Draw(img)

    # 根据标签绘制掩码，1表示“道路”，0表示“非道路”
    if label == "road":
        draw.polygon(points, outline=1, fill=1)
    elif label == "non-road":
        draw.polygon(points, outline=0, fill=0)

    # 将绘制的掩码添加到最终的掩码中
    mask = np.maximum(mask, np.array(img))
cv2.imwrite('../data/imgs/mask0854.png', mask * 255)