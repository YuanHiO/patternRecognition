import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
# 加载图像
image_path = '../data/imgs/0618.png'
image = Image.open(image_path)
image_np = np.array(image)

# 显示图像
fig, ax = plt.subplots()
ax.imshow(image_np)
plt.title('Left road, Right non-road, close window to finish')

# 初始化训练样本列表
trnx = []
trny = []

def onclick(event):
    global trnx, trny
    if event.button == 1:  # 左键点击
        label = 1  # 道路区域
    else:  # 右键点击
        label = 0  # 非道路区域
    
    # 获取当前点击位置的RGB值
    x, y = int(event.xdata), int(event.ydata)
    pixel_value = image_np[y, x]
    
    # 记录样本
    trnx.append(pixel_value)
    trny.append(label)
    
    # 在图像上标记点击位置
    ax.plot(x, y, 'go' if label == 1 else 'ro')  # 绿色标记道路区域，红色标记非道路区域
    fig.canvas.draw_idle()

# 连接点击事件
cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()
# 断开点击事件连接
fig.canvas.mpl_disconnect(cid)

trnx = np.array(trnx)
trny = np.array(trny)
# 保存训练样本
np.savez('training_data.npz', trnx=trnx, trny=trny)