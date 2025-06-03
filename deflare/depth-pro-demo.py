from PIL import Image
import depth_pro
import torch # 确保导入 torch
import numpy as np

image_path = '/home/user2/wns/deflare/test/input_000000.png'

# Load model and preprocessing transform
model, transform = depth_pro.create_model_and_transforms()
model.eval()

# Load and preprocess an image.
image, _, f_px = depth_pro.load_rgb(image_path)
image = transform(image)

# Run inference.
prediction = model.infer(image, f_px=f_px)
depth = prediction["depth"]  # Depth in [m].

# --- 添加以下代码来统计深度数据的上下限 ---
if isinstance(depth, torch.Tensor):
    # 如果 depth 是 PyTorch 张量，直接使用其 min() 和 max() 方法
    min_depth_actual = depth.min().item() # .item() 将张量转换为 Python 数值
    max_depth_actual = depth.max().item()
elif isinstance(depth, np.ndarray):
    # 如果 depth 最终转换为 NumPy 数组，则使用 NumPy 的 min() 和 max()
    min_depth_actual = depth.min()
    max_depth_actual = depth.max()
else:
    min_depth_actual = "未知"
    max_depth_actual = "未知"

print(f"针对图像 '{image_path}' 的实际深度数据：")
print(f"  最小值: {min_depth_actual:.4f} 米")
print(f"  最大值: {max_depth_actual:.4f} 米")

# 以下是您原有的代码，可以继续执行或打印其他信息
focallength_px = prediction["focallength_px"]  # Focal length in pixels.
# print(f"焦距: {focallength_px:.2f} 像素") # 如果您想打印焦距