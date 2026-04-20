
import cv2
import numpy as np
import base64

def get_resolution_and_viewpoint_base64(img_base64):
    """
    输入base64编码的图片，返回分辨率和红色圆形视点坐标（如未检测到则为None）。
    返回：(width, height), (x, y, radius) 或 None
    """
    img_bytes = base64.b64decode(img_base64)
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("无法解码base64图片")
    height, width = img.shape[:2]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 | mask2

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    viewpoint = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            circularity = 4 * np.pi * area / (cv2.arcLength(cnt, True) ** 2)
            if 0.7 < circularity < 1.2:
                viewpoint = (int(x), int(y), int(radius))
                break
    return (width, height), viewpoint

def get_resolution_and_viewpoint(image_path):
    """
    输入图像地址，返回分辨率和红色圆形视点坐标（如未检测到则为None）。
    返回：(width, height), (x, y, radius) 或 None
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")
    height, width = img.shape[:2]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 | mask2

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    viewpoint = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            circularity = 4 * np.pi * area / (cv2.arcLength(cnt, True) ** 2)
            if 0.7 < circularity < 1.2:
                viewpoint = (int(x), int(y), int(radius))
                break
    return (width, height), viewpoint

# # 示例用法
# if __name__ == "__main__":
#     with open('../IMG_0221.PNG', 'rb') as f:
#         img_bytes = f.read()
#     import base64
#     img_base64 = base64.b64encode(img_bytes).decode('utf-8')
#     res, vp = get_resolution_and_viewpoint_base64(img_base64)
#     print(f"分辨率: {res[0]}x{res[1]}")
#     if vp:
#         print(f"视点坐标: ({vp[0]}, {vp[1]})，半径: {vp[2]}")
#     else:
#         print("未检测到红色圆形视点。")