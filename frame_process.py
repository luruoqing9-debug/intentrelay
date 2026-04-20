import time
import threading
import requests
import base64
import cv2
import numpy as np
from flask import Flask, Response, render_template_string, request, jsonify
from ultralytics import YOLO
from viewpoint import get_resolution_and_viewpoint_base64
from copy import deepcopy


# 配置
API_URL = "http://localhost:29003/getCurrentFrame"
FETCH_INTERVAL = 0.1  # 秒
OUTPUT_VIDEO_FPS = 10
is_show_bbox = False  # 默认显示 bbox，可切换为 False 隐藏


app = Flask(__name__)
model = YOLO('yolov8x-seg.pt')

latest_processed_frame = None
lock = threading.Lock()

@app.route('/yolo_process', methods=['POST'])
def yolo_process():
    # W = 576 
    # H = 1024
    global latest_processed_frame
    data = request.get_json()
    img_base64 = data.get("frame")
    count = data.get("count")
    detection_info = []
    if img_base64:
        frame_bytes = base64.b64decode(img_base64)
        frame_np = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
        # results = model(frame_np, conf=0.4, iou=0.5)
        results = model(frame_np, conf=0.4, iou=0.5, imgsz=(frame_np.shape[0], frame_np.shape[1]))  # 新增 imgsz，尽量贴原图

        boxes = results[0].boxes
        # if boxes is not None and boxes.xyxy is not None and boxes.cls is not None:
        if boxes is not None and boxes.xyxyn is not None and boxes.cls is not None:
            for box, cls in zip(boxes.xyxy.cpu().numpy(), boxes.cls.cpu().numpy()):
                name = model.names[int(cls)]
                # x1, y1, x2, y2 = map(float, box)
                x1, y1, x2, y2 = box  # 0~1 的归一化坐标
                # print(box)
                detection_info.append({
                    'name': name,
                    # 'box': [x1, y1, x2, y2]
                    'box': [float(x1), float(y1), float(x2), float(y2)]
                })
        # 分割后画面
        # 隐藏 bbox：使用原图
        if is_show_bbox:
            processed_frame = results[0].plot(img=frame_np.copy())
        else:
            processed_frame = deepcopy(frame_np)  # 不绘制bbox，返回原始图像

        with lock:
            latest_processed_frame = processed_frame
        # _, img_encoded = cv2.imencode('.jpg', processed_frame)
        # processed_base64 = base64.b64encode(img_encoded).decode('utf-8')
        processed_base64 = img_base64 
        # 保存processed_base64到本地
        res, vp = get_resolution_and_viewpoint_base64(img_base64)
        view_point = [vp[0], vp[1]] if vp else None
        # print("VP: ",vp)
        return jsonify({
            "frame": processed_base64,
            "count": count,
            "detection": detection_info,
            "view_point": view_point
        })
    return jsonify({"error": "no frame"}), 400

@app.route('/show_video')
def show_video():
    HTML_TEMPLATE = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>YOLO分割实时画面</title>
        <style>
            body, html { margin: 0; padding: 0; width: 100%; height: 100%; background: #000; display: flex; justify-content: center; align-items: center; }
            img { max-width: 100%; max-height: 100%; object-fit: contain; }
        </style>
    </head>
    <body>
        <img src="{{ url_for('processed_feed') }}" ondblclick="toggleFullScreen(this)">
        <script>
            function toggleFullScreen(elem) {
                if (!document.fullscreenElement) {
                    if (elem.requestFullscreen) elem.requestFullscreen();
                } else {
                    if (document.exitFullscreen) document.exitFullscreen();
                }
            }
        </script>
    </body>
    </html>
    """
    return render_template_string(HTML_TEMPLATE)

def generate_live_preview():
    last_frame_sent = None
    while True:
        frame_to_send = None
        with lock:
            if latest_processed_frame is not None:
                frame_to_send = latest_processed_frame
                last_frame_sent = frame_to_send
        if frame_to_send is None and last_frame_sent is not None:
            frame_to_send = last_frame_sent
        if frame_to_send is not None:
            flag, encodedImage = cv2.imencode(".jpg", frame_to_send)
            if not flag:
                time.sleep(0.01)
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   bytearray(encodedImage) + b'\r\n')
            time.sleep(1/OUTPUT_VIDEO_FPS)
        else:
            time.sleep(0.1)

@app.route('/processed_feed')
def processed_feed():
    return Response(generate_live_preview(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')




if __name__ == '__main__':
    # t = threading.Thread(target=fetch_and_process_loop, daemon=True)
    # t.start()
    app.run(host="0.0.0.0", port=5000, debug=False)