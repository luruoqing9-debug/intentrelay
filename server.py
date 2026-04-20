from flask import Flask, Response, request, render_template_string
from ultralytics import YOLO
import cv2
import numpy as np
import time
import os
import threading
from collections import deque
import json

# --- 全局配置 ---
start_time = time.time()  # 记录脚本启动时间
SERVER_HOST = '0.0.0.0'
SERVER_PORT = 5000
OUTPUT_VIDEO_FILENAME = 'output_processed.mp4'
OUTPUT_VIDEO_FPS = 30 # 预设的输出视频帧率

app = Flask(__name__)

model = YOLO('yolov8x-seg.pt')

frame_buffer = deque(maxlen=100)
processed_frame_buffer = deque(maxlen=100)
lock = threading.Lock()
is_streaming_active = False
processing_thread_obj = None

video_writer = None

def initialize_writer(width, height):
    global video_writer
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    video_writer = cv2.VideoWriter(OUTPUT_VIDEO_FILENAME, fourcc, OUTPUT_VIDEO_FPS, (width, height))
    if not video_writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(OUTPUT_VIDEO_FILENAME, fourcc, OUTPUT_VIDEO_FPS, (width, height))
    


def save_detection_info(detections, save_path='detection_log.json'):
    with open(save_path, 'a') as f:
        for det in detections:
            f.write(json.dumps(det, ensure_ascii=False) + '\n')


def yolo_processing_thread():
    global video_writer, is_streaming_active

    last_save_time = time.time()
    save_interval = 5  # 秒

    # 等待第一帧数据
    while len(frame_buffer) == 0 and is_streaming_active:
        time.sleep(0.01)

    # 只要还在推流或缓冲区还有内容，就继续处理
    while is_streaming_active or len(frame_buffer) > 0:
        if len(frame_buffer) > 0:
            frame = frame_buffer.popleft()
            h, w, _ = frame.shape
            # 在处理第一帧时初始化视频写入器
            if video_writer is None:
                initialize_writer(w, h)

            # YOLOv8 推理
            # 通过调整 conf (置信度) 和 iou (交并比) 参数来微调模型的精度和灵敏度。
            # conf: 阈值越低，检测到的物体越多，但也可能增加误报。
            # iou: 用于消除冗余的重叠检测框。
            results = model(frame, conf=0.4, iou=0.5)
            processed_frame = results[0].plot()

            now = time.time()
            if now - last_save_time >= save_interval:
                detection_info = []
                boxes = results[0].boxes
                if boxes is not None and boxes.xyxy is not None and boxes.cls is not None:
                    for box, cls in zip(boxes.xyxy.cpu().numpy(), boxes.cls.cpu().numpy()):
                        name = model.names[int(cls)]
                        x1, y1, x2, y2 = map(float, box)
                        detection_info.append({
                            'timestamp': round(now - start_time, 2),  # 相对时间，保留2位小数
                            'name': name,
                            'box': [x1, y1, x2, y2]
                        })
                if detection_info:
                    save_detection_info(detection_info)
                last_save_time = now
            # 将处理后的帧放入另一个缓冲区以便实时预览
            with lock:
                processed_frame_buffer.append(processed_frame)

            # 写入视频文件
            if video_writer and video_writer.isOpened():
                video_writer.write(processed_frame)
        else:
            # 如果缓冲区为空，短暂休眠
            time.sleep(0.01)
    
    if video_writer:
        video_writer.release()
        video_writer = None


@app.route('/video_feed', methods=['POST'])
def video_feed():
    global is_streaming_active, processing_thread_obj
    
    if not is_streaming_active:
        is_streaming_active = True
        frame_buffer.clear()
        processed_frame_buffer.clear()
        processing_thread_obj = threading.Thread(target=yolo_processing_thread)
        processing_thread_obj.start()

    try:
        nparr = np.frombuffer(request.data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is not None:
            frame_buffer.append(frame)
            return "Frame received", 200
        else:
            return "Bad image data", 400
    except Exception as e:
        print(f"处理帧时出错: {e}")
        return "Internal Server Error", 500

@app.route('/stop_stream', methods=['POST'])
def stop_stream():
    global is_streaming_active
    is_streaming_active = False
    return "Stream stop signal received", 200

def generate_live_preview():

    last_frame_sent = None 

    while is_streaming_active:
        frame_to_show = None
        with lock:
            if len(processed_frame_buffer) > 0:
                frame_to_show = processed_frame_buffer.popleft()
                last_frame_sent = frame_to_show 

        frame_to_send = frame_to_show if frame_to_show is not None else last_frame_sent

        if frame_to_send is not None:
            (flag, encodedImage) = cv2.imencode(".jpg", frame_to_send)
            if not flag:
                time.sleep(0.01)
                continue
            
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                  bytearray(encodedImage) + b'\r\n')
            
            time.sleep(1/30)
        else:
            time.sleep(0.1)

def generate_playback():
    while not os.path.exists(OUTPUT_VIDEO_FILENAME):
        time.sleep(1)

    cap = cv2.VideoCapture(OUTPUT_VIDEO_FILENAME)
    if not cap.isOpened():
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        (flag, encodedImage) = cv2.imencode(".jpg", frame)
        if not flag:
            continue
        
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
              bytearray(encodedImage) + b'\r\n')
        
        time.sleep(1/30)

# --- HTML页面模板 ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>实时视频流</title>
    <style>
        body, html {
            margin: 0;
            padding: 0;
            width: 100%;
            height: 100%;
            overflow: hidden;
            background-color: #000;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        img {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain; /* 保持视频的宽高比 */
        }
    </style>
</head>
<body>
    <img src="{{ url_for('processed_feed') }}" ondblclick="toggleFullScreen(this)">
    <script>
        function toggleFullScreen(elem) {
            if (!document.fullscreenElement) {
                if (elem.requestFullscreen) {
                    elem.requestFullscreen();
                }
            } else {
                if (document.exitFullscreen) {
                    document.exitFullscreen();
                }
            }
        }
    </script>
</body>
</html>
"""

@app.route('/show_video')
def show_video():
    return render_template_string(HTML_TEMPLATE)

@app.route('/processed_feed')
def processed_feed():
    if is_streaming_active:
        return Response(generate_live_preview(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return Response(generate_playback(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host=SERVER_HOST, port=SERVER_PORT, debug=False)
