import threading
import time
import requests
import cv2
import numpy as np
import json
import os
import base64

# 创建必要的文件夹
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OPERATED_IMAGE_DIR = os.path.join(PROJECT_DIR, "Operated_image")
DETECTED_IMAGE_DIR = os.path.join(PROJECT_DIR, "detected_images")

os.makedirs(OPERATED_IMAGE_DIR, exist_ok=True)
os.makedirs(DETECTED_IMAGE_DIR, exist_ok=True)

# API 配置
API_BASE_URL = "http://localhost:5000"  # api.py 的服务地址

class StreamTrigger:
    def __init__(self, stream_source, listener_func, trigger_action, check_interval=0.1):
        """
        :param stream_source: 视频流（含音频）的数据源，可以是摄像头、文件或网络流
        :param listener_func: 监听函数，接收一帧数据，返回True/False
        :param trigger_action: 触发时执行的函数
        :param check_interval: 检查间隔（秒）
        """
        self.listener_func = listener_func
        self.trigger_action = trigger_action
        self.check_interval = check_interval
        self.running = False
        self.latest_payload = None
        self._feed_thread = threading.Thread(target=self._feed_loop, args=(stream_source,), daemon=True)

    def start(self):
        self.running = True
        self._feed_thread.start()
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self):
        self.running = False

    def _feed_loop(self, stream_source):
        for ret, payload in stream_source:
            if not self.running:
                break
            if ret:
                self.latest_payload = payload

    def _run(self):
        while self.running:
            payload = self.latest_payload
            if payload is not None:
                trigger = self.listener_func(payload)
                if trigger != False:
                    payload["trigger_type"] = trigger
                    self.trigger_action(payload)
            time.sleep(self.check_interval)


def get_current_frame(api_url="http://localhost:29003/getCurrentFrame"):
    try:
        resp = requests.get(api_url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            img_base64 = data.get("img_base64")
            count = data.get("count")
            return img_base64, count
    except Exception as e:
        print(f"获取视点图失败: {e}")
    return None, None


def unified_frame_stream(api_url="http://localhost:29003", yolo_url="http://127.0.0.1:5000/yolo_process", interval=0.1):
    """生成器：每次yield (True, {'frame': base64, 'count': count, 'detection': ...})"""
    while True:
        img_base64, count = get_current_frame(api_url + "/getCurrentFrame")
        if img_base64:
            try:
                resp = requests.post(yolo_url, json={"frame": img_base64, "count": count}, timeout=5)
                if resp.status_code == 200:
                    payload = resp.json()
                    yield True, payload
                else:
                    yield False, {}
            except Exception as e:
                print("YOLO处理请求失败:", e)
                yield False, {}
        else:
            yield False, {}
        time.sleep(interval)


def point_in_box(pt, box, margin=0):
    """判断点是否在框内；box = [x1,y1,x2,y2]，允许加 margin 作为抖动冗余"""
    if pt is None or box is None:
        return False
    x, y = pt
    x1, y1, x2, y2 = box
    left, right = (min(x1, x2) - margin, max(x1, x2) + margin)
    top, bottom = (min(y1, y2) - margin, max(y1, y2) + margin)
    return (left <= x <= right) and (top <= y <= bottom)


def make_obj_key(det):
    """给物体做稳定 key：优先 track_id；否则用量化后的 bbox 字符串"""
    tid = det.get("name", None)
    if tid is not None:
        return f"id:{tid}"
    b = det.get("box")
    if not b:
        return None
    # 量化到 8 像素网格，抗抖动
    q = [int(round(v / 8.0) * 8) for v in b]
    return f"box:{q[0]}_{q[1]}_{q[2]}_{q[3]}"


def bbox_iou(boxA, boxB):
    """计算两个 bbox 的 IoU"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    areaA = max(0, (boxA[2] - boxA[0])) * max(0, (boxA[3] - boxA[1]))
    areaB = max(0, (boxB[2] - boxB[0])) * max(0, (boxB[3] - boxB[1]))
    if areaA == 0 or areaB == 0:
        return 0
    return interArea / min(areaA, areaB)


if __name__ == "__main__":
    # 初始化跨调用状态
    if not hasattr(__import__("builtins"), "_fixation_state"):
        __import__("builtins")._fixation_state = {
            "current_key": None,     # 当前注视物体的 key
            "start_t": None,         # 开始注视该物体的时间
            "fired_for_key": None,   # 对该 key 是否已触发过
            "last_seen_t": None,     # 上次收到帧的时间
            "overlap_key": None,     # 当前重叠触发的 key（person+obj）
            "overlap_fired_for_key": None,  # 对该重叠 key 是否已触发
            "cooldown_until": None,  # 冷却截止的时间戳 (monotonic)
        }

    cool_down_time = 10  # 重叠触发冷却时间（秒）

    def example_listener(payload):
        state = __import__("builtins")._fixation_state
        now = time.monotonic()

        frame_b64 = payload.get('frame')
        detection = payload.get('detection')
        count = payload.get('count')
        view_point = payload.get('view_point', None)

        # 解码帧用于调试/保存
        if frame_b64:
            frame_bytes = base64.b64decode(frame_b64)
            frame_np = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
        else:
            frame_np = None

        # 检测手部与物体重叠
        if isinstance(detection, list):
            person_bboxes = [d.get('box') for d in detection if d.get('name') == 'person']
            nonperson_bboxes = [d.get('box') for d in detection if d.get('name') != 'person']

            overlap_hit_this_frame = False

            for pbox in person_bboxes:
                for nbox in nonperson_bboxes:
                    if pbox and nbox:
                        iou = bbox_iou(pbox, nbox)
                        if iou > 0.4:
                            overlap_hit_this_frame = True

                            key = ("overlap", tuple(map(int, pbox)), tuple(map(int, nbox)))
                            if key != state.get("overlap_key"):
                                state["overlap_key"] = key
                                state["overlap_fired_for_key"] = False

                            if state.get("overlap_fired_for_key") is False:
                                if (state.get("cooldown_until") is None) or (now >= state["cooldown_until"]):
                                    state["overlap_fired_for_key"] = True
                                    state["cooldown_until"] = now + cool_down_time
                                    example_listener.bbox["person"] = pbox
                                    example_listener.bbox["obj"] = nbox
                                    print("手部坐标与物体坐标重叠，触发一次")
                                    return "手部坐标与物体坐标重叠"

            if not overlap_hit_this_frame:
                state["overlap_key"] = None
                state["overlap_fired_for_key"] = None

        # 检测眼动焦点注视物体超过5秒
        if isinstance(detection, list):
            nonperson = [d for d in detection if d.get('name') != 'person']
            focus_det = None

            for det in nonperson:
                if point_in_box(view_point, det.get('box'), margin=4):
                    focus_det = det
                    break

            if focus_det is not None:
                key = make_obj_key(focus_det)

                if key != state["current_key"]:
                    state["current_key"] = key
                    state["start_t"] = now
                    state["fired_for_key"] = False
                else:
                    if state["start_t"] is None:
                        state["start_t"] = now
                    dwell = now - state["start_t"]
                    if (dwell >= 5.0) and (state["fired_for_key"] is False):
                        if (state.get("cooldown_until") is None) or (now >= state["cooldown_until"]):
                            state["fired_for_key"] = True
                            state["cooldown_until"] = now + cool_down_time
                            example_listener.bbox = getattr(example_listener, "bbox", {})
                            example_listener.bbox["obj"] = focus_det.get('box')
                            print("眼动焦点注视单一物体超过五秒钟，触发一次")
                            return "眼动焦点注视单一物体超过五秒钟"
            else:
                state["current_key"] = None
                state["start_t"] = None
                state["fired_for_key"] = None

        state["last_seen_t"] = now
        return False

    example_listener.bbox = {}

    def example_action(payload):
        """
        触发后执行的动作：
        1. 清空 Operated_image/ 文件夹
        2. 保存当前触发帧到 Operated_image/
        3. 调用 api.py 的 /vlm_analysis_images 接口进行 VLM 分析
        """
        trigger_type = payload.get("trigger_type")
        obj_bbox = example_listener.bbox.get("obj", None)
        person_bbox = example_listener.bbox.get("person", None)
        frame_b64 = payload.get('frame', None)

        print(f"\n{'='*50}")
        print(f"[触发] 类型: {trigger_type}")
        print(f"[触发] 物体 bbox: {obj_bbox}")
        print(f"[触发] Person bbox: {person_bbox}")

        if not frame_b64:
            print("[触发] 错误: 缺少帧数据")
            return

        # 1. 清空 Operated_image/ 文件夹
        for f in os.listdir(OPERATED_IMAGE_DIR):
            fpath = os.path.join(OPERATED_IMAGE_DIR, f)
            try:
                os.remove(fpath)
            except Exception as e:
                print(f"[触发] 清空文件失败: {fpath}, {e}")
        print(f"[触发] 已清空 Operated_image/")

        # 2. 保存当前触发帧到 Operated_image/
        frame_bytes = base64.b64decode(frame_b64)
        frame_np = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)

        # 绘制 bbox 标注
        if frame_np is not None:
            for label, bbox in example_listener.bbox.items():
                if bbox:
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(frame_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame_np, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 保存到 Operated_image/
        timestamp = int(time.time())
        operated_filename = f"trigger_frame_{timestamp}.jpg"
        operated_path = os.path.join(OPERATED_IMAGE_DIR, operated_filename)
        cv2.imwrite(operated_path, frame_np)
        print(f"[触发] 已保存帧到: {operated_path}")

        # 同时保存到 detected_images/ 用于调试
        detected_filename = f"detected_{trigger_type}_{timestamp}.jpg"
        detected_path = os.path.join(DETECTED_IMAGE_DIR, detected_filename)
        cv2.imwrite(detected_path, frame_np)
        print(f"[触发] 已保存调试图到: {detected_path}")

        # 3. 调用 /vlm_analysis_images 接口
        try:
            api_url = f"{API_BASE_URL}/vlm_analysis_images"
            request_data = {
                "trigger_types": [trigger_type]
            }

            print(f"[触发] 调用 API: {api_url}")
            print(f"[触发] 请求参数: {request_data}")

            resp = requests.post(api_url, json=request_data, timeout=30)

            if resp.status_code == 200:
                result = resp.json()
                print(f"[触发] VLM 分析结果:")
                print(f"  - success: {result.get('success')}")
                print(f"  - image_count: {result.get('image_count')}")
                print(f"  - node_type: {result.get('node_type')}")
                print(f"  - repeat_count: {result.get('repeat_count')}")
                print(f"  - should_feedback: {result.get('should_feedback')}")

                if result.get('vlm_result'):
                    vlm = result.get('vlm_result')
                    print(f"  - VLM 分析:")
                    print(f"    - type: {vlm.get('type')}")
                    print(f"    - label: {vlm.get('label')}")
                    print(f"    - User Speaking: {vlm.get('User Speaking')}")
                    print(f"    - Behavior description: {vlm.get('Behavior description')}")
                    print(f"    - User intent: {vlm.get('User intent')}")

                # 如果需要 AI 反馈，可以在这里调用 /ai_feedback
                if result.get('should_feedback') and result.get('parsed_vlm'):
                    print(f"[触发] 检测到需要 AI 反馈，可调用 /ai_feedback 接口")
                    # 可选：自动调用 AI 反馈
                    # feedback_resp = requests.post(
                    #     f"{API_BASE_URL}/ai_feedback",
                    #     json={
                    #         "component_name": result.get('vlm_result', {}).get('label', ''),
                    #         "parsed_vlm": result.get('parsed_vlm')
                    #     },
                    #     timeout=30
                    # )
            else:
                print(f"[触发] API 调用失败: {resp.status_code}")
                print(f"[触发] 错误信息: {resp.text}")

        except requests.exceptions.Timeout:
            print(f"[触发] API 调用超时")
        except Exception as e:
            print(f"[触发] API 调用异常: {e}")

        print(f"{'='*50}\n")

    def start_trigger():
        stream_gen = unified_frame_stream(api_url="http://localhost:29003", interval=0.1)
        trigger = StreamTrigger(stream_gen, example_listener, example_action, check_interval=0.5)
        trigger.start()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            trigger.stop()

    start_trigger()