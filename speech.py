"""
speech.py - 实时语音转写（科大讯飞RTASR）
支持两种模式：
- mico=0：分贝检测模式，自动触发回调
- mico=1：持续发送模式，手动获取文本
"""

import sys
import os

from dotenv import load_dotenv
load_dotenv()

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

import time
import threading
import hashlib
import hmac
import base64
import json
import pyaudio
import numpy as np
from urllib.parse import quote
import websocket

# ========== 讯飞配置 ==========
APPID = "3e3c8301"
API_KEY = "6d81482eb1e976f3402e20e69684f6a2"

print(f"[speech.py] 讯飞配置: APPID={APPID}, API_KEY已设置")

# ========== 音频参数 ==========
CHUNK = 1280
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

# ========== 分贝检测参数（仅 mico=0 使用）==========
DB_THRESHOLD = 30  # 分贝阈值
SILENCE_DURATION = 1.0  # 连续静音时长（秒）后停止

# ========== 模式状态 ==========
_current_mico = 0  # 当前模式，默认 mico=0

def set_mico_mode(mode: int):
    """
    设置当前模式

    Args:
        mode: 0 = 分贝检测模式（自动触发），1 = 持续发送模式（手动获取）
    """
    global _current_mico
    _current_mico = mode
    print(f"[speech] 模式切换: mico={mode}")

def get_mico_mode() -> int:
    """获取当前模式"""
    return _current_mico


# ========== 回调函数 ==========
_on_speech_end_callback = None

def set_speech_end_callback(callback):
    """设置语音结束回调函数（仅 mico=0 模式使用）"""
    global _on_speech_end_callback
    _on_speech_end_callback = callback


def calculate_db(audio_data: bytes) -> float:
    """计算音频数据的分贝值"""
    audio_array = np.frombuffer(audio_data, dtype=np.int16)

    if len(audio_array) == 0:
        return 0

    rms = np.sqrt(np.mean(audio_array ** 2))

    if rms == 0:
        return 0

    db = 20 * np.log10(rms / 32767.0) + 100

    return max(0, min(100, db))


class RTASRClient:
    """实时语音转写客户端"""

    def __init__(self):
        self.ws = None
        self.running = False
        self.all_text = ""
        self.current_sentence = ""
        self.trecv = None
        self.tsend = None
        self.end_tag = "{\"end\": true}"
        self.handshake_done = False  # WebSocket握手是否完成

        # 分贝检测状态（仅 mico=0 使用）
        self.is_speaking = False
        self.silence_start_time = None

    def _get_url(self):
        """讯飞 RTASR 签名算法"""
        base_url = "ws://rtasr.xfyun.cn/v1/ws"
        ts = str(int(time.time()))

        tt = (APPID + ts).encode('utf-8')
        md5 = hashlib.md5()
        md5.update(tt)
        baseString = md5.hexdigest()
        baseString = bytes(baseString, encoding='utf-8')

        apiKey = API_KEY.encode('utf-8')
        signa = hmac.new(apiKey, baseString, hashlib.sha1).digest()
        signa = base64.b64encode(signa)
        signa = str(signa, 'utf-8')

        url = base_url + "?appid=" + APPID + "&ts=" + ts + "&signa=" + quote(signa)
        return url

    def start(self):
        """启动语音识别"""
        self.running = True
        self.all_text = ""
        self.current_sentence = ""
        self.is_speaking = False
        self.silence_start_time = None
        self.handshake_done = False

        url = self._get_url()
        print(f"【连接WebSocket】")
        self.ws = websocket.create_connection(url)

        self.trecv = threading.Thread(target=self._recv, daemon=True)
        self.trecv.start()

        self.tsend = threading.Thread(target=self._send_mic, daemon=True)
        self.tsend.start()

        mode_str = "分贝检测" if _current_mico == 0 else "持续发送"
        print(f"【语音识别已启动】模式: mico={_current_mico} ({mode_str})")

    def stop(self):
        """停止语音识别"""
        self.running = False
        if self.ws:
            try:
                self.ws.send(bytes(self.end_tag.encode('utf-8')))
                self.ws.close()
            except:
                pass

    def _send_mic(self):
        """从麦克风录音，根据模式决定发送逻辑"""
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                       input=True, frames_per_buffer=CHUNK)

        print(f"【监听麦克风】等待握手完成...")

        # 等待握手完成
        handshake_wait = 0
        while not self.handshake_done and self.running and handshake_wait < 10:
            time.sleep(0.1)
            handshake_wait += 0.1

        if not self.handshake_done:
            print("【握手超时】连接失败")
            stream.close()
            p.terminate()
            return

        print(f"【握手完成】开始监听")

        while self.running:
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)

                if _current_mico == 1:
                    # mico=1 模式：持续发送，不做分贝检测
                    if self.ws and self.handshake_done:
                        self.ws.send(data)

                else:
                    # mico=0 模式：分贝检测
                    db = calculate_db(data)

                    if db >= DB_THRESHOLD:
                        # 音量高于阈值 → 正在说话
                        if not self.is_speaking:
                            print(f"【开始说话】分贝={db:.1f}")
                            self.is_speaking = True

                        self.silence_start_time = None

                        # 发送音频数据给讯飞（确保握手完成）
                        if self.ws and self.handshake_done:
                            self.ws.send(data)

                    else:
                        # 音量低于阈值
                        # 即使不说话，也发送静音数据保持连接活跃
                        if self.ws and self.handshake_done:
                            self.ws.send(data)

                        if self.is_speaking:
                            if self.silence_start_time is None:
                                self.silence_start_time = time.time()
                            else:
                                elapsed = time.time() - self.silence_start_time
                                if elapsed >= SILENCE_DURATION:
                                    # 连续静音超过阈值，说话结束
                                    print(f"【说话结束】静音{elapsed:.1f}秒")
                                    self.is_speaking = False

                                    # 触发回调
                                    self._trigger_callback()

                                    # 重置状态，准备下一次说话
                                    self.all_text = ""
                                    self.current_sentence = ""
                                    self.silence_start_time = None

            except Exception as e:
                print(f"【发送错误】{e}")
                # 不直接退出，等待重连
                if self.running:
                    print("【发送线程】等待重连...")
                    reconnect_wait = 0
                    while self.running and reconnect_wait < 5:
                        time.sleep(0.1)
                        reconnect_wait += 0.1
                        # 检查 ws 存在且握手完成
                        if self.ws and self.handshake_done:
                            print("【发送线程】检测到新连接，继续监听")
                            break

                    if self.ws and self.handshake_done:
                        continue
                    else:
                        print("【发送线程】重连失败，退出")
                        break
                else:
                    break

            time.sleep(0.04)

        stream.close()
        p.terminate()

    def _trigger_callback(self):
        """触发语音结束回调（仅 mico=0）"""
        global _on_speech_end_callback

        # 使用 get_text() 获取完整文本（包括中间结果）
        final_text = self.get_text().strip()

        if final_text and len(final_text) > 5:
            print(f"【回调触发】文本: '{final_text}'")
            if _on_speech_end_callback:
                try:
                    _on_speech_end_callback(final_text)
                except Exception as e:
                    print(f"【回调错误】{e}")
        else:
            print(f"【回调跳过】文本太短或为空: '{final_text}'")

    def _recv(self):
        """接收转写结果（支持自动重连）"""
        while self.running:
            try:
                # 等待 ws 存在
                if not self.ws:
                    time.sleep(0.1)
                    continue

                # 直接尝试接收，不检查 connected 属性
                result = str(self.ws.recv())
                if len(result) == 0:
                    print("【接收结束】")
                    if self.running:
                        self._reconnect()
                    continue

                result_dict = json.loads(result)

                if result_dict["action"] == "started":
                    self.handshake_done = True
                    print("【握手成功】")

                elif result_dict["action"] == "result":
                    # 忽略握手前的结果
                    if not self.handshake_done:
                        continue

                    data_str = result_dict.get("data", "")
                    if data_str:
                        try:
                            data_json = json.loads(data_str)
                            cn = data_json.get("cn", {})
                            st = cn.get("st", {})
                            rt_list = st.get("rt", [])

                            text = ""
                            for rt in rt_list:
                                for ws_item in rt.get("ws", []):
                                    for cw in ws_item.get("cw", []):
                                        text += cw.get("w", "")

                            if text:
                                result_type = st.get("type", "1")
                                text_stripped = text.strip()

                                # 过滤语气词
                                meaningless = ["嗯", "啊", "呃", "额", "唔", "噢", "哦", "哈", "恩"]
                                should_filter = any(text_stripped == w or
                                                   (len(text_stripped) > 0 and all(c == w[0] for c in text_stripped))
                                                   for w in meaningless)

                                if should_filter:
                                    if result_type == "0":
                                        self.current_sentence = ""
                                    print(f"○ 过滤语气词: {text}")
                                elif result_type == "0":
                                    if len(text_stripped) <= 5:
                                        print(f"○ 过滤短文本: {text}")
                                    else:
                                        self.all_text += text
                                        print(f"✓ 最终: {text}")
                                    self.current_sentence = ""
                                else:
                                    self.current_sentence = text
                                    print(f"~ 中间: {text}")

                        except Exception as e:
                            print(f"【解析失败】{e}")

                elif result_dict["action"] == "error":
                    print("【错误】" + result)
                    if self.running:
                        self._reconnect()

            except websocket.WebSocketConnectionClosedException:
                print("【连接关闭】")
                if self.running:
                    self._reconnect()

            except Exception as e:
                print(f"【接收异常】{e}")
                if self.running:
                    self._reconnect()

    def _reconnect(self):
        """重新连接 WebSocket（只创建连接，让 _recv 线程处理握手）"""
        try:
            print("【重连中】创建新连接...")
            time.sleep(0.5)

            # 关闭旧连接
            if self.ws:
                try:
                    self.ws.close()
                except:
                    pass

            # 创建新连接
            url = self._get_url()
            self.ws = websocket.create_connection(url)
            self.handshake_done = False
            print("【新连接已创建】等待 _recv 线程接收握手消息...")

            # 不在这里手动 recv()，让 _recv 线程自然接收握手消息

        except Exception as e:
            print(f"【重连异常】{e}")

    def get_text(self):
        return self.all_text + self.current_sentence

    def clear_text(self):
        self.all_text = ""
        self.current_sentence = ""


# ========== 全局管理器 ==========

_speech_client = None
_speech_running = False
_speech_lock = threading.Lock()


def start_continuous_speech():
    """启动持续语音识别"""
    global _speech_client, _speech_running

    with _speech_lock:
        if _speech_running:
            print("【语音识别已在运行】")
            return

        _speech_running = True
        _speech_client = RTASRClient()
        _speech_client.start()


def stop_continuous_speech():
    """停止语音识别"""
    global _speech_client, _speech_running

    with _speech_lock:
        if not _speech_running:
            return

        _speech_running = False
        if _speech_client:
            _speech_client.stop()
            _speech_client = None


def get_accumulated_text() -> str:
    """获取累积文本"""
    global _speech_client

    with _speech_lock:
        if _speech_client:
            return _speech_client.get_text().strip()
        return ""


def clear_accumulated_text():
    """清空累积文本"""
    global _speech_client

    with _speech_lock:
        if _speech_client:
            _speech_client.clear_text()


def is_speech_running() -> bool:
    """检查是否运行"""
    with _speech_lock:
        return _speech_running


def has_speech_text() -> bool:
    """检查是否有文本"""
    return len(get_accumulated_text()) > 0


def get_text_and_clear() -> str:
    """获取并清空文本"""
    text = get_accumulated_text()
    clear_accumulated_text()
    return text


# ========== 测试 ==========

if __name__ == "__main__":
    print("=" * 40)
    print("双模式语音识别测试")
    print("=" * 40)
    print(f"分贝阈值: {DB_THRESHOLD}")
    print(f"静音时长: {SILENCE_DURATION}秒")

    def on_end(text):
        print(f"\n>>> VLM触发: '{text}' <<<\n")

    set_speech_end_callback(on_end)
    set_mico_mode(0)  # 默认 mico=0

    print("\n当前模式: mico=0 (分贝检测)")
    print("说话测试...")

    start_continuous_speech()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n【停止】")
        stop_continuous_speech()