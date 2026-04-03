"""
speech.py - 语音录制与转文本模块
包含：录音、语音转文本功能（支持科大讯飞）
"""

import os
import sys

# 设置 Windows 控制台编码为 UTF-8
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import wave
import tempfile
import base64
import hashlib
import hmac
import datetime
import json
import ssl
from urllib.parse import urlencode, urlparse
from dotenv import load_dotenv
load_dotenv()

# 语音转文本方案选择
STT_PROVIDER = os.environ.get("STT_PROVIDER", "xunfei")  # xunfei / whisper / google / local_whisper

# ========== 科大讯飞语音转文本 ==========

XUNFEI_APPID = os.environ.get("XUNFEI_APPID", "")
XUNFEI_API_KEY = os.environ.get("XUNFEI_API_KEY", "")
XUNFEI_API_SECRET = os.environ.get("XUNFEI_API_SECRET", "")


def create_xunfei_url() -> str:
    """生成科大讯飞 WebSocket 认证 URL"""
    base_url = "wss://ws-api.xfyun.cn/v2/iat"

    # 生成鉴权参数
    now = datetime.datetime.now().strftime("%a, %d %b %Y %H:%M:%S GMT")

    signature_origin = f"host: ws-api.xfyun.cn\ndate: {now}\nGET /v2/iat HTTP/1.1"
    signature_sha = hmac.new(
        XUNFEI_API_SECRET.encode('utf-8'),
        signature_origin.encode('utf-8'),
        hashlib.sha256
    ).digest()
    signature = base64.b64encode(signature_sha).decode(encoding='utf-8')

    authorization_origin = f"api_key=\"{XUNFEI_API_KEY}\", algorithm=\"hmac-sha256\", headers=\"host date request-line\", signature=\"{signature}\""
    authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')

    params = {
        "authorization": authorization,
        "date": now,
        "host": "ws-api.xfyun.cn"
    }

    return base_url + "?" + urlencode(params)


def transcribe_with_xunfei(audio_data: bytes) -> str:
    """使用科大讯飞语音听写 API 转文本"""
    import websocket

    if not XUNFEI_APPID or not XUNFEI_API_KEY or not XUNFEI_API_SECRET:
        print("[Speech] Error: XUNFEI credentials not set (APPID, API_KEY, API_SECRET)")
        return ""

    ws_url = create_xunfei_url()
    transcript_result = []

    def on_message(ws, message):
        # 确保 message 是字符串（WebSocket 可能返回 bytes）
        if isinstance(message, bytes):
            message = message.decode('utf-8')

        data = json.loads(message)
        if data.get("code", 0) != 0:
            print(f"[Xunfei] Error: {data.get('code')} - {data.get('message')}")
            ws.close()
            return

        # 解码结果
        result_data = data.get("data", {})
        if result_data:
            status = result_data.get("status", 0)
            result = result_data.get("result", {})

            # 解析识别结果
            ws_list = result.get("ws", [])
            for ws_item in ws_list:
                for cw in ws_item.get("cw", []):
                    w = cw.get("w", "")
                    transcript_result.append(w)

            # status: 0-首帧, 1-中间帧, 2-最后一帧
            if status == 2:
                ws.close()

    def on_error(ws, error):
        print(f"[Xunfei] WebSocket error: {error}")

    def on_close(ws, close_status_code, close_msg):
        pass

    def on_open(ws):
        # 发送第一帧（包含业务参数）
        first_frame = {
            "common": {
                "app_id": XUNFEI_APPID
            },
            "business": {
                "language": "zh_cn",  # 中文
                "domain": "iat",      # 通用听写
                "accent": "mandarin", # 普通话
                "vad_eos": 2000,      # 静音检测超时（毫秒）
                "dwa": "wpgs"         # 动态修正
            },
            "data": {
                "status": 0,
                "format": "audio/L16;rate=16000",
                "encoding": "raw",
                "audio": base64.b64encode(audio_data[:1280]).decode('utf-8')
            }
        }
        ws.send(json.dumps(first_frame))

        # 发送中间帧
        chunk_size = 1280
        offset = chunk_size
        while offset < len(audio_data):
            chunk = audio_data[offset:offset + chunk_size]
            middle_frame = {
                "data": {
                    "status": 1,
                    "format": "audio/L16;rate=16000",
                    "encoding": "raw",
                    "audio": base64.b64encode(chunk).decode('utf-8')
                }
            }
            ws.send(json.dumps(middle_frame))
            offset += chunk_size

        # 发送最后一帧
        last_frame = {
            "data": {
                "status": 2,
                "format": "audio/L16;rate=16000",
                "encoding": "raw",
                "audio": base64.b64encode(audio_data[offset:]).decode('utf-8') if offset < len(audio_data) else ""
            }
        }
        ws.send(json.dumps(last_frame))

    # WebSocket 连接
    ws = websocket.WebSocketApp(
        ws_url,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
        on_open=on_open
    )

    ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})

    transcript = "".join(transcript_result)
    print(f"[Speech] Result: '{transcript}'")
    return transcript

def transcribe_with_whisper(audio_path: str) -> str:
    """使用 OpenAI Whisper API 转文本"""
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[Speech] Error: OPENAI_API_KEY not set")
        return ""

    client = OpenAI(api_key=api_key)

    with open(audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language="zh"  # 中文
        )

    print(f"[Whisper] Transcribed: '{transcript.text}'")
    return transcript.text


# ========== Google Speech-to-Text 方案 ==========

def transcribe_with_google(audio_path: str) -> str:
    """使用 Google Cloud Speech-to-Text"""
    from google.cloud import speech_v1

    client = speech_v1.SpeechClient()

    # 读取音频文件
    with open(audio_path, "rb") as audio_file:
        content = audio_file.read()

    audio = speech_v1.RecognitionAudio(content=content)
    config = speech_v1.RecognitionConfig(
        encoding=speech_v1.RecognitionConfig.AudioEncoding.LINEAR16,
        language_code="zh-CN",
        enable_automatic_punctuation=True
    )

    response = client.recognize(config=config, audio=audio)

    transcript = ""
    for result in response.results:
        transcript += result.alternatives[0].transcript

    print(f"[Google STT] Transcribed: '{transcript}'")
    return transcript


# ========== 本地 Whisper 模型方案 ==========

def transcribe_with_local_whisper(audio_path: str) -> str:
    """使用本地 Whisper 模型（无需 API key）"""
    import whisper

    # 加载模型（base 模型平衡速度和精度）
    model = whisper.load_model("base")

    result = model.transcribe(audio_path, language="zh")
    transcript = result["text"]

    print(f"[Local Whisper] Transcribed: '{transcript}'")
    return transcript


# ========== 统一转文本接口 ==========

def transcribe_audio(audio_data: bytes) -> str:
    """
    将音频数据转为文本。

    Args:
        audio_data: 音频原始数据（bytes）

    Returns:
        转录的文本内容
    """
    provider = STT_PROVIDER.lower()

    if provider == "xunfei":
        return transcribe_with_xunfei(audio_data)
    else:
        print(f"[Speech] Provider '{provider}' requires file path, fallback to xunfei")
        return transcribe_with_xunfei(audio_data)


# ========== 录音功能 ==========

import pyaudio
import threading

# 录音参数
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

# 录音状态
_is_recording = False
_audio_frames = []
_recording_thread = None


def _record_audio_async():
    """异步录音线程"""
    global _audio_frames

    p = pyaudio.PyAudio()

    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    print("[Speech] Recording started...")

    while _is_recording:
        data = stream.read(CHUNK, exception_on_overflow=False)
        _audio_frames.append(data)

    print("[Speech] Recording stopped.")

    stream.stop_stream()
    stream.close()
    p.terminate()


def start_recording():
    """开始录音"""
    global _is_recording, _audio_frames, _recording_thread

    if _is_recording:
        print("[Speech] Already recording!")
        return

    _is_recording = True
    _audio_frames = []
    _recording_thread = threading.Thread(target=_record_audio_async)
    _recording_thread.start()


def stop_recording() -> bytes:
    """
    停止录音，返回音频数据。

    Returns:
        音频原始数据（bytes）
    """
    global _is_recording, _audio_frames, _recording_thread

    if not _is_recording:
        print("[Speech] Not recording!")
        return b""

    _is_recording = False
    _recording_thread.join()

    audio_data = b"".join(_audio_frames)
    print(f"[Speech] Recording stopped, {len(audio_data)} bytes")
    return audio_data


def record_and_transcribe(duration: float = None) -> str:
    """
    录音并转文本（一键式接口）。

    Args:
        duration: 录音时长（秒），为 None 则需要手动调用 stop_recording

    Returns:
        转录的文本内容
    """
    start_recording()

    if duration:
        import time
        time.sleep(duration)
        audio_data = stop_recording()
    else:
        # 等待用户手动停止（需要外部调用 stop_recording）
        print("[Speech] Call stop_recording() to finish recording")
        return ""

    if audio_data:
        return transcribe_with_xunfei(audio_data)
    return ""


# ========== 便捷函数 ==========

def get_transcript_from_mic(duration: float = 5.0) -> str:
    """
    从麦克风录音并转文本（最简单的接口）。

    Args:
        duration: 录音时长（秒）

    Returns:
        转录的文本内容
    """
    print(f"[Speech] Recording for {duration} seconds...")
    return record_and_transcribe(duration)


# ========== 测试 ==========

if __name__ == "__main__":
    # 测试录音和转文本
    print("=== Speech Module Test ===")
    print("Speak now... (5 seconds)")

    transcript = get_transcript_from_mic(5.0)
    print(f"\nResult: {transcript}")