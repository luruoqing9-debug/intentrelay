"""
record.py - 记录处理模块
包含：图像/文本编码、VLM分析、图像处理、存储工具
"""

import os
import sys

# 禁用 transformers 警告和进度条
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# 设置 HuggingFace 镜像（解决国内连接问题）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 加载 .env 文件中的环境变量
from dotenv import load_dotenv
load_dotenv()

import io
import json
import base64
import numpy as np
from PIL import Image
from typing import Literal
import warnings
warnings.filterwarnings('ignore')

from transformers import CLIPProcessor, CLIPModel
from google import genai
from google.genai import types

# --- 项目路径 ---
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(PROJECT_DIR, "cache")

# --- 模型初始化 ---
print("[record.py] 正在加载CLIP模型...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir=CACHE_DIR)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir=CACHE_DIR)

# 从环境变量读取 API Key（请在系统环境变量或 .env 文件中设置）
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("[record.py] Warning: GEMINI_API_KEY not set. Please set it in environment variables or .env file")
client = genai.Client(
    http_options={"api_version": "v1beta"},
    api_key=api_key,
)
print("[record.py] 模型加载完毕。")


# ========== 编码函数 ==========

def image_encoder(image: Image.Image) -> np.ndarray:
    """将图像编码为512维向量"""
    print("[image_encoder] Encoding image...")
    inputs = processor(images=image, return_tensors="pt", padding=True)
    outputs = model.get_image_features(**inputs)

    if hasattr(outputs, 'pooler_output'):
        image_features = outputs.pooler_output.detach().cpu().numpy()
    elif hasattr(outputs, 'detach'):
        image_features = outputs.detach().cpu().numpy()
    else:
        raise ValueError("Unexpected output format from model.get_image_features")

    image_features = image_features / np.linalg.norm(image_features, axis=1, keepdims=True)
    print(f"[image_encoder] Feature dim: {image_features.shape}, sample: {image_features[0][:5]}")
    return image_features[0]


def text_encoder(text: str = "a face of a person") -> np.ndarray:
    """将文本编码为512维向量"""
    print("[text_encoder] Encoding text...")
    text_inputs = processor(text=text, return_tensors="pt", padding=True)
    outputs = model.get_text_features(**text_inputs)

    if hasattr(outputs, 'pooler_output'):
        text_features = outputs.pooler_output.detach().cpu().numpy()
    elif hasattr(outputs, 'detach'):
        text_features = outputs.detach().cpu().numpy()
    else:
        raise ValueError("Unexpected output format from model.get_text_features")

    text_features = text_features / np.linalg.norm(text_features, axis=1, keepdims=True)
    print(f"[text_encoder] Feature dim: {text_features.shape}, sample: {text_features[0][:5]}")
    return text_features[0]


# ========== 图像处理函数 ==========

def get_crop(frame: Image.Image, bbox) -> Image.Image:
    """根据边界框裁剪图像"""
    crop_box = (bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax)
    return frame.crop(crop_box)


def encode_image_to_base64(image: Image.Image) -> str:
    """将PIL图像编码为Base64字符串"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    base64_bytes = base64.b64encode(img_bytes)
    return base64_bytes.decode("utf-8")


def open_from_path(path: str = None) -> Image.Image:
    """从路径打开图像"""
    if path is None:
        path = os.path.join(PROJECT_DIR, "test.png")
    return Image.open(path).convert("RGB")


# ========== 触发类型定义 ==========

PhysicalTriggerType = Literal[
    "眼动焦点注视单一物体超过五秒钟",
    "手部坐标与物体坐标重叠",
    "语音输入触发",
    "其他"
]

# TODO: 虚拟世界触发类型待定
# VirtualTriggerType = Literal[
#     "待定触发类型1",
#     "待定触发类型2",
# ]


# ========== 物理世界Prompt构建 ==========

def get_physical_analysis_prompt(trigger_types: list[PhysicalTriggerType], transcript_text: str) -> str:
    """
    构建物理世界分析的Prompt，整合触发类型列表和语音文本。

    Args:
        trigger_types: 物理世界触发类型列表（支持多种触发同时发生）
        transcript_text: 语音转文本结果（如无语音则为空字符串）

    Returns:
        VLM分析用的Prompt字符串
    """
    # 将触发类型列表转为字符串描述
    trigger_desc = "、".join(trigger_types) if trigger_types else "无"

    prompt = f'''
You are a precise and efficient artificial intelligence assistant dedicated to real-time understanding of designers' behavior in physical space, specializing in visual analysis. Your task is to analyze the provided images, trigger context, and user voice, then determine the user's current behavioral intention.

Based on your analysis, you must generate a JSON object with the following keys. Your entire response must be ONLY the JSON object, with no introductory text or explanations.

### [Input Context]
- Trigger Types: {trigger_desc}（多种触发同时发生，需综合分析）
- User Voice (Transcript): {transcript_text}

### [JSON Keys]
1. "type": Identify if the target is "overall" (entire product) or "component" (specific part).
2. "label": The specific name of the target. Use "overall" or a specific noun (e.g., "handle", "base").
3. "User Speaking": Transcribe the user's exact words from the transcript. If silent or no transcript, return "".
4. "Behavior description": A single, concise sentence describing the user's interaction based on the trigger types and visual analysis (e.g., "The designer is gripping the handle to evaluate its ergonomic comfort").
5. "User intent": Classify the intent into EXACTLY one of the following strings:
    - "Appearance design": Related to visual form, proportions, materials, or surface details.
    - "Functional concept": Related to functional objectives or improving features.
    - "Structural design": Related to component relationships, layout, or adjustments.
    - "Still-uncertain Idea Exploration": Exploratory attempts or undecided design thoughts.
    - "Design Background supplement": Information related to goals, target users, or scenarios.
'''
    return prompt


# ========== VLM分析函数 ==========

def vlm_chat_mock(image_bytes: str, trigger_types: list[PhysicalTriggerType], transcript_text: str) -> str:
    """
    使用VLM分析物理世界的图像、触发类型列表和语音文本，返回JSON格式的分析结果。

    Args:
        image_bytes: Base64编码的图像
        trigger_types: 物理世界触发类型列表（多种触发同时发生时合并为一次分析）
        transcript_text: 语音转文本结果

    Returns:
        JSON字符串，包含 type, label, User Speaking, Behavior description, User intent
    """
    prompt = get_physical_analysis_prompt(trigger_types, transcript_text)
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type='image/jpeg',
            ),
            prompt
        ]
    )
    return response.text


def vlm_chat_multi_images(
    image_bytes_list: list,
    trigger_types: list[PhysicalTriggerType],
    transcript_text: str
) -> str:
    """
    使用VLM分析多张图像（视频帧序列）、触发类型列表和语音文本，返回JSON格式的分析结果。

    Args:
        image_bytes_list: Base64编码的图像列表（多张视频帧）
        trigger_types: 物理世界触发类型列表
        transcript_text: 语音转文本结果

    Returns:
        JSON字符串，包含 type, label, User Speaking, Behavior description, User intent
    """
    trigger_desc = "、".join(trigger_types) if trigger_types else "无"

    # 多图像分析的 Prompt
    prompt = f'''
You are a precise and efficient artificial intelligence assistant dedicated to real-time understanding of designers' behavior in physical space, specializing in visual analysis.

You are provided with {len(image_bytes_list)} sequential video frames showing the designer's interaction over time. Analyze all frames together to understand the complete context and behavior.

### [Input Context]
- Number of Frames: {len(image_bytes_list)}
- Trigger Types: {trigger_desc}
- User Voice (Transcript): {transcript_text}

### [Your Task]
Based on your analysis of ALL frames, determine the user's current behavioral intention.

Generate a JSON object with the following keys. Your entire response must be ONLY the JSON object:

1. "type": Identify if the target is "overall" (entire product) or "component" (specific part).
2. "label": The specific name of the target. Use "overall" or a specific noun (e.g., "handle", "base").
3. "User Speaking": Transcribe the user's exact words from the transcript. If silent or no transcript, return "".
4. "Behavior description": A single, concise sentence describing the user's interaction based on the video frames and trigger types (e.g., "The designer is gripping the handle to evaluate its ergonomic comfort").
5. "User intent": Classify the intent into EXACTLY one of the following strings:
    - "Appearance design": Related to visual form, proportions, materials, or surface details.
    - "Functional concept": Related to functional objectives or improving features.
    - "Structural design": Related to component relationships, layout, or adjustments.
    - "Still-uncertain Idea Exploration": Exploratory attempts or undecided design thoughts.
    - "Design Background supplement": Information related to goals, target users, or scenarios.
'''

    # 构建 contents：先添加所有图像，再添加 prompt
    contents = []

    for image_bytes in image_bytes_list:
        contents.append(
            types.Part.from_bytes(
                data=image_bytes,
                mime_type='image/jpeg',
            )
        )

    contents.append(prompt)

    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=contents
    )

    return response.text


def vlm_chat_text_only(trigger_types: list[PhysicalTriggerType], transcript_text: str) -> str:
    """
    使用VLM仅分析语音文本（无图像），返回JSON格式的分析结果。

    Args:
        trigger_types: 物理世界触发类型列表
        transcript_text: 语音转文本结果

    Returns:
        JSON字符串，包含 type, label, User Speaking, Behavior description, User intent
    """
    trigger_desc = "、".join(trigger_types) if trigger_types else "无"

    prompt = f'''
You are a precise and efficient artificial intelligence assistant dedicated to real-time understanding of designers' behavior and intent.

You are analyzing a designer's spoken words to understand their current design intention.

### [Input Context]
- Trigger Types: {trigger_desc}
- User Voice (Transcript): {transcript_text}

### [Your Task]
Based on the user's spoken words, determine their current design intention.

Generate a JSON object with the following keys. Your entire response must be ONLY the JSON object:

1. "type": Identify if the target is "overall" (entire product) or "component" (specific part).
2. "label": The specific name of the target. Use "overall" or a specific noun (e.g., "handle", "履带", "底座").
3. "User Speaking": Transcribe the user's exact words from the transcript.
4. "Behavior description": A single, concise sentence describing what the user is doing (e.g., "The designer is describing the overall design concept of a night-time patrol robot").
5. "User intent": Classify the intent into EXACTLY one of the following strings:
    - "Appearance design": Related to visual form, proportions, materials, or surface details.
    - "Functional concept": Related to functional objectives or improving features.
    - "Structural design": Related to component relationships, layout, or adjustments.
    - "Still-uncertain Idea Exploration": Exploratory attempts or undecided design thoughts.
    - "Design Background supplement": Information related to goals, target users, or scenarios.

Only output the JSON object, no other text.
'''

    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[prompt]
    )

    return response.text


# ========== 统一输入接口 ==========

TriggerMode = Literal[0, 1, 2]
# 0: 物理世界触发
# 1: 部件生成（虚拟触发）
# 2: 整体生成（虚拟触发）

MicMode = Literal[0, 1]
# mico=0: 麦克风关闭，语音用于触发VLM分析（自言自语）
#         - 实时语音识别持续运行
#         - 检测到有语音内容时自动触发VLM分析
# mico=1: 麦克风开启，语音用于LLM问答
#         - 实时语音识别持续运行，文本累积
#         - 用户切换到mico=0时，文本传给LLM问答


def process_user_input(
    t: TriggerMode,
    mico: MicMode = 0,
    image_bytes: str = None,
    virtual_json: dict = None,
    trigger_types: list[PhysicalTriggerType] = None
) -> str:
    """
    统一输入接口：根据 t 和 mico 值决定处理方式。

    注意：此函数不再自动录音，语音识别通过 speech.py 持续运行。
    调用此函数时，语音文本已经通过 speech.py 的管理器累积。

    Args:
        t: 触发模式
            - 0: 物理世界触发（需要 image_bytes）
            - 1: 部件生成（虚拟触发，需要 virtual_json）
            - 2: 整体生成（虚拟触发，需要 virtual_json）
        mico: 麦克风模式
            - 0: 自言自语模式，语音用于触发VLM分析
            - 1: 问答模式，语音用于LLM问答（需要用户切换到mico=0后处理）
        image_bytes: Base64编码的图像（t=0 时必需）
        virtual_json: 虚拟界面操作数据（t=1或2 时必需）
        trigger_types: 物理世界触发类型列表（t=0时使用）

    Returns:
        JSON字符串，包含分析结果或LLM回答
    """
    from speech import get_accumulated_text, has_speech_text

    # 获取当前累积的语音文本
    transcript_text = get_accumulated_text()
    print(f"[Input] Mode t={t}, mico={mico}")
    print(f"[Input] Transcript: '{transcript_text}'")

    # 根据 mico 值决定处理方式
    if mico == 1:
        # mico=1: 问答模式
        # 语音累积中，等待用户切换到 mico=0 后处理
        # 此函数调用时如果 mico=1，说明用户还没切换
        print("[Input] mico=1: 语音累积中，等待用户切换到 mico=0 后处理问答")
        return ""

    else:
        # mico=0: 自言自语模式
        # 检查是否有语音内容，有则触发相应处理

        if t == 0:
            # 物理世界触发
            if image_bytes is None:
                print("[Input] Error: image_bytes required for t=0")
                return ""

            # 如果有语音内容，触发VLM分析
            if has_speech_text():
                print("[Input] 检测到语音内容，触发VLM分析")
                # 使用传入的 trigger_types 或默认值
                if trigger_types is None:
                    trigger_types = ["语音输入触发"]
                return vlm_chat_mock(image_bytes, trigger_types, transcript_text)
            else:
                print("[Input] 无语音内容，不触发VLM分析")
                return ""

        elif t == 1:
            # 部件生成（虚拟触发）
            if virtual_json is None:
                print("[Input] Error: virtual_json required for t=1")
                return ""

            trigger_type = "部件生成语音触发"
            if has_speech_text():
                print("[Input] 检测到语音内容，触发部件生成VLM分析")
                return vlm_chat_virtual(virtual_json, trigger_type, transcript_text)
            else:
                print("[Input] 无语音内容，不触发VLM分析")
                return ""

        elif t == 2:
            # 整体生成（虚拟触发）
            if virtual_json is None:
                print("[Input] Error: virtual_json required for t=2")
                return ""

            trigger_type = "整体生成语音触发"
            if has_speech_text():
                print("[Input] 检测到语音内容，触发整体生成VLM分析")
                return vlm_chat_virtual(virtual_json, trigger_type, transcript_text)
            else:
                print("[Input] 无语音内容，不触发VLM分析")
                return ""

        else:
            print(f"[Input] Error: Invalid t value: {t}")
            return ""


def handle_mode_switch_to_mico0(
    previous_mico: MicMode = 1
) -> str:
    """
    处理从 mico=1 切换到 mico=0 的逻辑。

    当用户从问答模式切换回自言自语模式时：
    - 获取累积的语音文本
    - 传给LLM进行问答
    - 清空累积文本，准备新一轮累积

    Args:
        previous_mico: 之前的模式（应该是1，表示从问答模式切换）

    Returns:
        LLM的回答内容
    """
    from speech import get_text_and_clear

    if previous_mico != 1:
        print("[Mode Switch] Error: previous_mico should be 1")
        return ""

    # 获取累积文本并清空
    transcript_text = get_text_and_clear()
    print(f"[Mode Switch] 从 mico=1 切换到 mico=0")
    print(f"[Mode Switch] 累积语音文本: '{transcript_text}'")

    if not transcript_text:
        print("[Mode Switch] 无累积语音文本，不触发LLM问答")
        return ""

    # 调用LLM问答
    print("[Mode Switch] 触发LLM问答")
    return process_user_question(transcript_text)


# ========== 最新 AI 回答记录 ==========
# 用于 mico=1 问答后自动生成图片

_latest_ai_answer = ""


def get_latest_ai_answer() -> str:
    """获取最新一次 AI 回答（用于生成图片）"""
    return _latest_ai_answer


def clear_latest_ai_answer():
    """清空最新 AI 回答记录"""
    global _latest_ai_answer
    _latest_ai_answer = ""
    print("[AI Answer] Cleared latest answer record")


def process_user_question(question: str) -> str:
    """
    处理用户提问（mico=1时调用），让LLM生成回答。

    同时记录最新回答，供后续生成图片使用。

    Args:
        question: 用户的问题文本

    Returns:
        LLM的回答内容
    """
    global _latest_ai_answer

    prompt = f'''
你是一个专业的设计助手，正在协助用户进行产品设计。请回答用户的问题，给出简洁、专业的建议。

用户问题：{question}

请直接回答，不要有多余的开场白。
'''

    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[prompt]
    )

    answer = response.text

    # 记录最新 AI 回答（供后续生成图片使用）
    _latest_ai_answer = answer
    print(f"[LLM Answer] {answer}")
    print("[AI Answer] Recorded for image generation")

    return answer


def vlm_chat_virtual(virtual_json: dict, trigger_type: str, transcript_text: str) -> str:
    """
    使用VLM分析虚拟世界的操作数据、触发类型和语音文本，返回JSON格式的分析结果。

    TODO: 虚拟世界触发类型和virtual_json格式待定，需要用户确认后完善此函数。

    Args:
        virtual_json: 虚拟界面操作数据（格式待定）
        trigger_type: 虚拟世界触发类型（待定）
        transcript_text: 语音转文本结果

    Returns:
        JSON字符串，包含 type, label, User Speaking, Behavior description, User intent
    """
    # TODO: 待用户确定虚拟世界触发类型和virtual_json格式后，完善prompt构建逻辑
    prompt = f'''
You are a precise and efficient artificial intelligence assistant dedicated to real-time understanding of designers' behavior in virtual space, specializing in operation analysis. Your task is to analyze the provided virtual operation data, user voice, and behavior, then determine the user's current behavioral intention.

Based on your analysis, you must generate a JSON object with the following keys. Your entire response must be ONLY the JSON object, with no introductory text or explanations.

### [Input Context]
- Trigger Type: {trigger_type} (TODO: 虚拟世界触发类型待定)
- User Voice (Transcript): {transcript_text}
- Virtual Operation Data: {json.dumps(virtual_json, indent=2, ensure_ascii=False)}

### [JSON Keys]
1. "type": Identify if the target is "overall" (entire product) or "component" (specific part).
2. "label": The specific name of the target. Use "overall" or a specific noun (e.g., "handle", "base").
3. "User Speaking": Transcribe the user's exact words from the transcript. If silent or no transcript, return "".
4. "Behavior description": A single, concise sentence describing the user's interaction in the virtual environment (e.g., "The designer is modifying the handle geometry in the CAD software").
5. "User intent": Classify the intent into EXACTLY one of the following strings:
    - "Appearance design": Related to visual form, proportions, materials, or surface details.
    - "Functional concept": Related to functional objectives or improving features.
    - "Structural design": Related to component relationships, layout, or adjustments.
    - "Still-uncertain Idea Exploration": Exploratory attempts or undecided design thoughts.
    - "Design Background supplement": Information related to goals, target users, or scenarios.
'''
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[prompt]
    )
    return response.text


# ========== 存储函数 ==========

def load_memory_from_json(path: str) -> dict:
    """从JSON文件加载记忆数据库"""
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
            if content.strip():  # 文件有内容
                print(f"[load_memory] Loading from '{path}'...")
                return json.loads(content)
            else:  # 文件为空
                print(f"[load_memory] File '{path}' is empty, starting fresh.")
                return {}
    print(f"[load_memory] No file at '{path}', starting empty.")
    return {}


def save_memory_to_json(memory_db: dict, path: str):
    """保存记忆数据库到JSON文件"""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(memory_db, f, indent=2, default=str, ensure_ascii=False)
    print(f"[save_memory] Saved to '{path}', total: {len(memory_db)} records")


# ========== JSON解析函数 ==========

def extract_and_parse_json(text_with_json: str) -> dict:
    """从VLM返回的文本中提取并解析JSON对象"""
    try:
        start_index = text_with_json.find('{')
        end_index = text_with_json.rfind('}')

        if start_index != -1 and end_index != -1 and end_index > start_index:
            json_string = text_with_json[start_index : end_index + 1]
            data = json.loads(json_string)
            return data
        else:
            print("[parse_json] Error: No valid JSON block found.")
            return None

    except json.JSONDecodeError as e:
        print(f"[parse_json] Error: Invalid JSON format. {e}")
        return None
    except Exception as e:
        print(f"[parse_json] Error: {e}")
        return None


# ========== LLM辅助函数 ==========

def llm_analyze_design_info(user_speaking: str, behavior_desc: str, user_intent: str) -> dict:
    """
    分析用户话语和行为描述，提取外形/功能/结构/设计背景信息。

    Returns:
        {
            "component": "部件名或overall",
            "appearance": [{"description": "外形描述", "status": 1}, ...],
            "function": [{"description": "功能描述", "status": 1}, ...],
            "structure": [{"description": "结构描述", "status": 1}, ...],
            "design_background": "设计背景描述" 或 null
        }
        status: 1=确定，0=不确定
    """
    prompt = f'''
你是一个专业的设计信息提取助手。请分析以下用户话语和行为描述，识别设计相关信息。

[用户话语]
{user_speaking}

[行为描述]
{behavior_desc}

[用户意图]
{user_intent}

请按以下要求分析，并返回JSON格式结果：

1. 部件识别：识别用户谈论的是哪个部件，返回部件名称；如果是整体则返回"overall"
2. 外形设计：是否提到产品外形（外观、尺寸、材质、颜色、表面纹理等）？如果有，提取具体描述。
3. 功能设计：是否提到产品功能（功能目标、使用方式、性能要求等）？如果有，提取具体描述。
4. 结构设计：是否提到产品结构（部件关系、连接方式、布局等）？如果有，提取具体描述。
5. 设计背景：是否提到设计背景（设计目标、目标用户、使用场景等）？如果有，提取具体内容。
6. 不确定检测：如果用户话语中出现"不确定"、"我想想"等明显的不确定表达，则该描述的status设为0，否则设为1。

返回JSON格式：
{{
    "component": "部件名称或overall",
    "appearance": [
        {{"description": "外形描述内容", "status": 1}}
    ],
    "function": [
        {{"description": "功能描述内容", "status": 1}}
    ],
    "structure": [
        {{"description": "结构描述内容", "status": 1}}
    ],
    "design_background": "设计背景内容或null"
}}

注意：
- 如果某类信息没有提到，对应字段返回空列表 [] 或 null
- component 可以是 "overall"（整体）或具体部件名称（如 "handle", "base"）
- status: 1=确定，0=不确定（根据用户是否有不确定表达判断）
- 只返回JSON，不要其他解释
'''
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[prompt]
    )

    result = extract_and_parse_json(response.text)
    if result:
        print(f"[llm_analyze] Component: {result.get('component', 'unknown')}")
        print(f"[llm_analyze] Appearance: {result.get('appearance', [])}")
        print(f"[llm_analyze] Function: {result.get('function', [])}")
        print(f"[llm_analyze] Structure: {result.get('structure', [])}")
        print(f"[llm_analyze] Background: {result.get('design_background', None)}")
    else:
        print("[llm_analyze] Failed to parse JSON, using empty result")
        result = {
            "component": "unknown",
            "appearance": [],
            "function": [],
            "structure": [],
            "design_background": None
        }
    return result


def llm_merge_names(name1: str, name2: str) -> tuple:
    """使用LLM判断两个名称是否指代同一事物，返回合并后的名称"""
    prompt = f'''请判断以下两个部件名称是否指代同一个部件。如果是同一个部件，请在这两个称呼中返回一个更准确、完整的名称；如果不是同一个部件，请返回"不同"。

部件名称1：{name1}
部件名称2：{name2}

只输出结果，不要解释。如果相同，输出合并后的名称；如果不同，输出"不同"。'''

    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[prompt]
    )
    result = response.text.strip()

    if result == "不同" or result.lower() == "different":
        print(f"[llm_merge] '{name1}' and '{name2}' are different")
        return name1, False
    else:
        print(f"[llm_merge] '{name1}' and '{name2}' merged to '{result}'")
        return result, True


def llm_merge_descriptions(original: str, new: str) -> str:
    """使用LLM合并两个描述内容"""
    prompt = f'''请你检查这两块内容是否存在矛盾，如存在矛盾，以最新的描述为主，若不存在矛盾，则将这两块内容进行合并更新，生成一个完整、准确、简洁的描述。只输出合并后的内容，不要添加任何解释。

原有内容：{original}
新内容：{new}

合并后的内容：'''

    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[prompt]
    )
    merged = response.text.strip()
    print(f"[llm_merge_desc] Merged: '{merged[:50]}...'")
    return merged