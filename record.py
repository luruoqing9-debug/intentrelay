"""
record.py - 记录处理模块
包含：图像/文本编码、VLM分析、图像处理、存储工具
"""

import os

# 设置 HuggingFace 镜像（解决国内连接问题）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import io
import json
import base64
import numpy as np
from PIL import Image
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


# ========== VLM分析函数 ==========

def vlm_chat_mock(image_bytes: str, context: str) -> str:
    """使用VLM分析物理世界的图像和用户意图，返回JSON格式的分析结果"""
    prompt = f'''
You are a precise and efficient artificial intelligence assistant dedicated to real-time understanding of designers' behavior in physical space, specializing in visual analysis. Your task is to analyze the provided images, user voice, and behavior, and then determine the user's current behavioral intention.
Based on your analysis, you must generate a JSON object with the following three keys. Your entire response must be ONLY the JSON object, with no introductory text or explanations.

### [JSON Keys]
1. "type": Identify if the target is "overall" (entire product) or "component" (specific part).
2. "label": The specific name of the target. Use "overall" or a specific noun (e.g., "handle", "base").
3. "User Speaking": Transcribe the user's exact words. If silent, return "".
4. "Behavior description": A single, concise sentence describing the user's interaction (e.g., "The designer is gripping the handle to evaluate its ergonomic comfort").
5. "User intent": Classify the intent into EXACTLY one of the following strings:
    - "Appearance design": Related to visual form, proportions, materials, or surface details.
    - "Functional concept": Related to functional objectives or improving features.
    - "Structural design": Related to component relationships, layout, or adjustments.
    - "Still-uncertain Idea Exploration": Exploratory attempts or undecided design thoughts.
    - "Design Background supplement": Information related to goals, target users, or scenarios.
[User's Statement / Context]
{context}
'''
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


def vlm_chat_virtual(virtual_json: dict, context: str) -> str:
    """使用VLM分析虚拟世界的JSON数据和用户意图，返回JSON格式的分析结果"""
    prompt = f'''
You are a precise and efficient artificial intelligence assistant dedicated to real-time understanding of designers' behavior in virtual space, specializing in operation analysis. Your task is to analyze the provided virtual operation data, user voice, and behavior, and then determine the user's current behavioral intention.
Based on your analysis, you must generate a JSON object with the following three keys. Your entire response must be ONLY the JSON object, with no introductory text or explanations.

### [JSON Keys]
1. "type": Identify if the target is "overall" (entire product) or "component" (specific part).
2. "label": The specific name of the target. Use "overall" or a specific noun (e.g., "handle", "base").
3. "User Speaking": Transcribe the user's exact words. If silent, return "".
4. "Behavior description": A single, concise sentence describing the user's interaction (e.g., "The designer is modifying the handle geometry in the CAD software").
5. "User intent": Classify the intent into EXACTLY one of the following strings:
    - "Appearance design": Related to visual form, proportions, materials, or surface details.
    - "Functional concept": Related to functional objectives or improving features.
    - "Structural design": Related to component relationships, layout, or adjustments.
    - "Still-uncertain Idea Exploration": Exploratory attempts or undecided design thoughts.
    - "Design Background supplement": Information related to goals, target users, or scenarios.
[Virtual Operation Data]
{json.dumps(virtual_json, indent=2, ensure_ascii=False)}
[User's Statement / Context]
{context}
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
            print(f"[load_memory] Loading from '{path}'...")
            return json.load(f)
    print(f"[load_memory] No file at '{path}', starting empty.")
    return {}


def save_memory_to_json(memory_db: dict, path: str):
    """保存记忆数据库到JSON文件"""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(memory_db, f, indent=2, default=str)
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