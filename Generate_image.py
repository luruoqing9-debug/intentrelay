"""
Generate_image.py - ComfyUI API 图片生成模块
调用 Gemini API (gemini-2.5-flash-image) 实现图片生成
支持两种模式：单图生成（部件）和多图生成（整体）

文件夹说明：
├── original_image/      → 参考图（始终只有一张，上传新图时自动替换旧的）
│                          部件生成：部件参考图
│                          整体生成：粗糙结构参考图（添加到图片数组末尾）
├── Operated_image/      → 操作记录图（临时，VLM分析后清空）
├── generated_images/    → 生成的图片（临时，存入记忆后移动到 processed_images）
├── processed_images/    → 已存记忆的图片（永久保留）

使用流程：
1. 前端上传参考图到 original_image/（调用 /upload_reference_image）
2. 系统自动从 original_image/ 读取参考图进行生成
3. 生成的图片输出到 generated_images/ 文件夹
4. 存入记忆后，图片移动到 processed_images/ 文件夹

关键函数：
- get_original_image(): 获取 original_image/ 中的唯一图片
- update_original_image(new_path): 更新 original_image/ 中的图片（删除旧的）
"""

import sys
import os
import json
import time
import uuid
import base64
import requests
import websocket
import shutil
from typing import Literal, Optional, List, Dict, Union
from pathlib import Path

# 禁用警告
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
import warnings
warnings.filterwarnings('ignore')

# 导入 generate.py 的提示词生成函数
from generate import (
    generate_component_prompt,
    generate_overall_prompt,
    process_generate_request
)


# ========== 图片文件夹处理 ==========

# original_image 文件夹路径
ORIGINAL_IMAGE_DIR = os.path.join(os.path.dirname(__file__), "original_image")
# processed_images 文件夹路径
PROCESSED_IMAGES_DIR = os.path.join(os.path.dirname(__file__), "processed_images")


def get_original_image() -> str:
    """
    从 original_image/ 文件夹获取唯一的参考图片

    该文件夹始终只有一张图片（用户放入新图片时会替换旧的）

    Returns:
        图片路径（如果文件夹为空返回 None）
    """
    supported_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp']

    if not os.path.exists(ORIGINAL_IMAGE_DIR):
        os.makedirs(ORIGINAL_IMAGE_DIR)
        print(f"[original_image] Created folder: {ORIGINAL_IMAGE_DIR}")
        return None

    # 获取所有图片文件
    image_files = []
    for f in os.listdir(ORIGINAL_IMAGE_DIR):
        ext = os.path.splitext(f)[1].lower()
        if ext in supported_extensions:
            image_files.append(os.path.join(ORIGINAL_IMAGE_DIR, f))

    if len(image_files) == 0:
        print(f"[original_image] No image found in folder")
        return None

    # 返回找到的第一张图片（文件夹中应该只有一张）
    image_path = image_files[0]
    print(f"[original_image] Found reference image: {image_path}")
    return image_path


def get_processed_component_images(exclude_overall: bool = True) -> List[str]:
    """
    从 processed_images/ 文件夹获取所有部件图片（排除 overall.png）

    用于整体生成时自动读取所有已生成的部件图片

    Args:
        exclude_overall: 是否排除 overall.png（默认 True）

    Returns:
        部件图片路径列表（按文件名排序）
    """
    supported_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp']
    excluded_names = ['overall.png', 'overall.jpg', 'overall.jpeg', 'overall.bmp', 'overall.gif', 'overall.webp']

    if not os.path.exists(PROCESSED_IMAGES_DIR):
        os.makedirs(PROCESSED_IMAGES_DIR)
        print(f"[processed_images] Created folder: {PROCESSED_IMAGES_DIR}")
        return []

    # 获取所有图片文件
    image_files = []
    for f in os.listdir(PROCESSED_IMAGES_DIR):
        ext = os.path.splitext(f)[1].lower()
        if ext in supported_extensions:
            # 排除 overall 图片
            if exclude_overall and f.lower() in [name.lower() for name in excluded_names]:
                continue
            image_files.append(os.path.join(PROCESSED_IMAGES_DIR, f))

    # 按文件名排序
    image_files.sort()

    print(f"[processed_images] Found {len(image_files)} component images")
    for i, path in enumerate(image_files):
        print(f"  [{i}] {os.path.basename(path)}")

    return image_files


def update_original_image(new_image_path: str) -> str:
    """
    更新 original_image/ 文件夹中的图片（删除旧的，放入新的）

    Args:
        new_image_path: 新图片的路径（前端上传的临时路径或外部路径）

    Returns:
        新图片在 original_image/ 中的最终路径
    """
    supported_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp']

    # 确保文件夹存在
    if not os.path.exists(ORIGINAL_IMAGE_DIR):
        os.makedirs(ORIGINAL_IMAGE_DIR)

    # 删除文件夹中所有现有图片
    deleted_count = 0
    for f in os.listdir(ORIGINAL_IMAGE_DIR):
        ext = os.path.splitext(f)[1].lower()
        if ext in supported_extensions:
            old_path = os.path.join(ORIGINAL_IMAGE_DIR, f)
            try:
                os.remove(old_path)
                deleted_count += 1
                print(f"[original_image] Deleted old image: {old_path}")
            except Exception as e:
                print(f"[original_image] Failed to delete {old_path}: {e}")

    # 获取新图片的文件名和扩展名
    new_filename = os.path.basename(new_image_path)
    new_ext = os.path.splitext(new_filename)[1].lower()

    # 如果不是图片格式，报错
    if new_ext not in supported_extensions:
        raise ValueError(f"Unsupported image format: {new_ext}")

    # 复制新图片到 original_image/
    final_path = os.path.join(ORIGINAL_IMAGE_DIR, new_filename)
    shutil.copy(new_image_path, final_path)
    print(f"[original_image] Added new image: {final_path} (deleted {deleted_count} old images)")

    return final_path


def get_images_from_folder(folder_path: str, max_images: int = 9) -> List[str]:
    """
    从文件夹中按顺序读取图片（最多 max_images 张）

    Args:
        folder_path: 图片文件夹路径
        max_images: 最大图片数量（默认9张）

    Returns:
        图片路径列表（按文件名排序）
    """
    supported_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp']
    # 排除的文件名（占位图）
    excluded_names = ['black.png', 'white_placeholder.png', 'white.png']

    if not os.path.exists(folder_path):
        print(f"[Warning] Folder not found: {folder_path}")
        return []

    # 获取所有图片文件
    image_files = []
    for f in os.listdir(folder_path):
        ext = os.path.splitext(f)[1].lower()
        if ext in supported_extensions and f not in excluded_names:
            image_files.append(os.path.join(folder_path, f))

    # 按文件名排序
    image_files.sort()

    # 限制数量
    image_files = image_files[:max_images]

    print(f"[Folder] Found {len(image_files)} images in {folder_path}")
    return image_files


def create_white_placeholder(output_dir: str = None) -> str:
    """
    创建白色占位图片

    Args:
        output_dir: 输出目录（默认为 ComfyUI input 目录）

    Returns:
        白色图片路径
    """
    from PIL import Image

    if output_dir is None:
        output_dir = "d:/ComfyUI/input"

    white_path = os.path.join(output_dir, "white_placeholder.png")

    # 如果已存在，直接返回
    if os.path.exists(white_path):
        return white_path

    # 创建 512x512 白色图片
    img = Image.new('RGB', (512, 512), color='white')
    img.save(white_path)
    print(f"[Placeholder] Created white image: {white_path}")

    return white_path


def prepare_images_with_padding(image_paths: List[str], total_slots: int = 9) -> List[str]:
    """
    准备图片列表，不足的部分用白色填充

    Args:
        image_paths: 实际图片路径列表
        total_slots: 总图片槽位数量（默认9）

    Returns:
        填充后的图片路径列表（长度为 total_slots）
    """
    # 创建白色占位图
    white_placeholder = create_white_placeholder()

    # 填充到指定数量
    result = image_paths[:total_slots]  # 截断超出部分

    while len(result) < total_slots:
        result.append(white_placeholder)

    print(f"[Padding] Prepared {len(result)} images ({len(image_paths)} actual + {total_slots - len(image_paths)} white padding)")
    return result


# ========== ComfyUI API 配置 ==========

COMFYUI_URL = "http://localhost:8000"
API_CLIENT_ID = "comfyui-54ae45038ffb9ec10eb61cf40f84c436623b70e64a5cbcd9bb4cc864a099c7bd"
COMFY_API_KEY = "comfyui-54ae45038ffb9ec10eb61cf40f84c436623b70e64a5cbcd9bb4cc864a099c7bd"  # ComfyAPI API Key

# 工作流模板路径
COMPONENT_WORKFLOW_PATH = "D:/ComfyUI/user/default/workflows/Component_generation.json"  # 部件生成
OVERALL_WORKFLOW_PATH = "D:/ComfyUI/user/default/workflows/google_Gemini_image.json"     # 整体生成


# ========== ComfyUI API 客户端 ==========

class ComfyUIClient:
    """ComfyUI API 客户端"""

    def __init__(self, base_url: str = COMFYUI_URL, client_id: str = API_CLIENT_ID, api_key: str = COMFY_API_KEY):
        self.base_url = base_url
        self.client_id = client_id
        self.api_key = api_key
        self.ws_url = f"ws://{base_url.replace('http://', '').replace('https://', '')}/ws?clientId={client_id}"

    def upload_image(self, image_path: str, subfolder: str = "", overwrite: bool = True) -> dict:
        """
        上传图片到 ComfyUI input 目录

        Args:
            image_path: 本地图片路径
            subfolder: 子目录
            overwrite: 是否覆盖同名文件

        Returns:
            {"name": "文件名", "subfolder": "子目录", "type": "input"}
        """
        url = f"{self.base_url}/upload/image"

        with open(image_path, 'rb') as f:
            files = {'image': (os.path.basename(image_path), f, 'image/png')}
            data = {'overwrite': 'true' if overwrite else 'false', 'subfolder': subfolder}

            response = requests.post(url, files=files, data=data)

        if response.status_code != 200:
            raise Exception(f"Upload failed: {response.text}")

        result = response.json()
        print(f"[ComfyUI] Uploaded image: {result.get('name')}")
        return result

    def queue_prompt(self, workflow: dict, use_api_format: bool = True) -> str:
        """
        提交工作流到队列

        Args:
            workflow: 工作流 JSON（可以是 UI 或 API 格式）
            use_api_format: 是否转换为 API 格式（默认 True）

        Returns:
            prompt_id
        """
        url = f"{self.base_url}/prompt"

        # 如果是 UI 格式，转换为 API 格式
        if use_api_format and 'nodes' in workflow:
            workflow = convert_ui_to_api_workflow(workflow)

        data = {
            "prompt": workflow,
            "client_id": self.client_id,
            "extra_data": {
                "api_key_comfy_org": self.api_key  # 添加 API Key 认证
            }
        }

        response = requests.post(url, json=data)

        if response.status_code != 200:
            raise Exception(f"Queue prompt failed: {response.text}")

        result = response.json()
        prompt_id = result.get('prompt_id')
        print(f"[ComfyUI] Queued prompt: {prompt_id}")
        return prompt_id

    def wait_for_completion(self, prompt_id: str, timeout: int = 300) -> dict:
        """
        等待工作流完成（通过 WebSocket）

        Args:
            prompt_id: 提示词 ID
            timeout: 超时时间（秒）

        Returns:
            输出节点信息
        """
        ws = websocket.create_connection(self.ws_url)
        outputs = {}
        start_time = time.time()

        try:
            while True:
                if time.time() - start_time > timeout:
                    raise Exception(f"Timeout waiting for completion: {prompt_id}")

                try:
                    message = ws.recv()
                    if not message:
                        continue

                    # 处理可能的二进制消息
                    if isinstance(message, bytes):
                        message = message.decode('utf-8', errors='ignore')

                    data = json.loads(message)

                    # 监听执行完成事件
                    if data.get('type') == 'executing':
                        exec_data = data.get('data', {})
                        if exec_data.get('prompt_id') == prompt_id:
                            if exec_data.get('node') is None:
                                # 执行完成
                                print(f"[ComfyUI] Execution completed: {prompt_id}")
                                break

                    # 监听执行开始事件
                    if data.get('type') == 'execution_start':
                        print(f"[ComfyUI] Execution started")

                    # 监听进度事件
                    if data.get('type') == 'progress':
                        progress_data = data.get('data', {})
                        current = progress_data.get('value', 0)
                        total = progress_data.get('max', 0)
                        if total > 0:
                            print(f"[ComfyUI] Progress: {current}/{total}")

                except json.JSONDecodeError:
                    continue
                except UnicodeDecodeError:
                    continue

        finally:
            ws.close()

        return outputs

    def get_history(self, prompt_id: str) -> dict:
        """
        获取执行历史

        Args:
            prompt_id: 提示词 ID

        Returns:
            历史记录
        """
        url = f"{self.base_url}/history/{prompt_id}"
        response = requests.get(url)

        if response.status_code != 200:
            raise Exception(f"Get history failed: {response.text}")

        return response.json()

    def get_output_images(self, prompt_id: str, output_dir: str = None, save_name: str = None) -> List[str]:
        """
        获取生成的图片并保存到本地

        Args:
            prompt_id: 提示词 ID
            output_dir: 输出目录（默认为项目目录下的 generated_images 文件夹）
            save_name: 保存的文件名（不含扩展名，如 "把手" 或 "overall"）

        Returns:
            保存的图片路径列表
        """
        # 默认输出目录为 generated_images 文件夹
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(__file__), "generated_images")

        # 确保目录存在
        os.makedirs(output_dir, exist_ok=True)

        history = self.get_history(prompt_id)
        prompt_history = history.get(prompt_id, {})
        outputs = prompt_history.get('outputs', {})

        saved_paths = []

        for node_id, node_output in outputs.items():
            if 'images' in node_output:
                for idx, image_info in enumerate(node_output['images']):
                    filename = image_info.get('filename')
                    subfolder = image_info.get('subfolder', '')

                    # 获取文件扩展名
                    ext = os.path.splitext(filename)[1] or '.png'

                    # 下载图片
                    url = f"{self.base_url}/view"
                    params = {
                        'filename': filename,
                        'subfolder': subfolder,
                        'type': 'output'
                    }

                    response = requests.get(url, params=params)
                    if response.status_code == 200:
                        # 确定保存的文件名
                        if save_name:
                            # 如果指定了保存名称，使用该名称（多张图片时加序号）
                            if len(node_output['images']) > 1:
                                final_name = f"{save_name}_{idx}{ext}"
                            else:
                                final_name = f"{save_name}{ext}"
                        else:
                            # 否则使用原文件名加前缀
                            final_name = f"generated_{filename}"

                        local_path = os.path.join(output_dir, final_name)
                        with open(local_path, 'wb') as f:
                            f.write(response.content)
                        saved_paths.append(local_path)
                        print(f"[ComfyUI] Saved image: {local_path}")

        return saved_paths


# ========== 辅助函数 ==========

def translate_to_english(text: str) -> str:
    """
    将中文文本翻译成英文（简单检测，如果包含中文则尝试翻译）

    Args:
        text: 输入文本（可能包含中文）

    Returns:
        英文文本
    """
    # 检测是否包含中文
    has_chinese = any('\u4e00' <= c <= '\u9fff' for c in text)

    if not has_chinese:
        return text

    # 使用 Google Translate 免费 API
    try:
        import urllib.parse
        import urllib.request

        url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=zh&tl=en&dt=t&q={urllib.parse.quote(text)}"

        with urllib.request.urlopen(url, timeout=10) as response:
            result = response.read().decode('utf-8')
            # 解析返回结果
            import json
            data = json.loads(result)
            translated = ''.join([item[0] for item in data[0] if item[0]])
            print(f"[Translate] {text[:50]}... -> {translated[:50]}...")
            return translated
    except Exception as e:
        print(f"[Translate] Failed: {e}, using original text")
        return text


# ========== 工作流模板 ==========

def convert_ui_to_api_workflow(ui_workflow: dict) -> dict:
    """
    将 UI 格式的工作流转换为 API 格式

    UI 格式: {"nodes": [...], "links": [...]}
    API 格式: {"node_id": {"class_type": "...", "inputs": {...}}}
    """
    api_workflow = {}

    # 非执行节点类型（注释、说明等）
    SKIP_NODE_TYPES = ['MarkdownNote', 'Note', 'Reroute']

    # 构建 links 映射：link_id -> (from_node, from_slot, to_node, to_slot)
    links_map = {}
    for link in ui_workflow.get('links', []):
        link_id = link[0]
        from_node = link[1]
        from_slot = link[2]
        to_node = link[3]
        to_slot = link[4]
        links_map[link_id] = (from_node, from_slot, to_node, to_slot)

    # 构建每个节点的 API 格式
    for node in ui_workflow.get('nodes', []):
        node_id = str(node['id'])
        class_type = node['type']

        # 跳过非执行节点
        if class_type in SKIP_NODE_TYPES:
            continue

        # 构建 inputs
        inputs = {}

        # 从 widgets_values 获取参数值
        widgets_values = node.get('widgets_values', [])

        # 获取节点的输入定义
        input_defs = node.get('inputs', [])

        # 处理 widgets（非连接的输入）
        widget_idx = 0
        for input_def in input_defs:
            input_name = input_def.get('name')
            if input_name not in ['upload']:  # 跳过特殊输入
                # 检查是否有连接
                link_id = input_def.get('link')
                if link_id is not None and link_id in links_map:
                    # 有连接，使用连接的输出
                    from_node, from_slot, _, _ = links_map[link_id]
                    inputs[input_name] = [str(from_node), from_slot]
                elif widget_idx < len(widgets_values):
                    # 无连接，使用 widgets_values
                    inputs[input_name] = widgets_values[widget_idx]
                    widget_idx += 1

        # 特殊处理：GeminiImageNode 需要正确的参数顺序
        # widgets_values 格式: [prompt, model, seed, randomize, aspect_ratio, response_modalities, system_prompt]
        if class_type == 'GeminiImageNode':
            if len(widgets_values) >= 7:
                inputs['prompt'] = widgets_values[0]
                inputs['model'] = widgets_values[1]
                inputs['seed'] = widgets_values[2]
                # inputs['randomize'] = widgets_values[3]  # 这个参数不需要传
                inputs['aspect_ratio'] = widgets_values[4]
                inputs['response_modalities'] = widgets_values[5]
                inputs['system_prompt'] = widgets_values[6]

        # 特殊处理：LoadImage
        if class_type == 'LoadImage':
            if len(widgets_values) >= 2:
                inputs['image'] = widgets_values[0]

        # 特殊处理：SaveImage
        if class_type == 'SaveImage':
            if len(widgets_values) >= 1:
                inputs['filename_prefix'] = widgets_values[0]

        api_workflow[node_id] = {
            'class_type': class_type,
            'inputs': inputs
        }

    return api_workflow


def load_workflow_template(workflow_path: str = OVERALL_WORKFLOW_PATH) -> dict:
    """
    加载工作流模板

    Args:
        workflow_path: 工作流 JSON 文件路径

    Returns:
        工作流字典
    """
    if not os.path.exists(workflow_path):
        raise FileNotFoundError(f"Workflow not found: {workflow_path}")

    with open(workflow_path, 'r', encoding='utf-8') as f:
        workflow = json.load(f)

    print(f"[ComfyUI] Loaded workflow: {workflow_path}")
    return workflow


def prepare_workflow_component(
    workflow: dict,
    prompt: str,
    image_path: str,
    seed: int = None
) -> dict:
    """
    准备部件生成工作流（单图模式）

    Args:
        workflow: 工作流模板（Component_generation.json）
        prompt: 图像生成提示词
        image_path: 输入图片路径
        seed: 种子值（可选）

    Returns:
        修改后的工作流（API 格式）
    """
    # 上传图片
    client = ComfyUIClient()
    upload_result = client.upload_image(image_path)
    uploaded_filename = upload_result.get('name')

    # 翻译提示词为英文
    english_prompt = translate_to_english(prompt)

    # 直接构建 API 格式工作流
    # 节点 ID: 2 (LoadImage), 5 (GeminiImageNode), 30 (SaveImage)
    api_workflow = {
        "2": {
            "class_type": "LoadImage",
            "inputs": {
                "image": uploaded_filename
            }
        },
        "5": {
            "class_type": "GeminiImageNode",
            "inputs": {
                "images": ["2", 0],
                "prompt": english_prompt,
                "model": "gemini-2.5-flash-image-preview",
                "seed": seed if seed is not None else 42,
                "aspect_ratio": "auto",
                "response_modalities": "IMAGE+TEXT",
                "system_prompt": "You are an expert image-generation engine. You must ALWAYS produce an image."
            }
        },
        "30": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["5", 0],
                "filename_prefix": "ComfyUI"
            }
        }
    }

    return api_workflow


def prepare_workflow_overall(
    workflow: dict,
    prompt: str,
    image_paths: List[str],
    seed: int = None
) -> dict:
    """
    准备整体生成工作流（多图模式）

    Args:
        workflow: 工作流模板（google_Gemini_image.json）
        prompt: 图像生成提示词（可包含 [@Image N] 引用）
        image_paths: 输入图片路径列表（最多9张）
        seed: 种子值（可选）

    Returns:
        修改后的工作流（API 格式）
    """
    if len(image_paths) > 9:
        raise ValueError("最多支持9张输入图片")

    # 上传所有图片
    client = ComfyUIClient()
    uploaded_filenames = []

    for i, image_path in enumerate(image_paths):
        upload_result = client.upload_image(image_path)
        uploaded_filenames.append(upload_result.get('name'))

    # 翻译提示词为英文
    english_prompt = translate_to_english(prompt)

    # LoadImage 节点 ID 映射（按顺序）
    load_image_ids = [2, 33, 34, 36, 39, 40, 41, 42, 43]

    # 构建 API 格式工作流
    api_workflow = {}

    # 创建 LoadImage 节点
    for i, node_id in enumerate(load_image_ids):
        if i < len(uploaded_filenames):
            api_workflow[str(node_id)] = {
                "class_type": "LoadImage",
                "inputs": {
                    "image": uploaded_filenames[i]
                }
            }
        else:
            # 未使用的节点用白色占位图
            api_workflow[str(node_id)] = {
                "class_type": "LoadImage",
                "inputs": {
                    "image": "white_placeholder.png"
                }
            }

    # 创建 BatchImagesNode (节点 ID 35)
    # 输入参数名格式: images.image0, images.image1, ...
    batch_inputs = {}
    for i in range(9):
        batch_inputs[f"images.image{i}"] = [str(load_image_ids[i]), 0]

    api_workflow["35"] = {
        "class_type": "BatchImagesNode",
        "inputs": batch_inputs
    }

    # 创建 GeminiImageNode (节点 ID 5)
    api_workflow["5"] = {
        "class_type": "GeminiImageNode",
        "inputs": {
            "images": ["35", 0],
            "prompt": english_prompt,
            "model": "gemini-2.5-flash-image",
            "seed": seed if seed is not None else 42,
            "aspect_ratio": "auto",
            "response_modalities": "IMAGE+TEXT",
            "system_prompt": "You are an expert image-generation engine. You must ALWAYS produce an image."
        }
    }

    # 创建 SaveImage (节点 ID 30)
    api_workflow["30"] = {
        "class_type": "SaveImage",
        "inputs": {
            "images": ["5", 0],
            "filename_prefix": "ComfyUI"
        }
    }

    return api_workflow


# ========== 统一生成接口 ==========

GenerateMode = Literal[1, 2]
# 1: 部件生成（单图）
# 2: 整体生成（多图）


def generate_component_image(
    prompt: str,
    image_path: str,
    workflow_path: str = COMPONENT_WORKFLOW_PATH,
    seed: int = None,
    output_dir: str = None,
    save_name: str = None,
    timeout: int = 300
) -> List[str]:
    """
    部件图片生成：文本 + 1张参考图 → 新图片

    Args:
        prompt: 图像生成提示词
        image_path: 输入图片路径（部件参考图）
        workflow_path: 工作流模板路径（默认为 Component_generation.json）
        seed: 种子值
        output_dir: 输出目录（默认为 generated_images 文件夹）
        save_name: 保存的文件名（不含扩展名，如 "把手"）
        timeout: 超时时间（秒）

    Returns:
        生成的图片路径列表
    """
    print(f"\n=== 部件生成模式 ===")
    print(f"[Input] Prompt: {prompt[:100]}...")
    print(f"[Input] Image: {image_path}")

    # 默认输出目录
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "generated_images")

    # 加载工作流
    workflow = load_workflow_template(workflow_path)

    # 准备工作流（已转换为 API 格式）
    prepared_workflow = prepare_workflow_component(workflow, prompt, image_path, seed)

    # 提交执行
    client = ComfyUIClient()
    prompt_id = client.queue_prompt(prepared_workflow, use_api_format=False)

    # 等待完成
    client.wait_for_completion(prompt_id, timeout)

    # 获取输出图片
    saved_paths = client.get_output_images(prompt_id, output_dir, save_name)

    return saved_paths


def generate_overall_image(
    prompt: str,
    image_paths: List[str],
    workflow_path: str = OVERALL_WORKFLOW_PATH,
    seed: int = None,
    output_dir: str = None,
    save_name: str = "overall",
    timeout: int = 300
) -> List[str]:
    """
    整体图片生成：文本 + 多张部件图 → 整体设计图

    Args:
        prompt: 图像生成提示词（可包含 [@Image N] 引用）
        image_paths: 输入图片路径列表（最多10张部件图）
        workflow_path: 工作流模板路径（默认为 google_Gemini_image.json）
        seed: 种子值
        output_dir: 输出目录（默认为 generated_images 文件夹）
        save_name: 保存的文件名（默认为 "overall"）
        timeout: 超时时间（秒）

    Returns:
        生成的图片路径列表
    """
    print(f"\n=== 整体生成模式 ===")
    print(f"[Input] Prompt: {prompt[:100]}...")
    print(f"[Input] Images: {len(image_paths)} 张")

    # 默认输出目录
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "generated_images")

    # 加载工作流
    workflow = load_workflow_template(workflow_path)

    # 准备工作流（已转换为 API 格式）
    prepared_workflow = prepare_workflow_overall(workflow, prompt, image_paths, seed)

    # 提交执行
    client = ComfyUIClient()
    prompt_id = client.queue_prompt(prepared_workflow, use_api_format=False)

    # 等待完成
    client.wait_for_completion(prompt_id, timeout)

    # 获取输出图片
    saved_paths = client.get_output_images(prompt_id, output_dir, save_name)

    return saved_paths


def generate_image(
    mode: GenerateMode,
    prompt: str,
    image_paths: Union[str, List[str]],
    workflow_path: str = None,
    seed: int = None,
    output_dir: str = None,
    save_name: str = None,
    timeout: int = 300
) -> List[str]:
    """
    统一图片生成接口

    Args:
        mode: 生成模式（1=部件生成，2=整体生成）
        prompt: 图像生成提示词
        image_paths: 图片路径（部件模式为单张，整体模式为多张）
        workflow_path: 工作流模板路径（默认自动选择）
        seed: 种子值
        output_dir: 输出目录（默认为 generated_images 文件夹）
        save_name: 保存的文件名（部件模式为部件名，整体模式默认为 "overall"）
        timeout: 超时时间（秒）

    Returns:
        生成的图片路径列表
    """
    if mode == 1:
        # 部件生成模式
        if isinstance(image_paths, list):
            if len(image_paths) == 0:
                raise ValueError("部件生成需要提供参考图片")
            image_paths = image_paths[0]

        if workflow_path is None:
            workflow_path = COMPONENT_WORKFLOW_PATH

        return generate_component_image(
            prompt=prompt,
            image_path=image_paths,
            workflow_path=workflow_path,
            seed=seed,
            output_dir=output_dir,
            save_name=save_name,
            timeout=timeout
        )

    elif mode == 2:
        # 整体生成模式
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        if workflow_path is None:
            workflow_path = OVERALL_WORKFLOW_PATH

        # 整体模式默认 save_name 为 "overall"
        if save_name is None:
            save_name = "overall"

        return generate_overall_image(
            prompt=prompt,
            image_paths=image_paths,
            workflow_path=workflow_path,
            seed=seed,
            output_dir=output_dir,
            save_name=save_name,
            timeout=timeout
        )

    else:
        raise ValueError(f"无效的生成模式: {mode}")


# ========== 从文件夹生成 ==========

def generate_image_from_folder(
    prompt: str,
    folder_path: str,
    workflow_path: str = OVERALL_WORKFLOW_PATH,
    seed: int = None,
    output_dir: str = None,
    save_name: str = "overall",
    timeout: int = 300,
    max_images: int = 9
) -> List[str]:
    """
    从文件夹读取图片并生成整体设计图（自动填充白色占位图）

    Args:
        prompt: 图像生成提示词（可包含 [@Image N] 引用）
        folder_path: 图片文件夹路径（自动按顺序读取）
        workflow_path: 工作流模板路径
        seed: 种子值
        output_dir: 输出目录（默认为 generated_images 文件夹）
        save_name: 保存的文件名（默认为 "overall"）
        timeout: 超时时间（秒）
        max_images: 最大图片数量（默认9张，不足用白色填充）

    Returns:
        生成的图片路径列表
    """
    print(f"\n=== 从文件夹生成 ===")
    print(f"[Input] Folder: {folder_path}")
    print(f"[Input] Prompt: {prompt[:100]}...")

    # 从文件夹读取图片
    image_paths = get_images_from_folder(folder_path, max_images)

    if len(image_paths) == 0:
        print("[Warning] No images found in folder, using white placeholders only")
        image_paths = []

    # 用白色填充到 max_images 张
    padded_images = prepare_images_with_padding(image_paths, max_images)

    print(f"[Input] Total images (with padding): {len(padded_images)}")

    # 调用整体生成
    return generate_overall_image(
        prompt=prompt,
        image_paths=padded_images,
        workflow_path=workflow_path,
        seed=seed,
        output_dir=output_dir,
        save_name=save_name,
        timeout=timeout
    )


# ========== 提示词生成 + 图片生成统一接口 ==========

def generate_component_with_prompt(
    component_name: str,
    image_path: str,
    memory_db: dict = None,
    memory_path: str = None,
    trigger_generate: Literal[0, 1] = 1,
    workflow_path: str = COMPONENT_WORKFLOW_PATH,
    seed: int = None,
    output_dir: str = None,
    timeout: int = 300
) -> List[str]:
    """
    部件生成统一接口：自动生成提示词 + 调用 ComfyUI 生成图片

    流程：
    1. generate_component_prompt() 生成中文提示词
    2. prepare_workflow_component() 翻译为英文 + 构建工作流
    3. 提交到 ComfyUI 并获取生成的图片

    生成的图片保存到 generated_images 文件夹，文件名为部件名

    Args:
        component_name: 部件名称（同时作为保存文件名）
        image_path: 输入图片路径（部件参考图）
        memory_db: 记忆数据库（为 None 则从文件加载）
        memory_path: 记忆文件路径
        trigger_generate: 是否触发生成（1=触发，0=不触发）
        workflow_path: 工作流模板路径
        seed: 种子值
        output_dir: 输出目录（默认为 generated_images 文件夹）
        timeout: 超时时间（秒）

    Returns:
        生成的图片路径列表
    """
    print(f"\n=== 部件生成（提示词自动生成模式）===")
    print(f"[Input] Component: {component_name}")
    print(f"[Input] Image: {image_path}")

    # 1. 生成提示词
    prompt = generate_component_prompt(
        component_name=component_name,
        memory_db=memory_db,
        trigger_generate=trigger_generate
    )

    if not prompt:
        print("[Generate] No prompt generated, skipping image generation")
        return []

    print(f"[Prompt] Generated: {prompt[:100]}...")

    # 2. 调用图片生成（使用部件名作为保存文件名）
    return generate_component_image(
        prompt=prompt,
        image_path=image_path,
        workflow_path=workflow_path,
        seed=seed,
        output_dir=output_dir,
        save_name=component_name,  # 使用部件名作为文件名
        timeout=timeout
    )


def generate_overall_with_prompt(
    image_paths: List[str],
    memory_db: dict = None,
    memory_path: str = None,
    trigger_generate: Literal[0, 1] = 1,
    component_image_mapping: dict = None,
    overall_image_index: int = None,
    workflow_path: str = OVERALL_WORKFLOW_PATH,
    seed: int = None,
    output_dir: str = None,
    timeout: int = 300
) -> List[str]:
    """
    整体生成统一接口：自动生成提示词 + 调用 ComfyUI 生成图片

    流程：
    1. generate_overall_prompt() 生成中文提示词（包含 [@Image N] 引用）
    2. prepare_workflow_overall() 翻译为英文 + 构建工作流
    3. 提交到 ComfyUI 并获取生成的图片

    生成的图片保存到 generated_images 文件夹，文件名为 overall

    Args:
        image_paths: 输入图片路径列表（最多9张部件图）
        memory_db: 记忆数据库（为 None 则从文件加载）
        memory_path: 记忆文件路径
        trigger_generate: 是否触发生成（1=触发，0=不触发）
        component_image_mapping: 部件图片索引映射，格式如 {"车架": 0, "车轮": 1, ...}
        overall_image_index: 整体图片的索引（如有整体参考图）
        workflow_path: 工作流模板路径
        seed: 种子值
        output_dir: 输出目录（默认为 generated_images 文件夹）
        timeout: 超时时间（秒）

    Returns:
        生成的图片路径列表
    """
    print(f"\n=== 整体生成（提示词自动生成模式）===")
    print(f"[Input] Images: {len(image_paths)} 张")

    # 1. 生成提示词
    prompt = generate_overall_prompt(
        memory_db=memory_db,
        trigger_generate=trigger_generate,
        component_image_mapping=component_image_mapping,
        overall_image_index=overall_image_index
    )

    if not prompt:
        print("[Generate] No prompt generated, skipping image generation")
        return []

    print(f"[Prompt] Generated: {prompt[:100]}...")

    # 2. 调用图片生成（保存为 overall.png）
    return generate_overall_image(
        prompt=prompt,
        image_paths=image_paths,
        workflow_path=workflow_path,
        seed=seed,
        output_dir=output_dir,
        save_name="overall",  # 整体图片保存为 overall.png
        timeout=timeout
    )


def generate_image_with_memory(
    mode: GenerateMode,
    image_paths: Union[str, List[str]],
    component_name: str = None,
    memory_db: dict = None,
    memory_path: str = None,
    trigger_generate: Literal[0, 1] = 1,
    component_image_mapping: dict = None,
    overall_image_index: int = None,
    seed: int = None,
    output_dir: str = None,
    timeout: int = 300
) -> List[str]:
    """
    统一图片生成接口（带记忆）：根据模式自动生成提示词 + 图片

    Args:
        mode: 生成模式（1=部件生成，2=整体生成）
        image_paths: 图片路径（部件模式为单张，整体模式为多张）
        component_name: 部件名称（mode=1 时必需）
        memory_db: 记忆数据库
        memory_path: 记忆文件路径
        trigger_generate: 是否触发生成
        component_image_mapping: 部件图片索引映射（mode=2时使用）
        overall_image_index: 整体图片索引（mode=2时使用）
        seed: 种子值
        output_dir: 输出目录
        timeout: 超时时间（秒）

    Returns:
        生成的图片路径列表
    """
    if mode == 1:
        # 部件生成
        if component_name is None:
            raise ValueError("部件生成模式需要提供 component_name")

        if isinstance(image_paths, list):
            if len(image_paths) == 0:
                raise ValueError("部件生成需要提供参考图片")
            image_paths = image_paths[0]

        return generate_component_with_prompt(
            component_name=component_name,
            image_path=image_paths,
            memory_db=memory_db,
            memory_path=memory_path,
            trigger_generate=trigger_generate,
            seed=seed,
            output_dir=output_dir,
            timeout=timeout
        )

    elif mode == 2:
        # 整体生成
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        return generate_overall_with_prompt(
            image_paths=image_paths,
            memory_db=memory_db,
            memory_path=memory_path,
            trigger_generate=trigger_generate,
            component_image_mapping=component_image_mapping,
            overall_image_index=overall_image_index,
            seed=seed,
            output_dir=output_dir,
            timeout=timeout
        )

    else:
        raise ValueError(f"无效的生成模式: {mode}")


# ========== 测试 ==========

if __name__ == "__main__":
    print("=== Generate_image Module Test ===")

    # 测试 ComfyUI 连接
    print("\n--- Test 1: Check ComfyUI connection ---")
    try:
        client = ComfyUIClient()
        response = requests.get(f"{client.base_url}/system_stats")
        print(f"ComfyUI status: {response.status_code}")
        if response.status_code == 200:
            print("ComfyUI 连接成功！")
    except Exception as e:
        print(f"ComfyUI 连接失败: {e}")

    # 测试部件生成工作流加载
    print("\n--- Test 2: Load Component workflow ---")
    try:
        workflow = load_workflow_template(COMPONENT_WORKFLOW_PATH)
        print(f"部件工作流节点数: {len(workflow['nodes'])}")
        print(f"部件工作流链接数: {len(workflow['links'])}")
    except Exception as e:
        print(f"部件工作流加载失败: {e}")

    # 测试整体生成工作流加载
    print("\n--- Test 3: Load Overall workflow ---")
    try:
        workflow = load_workflow_template(OVERALL_WORKFLOW_PATH)
        print(f"整体工作流节点数: {len(workflow['nodes'])}")
        print(f"整体工作流链接数: {len(workflow['links'])}")
    except Exception as e:
        print(f"整体工作流加载失败: {e}")

    # 示例：部件生成（手动提示词）
    print("\n--- Test 4: generate_component_image (手动提示词) ---")
    print("调用示例:")
    print("""
generate_component_image(
    prompt="设计一个现代简约风格的自行车把手，金属材质，带有舒适的手柄套",
    image_path="path/to/component.png"
)
""")

    # 示例：整体生成（手动提示词）
    print("\n--- Test 5: generate_overall_image (手动提示词) ---")
    print("调用示例:")
    print("""
generate_overall_image(
    prompt="[@Image 0] 至 [@Image 3] 的部件共同构成一辆自行车...",
    image_paths=["frame.png", "wheel.png", "seat.png", "handle.png"]
)
""")

    # 示例：部件生成（自动提示词）
    print("\n--- Test 6: generate_component_with_prompt (自动提示词) ---")
    print("调用示例:")
    print("""
generate_component_with_prompt(
    component_name="把手",
    image_path="handle.png",
    memory_db={"...": "..."}  # 或使用默认 object_nodes.json
)
""")

    # 示例：整体生成（自动提示词）
    print("\n--- Test 7: generate_overall_with_prompt (自动提示词) ---")
    print("调用示例:")
    print("""
generate_overall_with_prompt(
    image_paths=["frame.png", "wheel.png", "seat.png"],
    component_image_mapping={"车架": 0, "车轮": 1, "车座": 2}
)
""")