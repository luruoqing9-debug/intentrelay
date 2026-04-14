"""
main.py - IntentRelay 后台服务入口

系统角色：
    main.py 作为后台服务，提供处理函数供外部系统调用。
    mico 状态切换由外部系统管理，main.py 不主动改变状态。

职责：
    1. 系统初始化（加载环境变量、记忆数据库、启动语音识别）
    2. 维护后台服务（监控语音识别状态、定期保存记忆）
    3. 提供处理函数入口（供外部系统调用）
    4. 管理记忆数据库
    5. 系统清理

图像分析流程：
    - Operated_image 文件夹存储视频帧（001.png, 002.png, ...）
    - 触发发生时，外部系统调用 handle_vlm_analysis()
    - main.py 读取 Operated_image 中所有图像
    - 进行多图像 VLM 分析
    - 分析完成后清空 Operated_image

外部系统调用方式：
    - handle_vlm_analysis()      → VLM 分析 + 存入记忆
    - handle_qa_switch()         → LLM 问答
    - handle_image_generation()  → 图片生成
    - get_memory_status()        → 查询记忆状态
    - clear_speech_text()        → 清空语音文本
    - stop_system()              → 停止系统
"""

import sys
import os

# 禁用 transformers 警告
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import time
import threading
import io
import base64
import glob
import shutil
import warnings
warnings.filterwarnings('ignore')

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()

# 项目路径
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OPERATED_IMAGE_DIR = os.path.join(PROJECT_DIR, "Operated_image")

# ==================== 导入模块 ====================

print("[main.py] 正在导入模块...")

# 语音模块
from speech import (
    start_continuous_speech,
    stop_continuous_speech,
    get_accumulated_text,
    get_text_and_clear,
    clear_accumulated_text,
    has_speech_text,
    is_speech_running,
    set_speech_end_callback,
    set_mico_mode,  # 新增：模式切换
    get_mico_mode   # 新增：获取当前模式
)

# 处理模块
from record import (
    process_user_input,
    handle_mode_switch_to_mico0,
    encode_image_to_base64,
    load_memory_from_json,
    save_memory_to_json,
    extract_and_parse_json,
    vlm_chat_multi_images,
    vlm_chat_text_only  # 新增：纯语音 VLM 分析
)

# 记忆模块
from Memory import (
    process_vlm_result,
    batch_update_images,
    get_all_components,
    get_overall_node
)

# 反馈模块
from Feedback import (
    check_vlm_output,
    generate_ai_feedback
)

# 生成模块
from generate import (
    process_generate_request,
    get_components_info
)

# 图片生成模块
from Generate_image import (
    generate_image,
    generate_component_with_prompt,
    generate_overall_with_prompt
)

from PIL import Image

print("[main.py] 模块导入完成")


# ==================== 全局状态 ====================

running = False           # 系统运行标志
memory_db = {}            # 记忆数据库（全局）
monitor_thread = None     # 监控线程
_init_complete = False    # 初始化完成标志


# ==================== 语音回调处理 ====================

def on_speech_end(text: str):
    """
    语音结束回调函数（仅 mico=0 模式，分贝检测触发）

    Args:
        text: 识别到的语音文本
    """
    global memory_db

    # 只在 mico=0 模式下触发 VLM 分析
    if get_mico_mode() != 0:
        return

    print(f"\n{'='*50}")
    print(f"[语音触发] 文本: '{text}'")

    # 直接调用分析，不管有没有图像
    handle_vlm_analysis_with_text(
        trigger_types=["语音输入触发"],
        transcript_text=text
    )


def handle_vlm_analysis_with_text(trigger_types: list, transcript_text: str) -> dict:
    """
    处理 VLM 分析请求（直接传入语音文本）
    - 有图像：多图像 VLM 分析
    - 无图像：纯语音 VLM 分析

    Args:
        trigger_types: 触发类型列表
        transcript_text: 语音文本（直接传入）

    Returns:
        dict: 分析结果
    """
    global memory_db

    result = {
        "success": False,
        "image_count": 0,
        "vlm_result": None,
        "node": None,
        "node_type": None,
        "feedback": None,
        "repeat_count": 0
    }

    print("\n" + "="*50)
    print("[VLM分析] 开始处理")
    print("="*50)

    # 1. 检查是否有语音文本
    if not transcript_text or len(transcript_text.strip()) <= 5:
        print("[VLM分析] 语音文本太短，跳过分析")
        return result

    print(f"\n[Step 1] 语音文本: '{transcript_text}'")

    # 2. 从 Operated_image 读取图像
    print("\n[Step 2] 读取 Operated_image...")
    images = load_images_from_operated_folder()
    result["image_count"] = len(images)

    # 3. 调用 VLM 分析
    if len(images) > 0:
        # 有图像：多图像 VLM 分析
        print(f"  ✓ 已加载 {len(images)} 张图像，使用多图像模式")

        print("\n[Step 3] 编码图像...")
        image_bytes_list = []
        for image, filename in images:
            image_bytes = encode_image_to_base64(image)
            image_bytes_list.append(image_bytes)
            print(f"  ✓ {filename}")

        print("\n[Step 4] VLM 分析（多图像模式）...")
        try:
            vlm_response = vlm_chat_multi_images(
                image_bytes_list=image_bytes_list,
                trigger_types=trigger_types,
                transcript_text=transcript_text
            )
        except Exception as e:
            print(f"[VLM分析] 多图像分析异常: {e}")
            return result

    else:
        # 无图像：纯语音 VLM 分析
        print("  ○ 无图像，使用纯语音模式")

        print("\n[Step 3] VLM 分析（纯语音模式）...")
        try:
            vlm_response = vlm_chat_text_only(
                trigger_types=trigger_types,
                transcript_text=transcript_text
            )
        except Exception as e:
            print(f"[VLM分析] 纯语音分析异常: {e}")
            return result

    if not vlm_response:
        print("[VLM分析] VLM 无返回结果")
        return result

    print(f"  ✓ VLM 返回成功")

    # 4. 解析 JSON
    print("\n[Step 4] 解析 JSON...")
    vlm_json = extract_and_parse_json(vlm_response)

    if not vlm_json:
        print("[VLM分析] JSON 解析失败")
        print(f"  原始响应: {vlm_response[:200]}...")
        return result

    result["vlm_result"] = vlm_json
    print(f"  ✓ 解析成功:")
    print(f"    - type: {vlm_json.get('type')}")
    print(f"    - label: {vlm_json.get('label')}")
    print(f"    - User intent: {vlm_json.get('User intent')}")

    # 5. 存入记忆
    print("\n[Step 5] 存入记忆...")
    try:
        # 有图像则使用第一张，否则不传图像
        first_image = images[0][0] if len(images) > 0 else None

        node, node_type = process_vlm_result(
            vlm_result=vlm_json,
            memory_db=memory_db,
            component_image=first_image
        )

        if node:
            memory_db[node.node_id] = node.model_dump()
            result["node"] = node.model_dump()
            result["node_type"] = node_type
            print(f"  ✓ 已存储: {node_type}")
            print(f"    - node_id: {node.node_id}")

            save_memory()

    except Exception as e:
        print(f"[记忆] 存储异常: {e}")

    # 6. 重复检测 + AI反馈
    print("\n[Step 6] 重复检测...")
    try:
        parsed, should_feedback, count = check_vlm_output(vlm_response, trigger_types[0])
        result["repeat_count"] = count
        print(f"  ✓ 重复计数: {count}")

        if should_feedback:
            print(f"  ⚠ 重复超过阈值，生成 AI 反馈...")
            component_name = vlm_json.get("label", "unknown")
            feedback = generate_ai_feedback(component_name, parsed, memory_db)
            result["feedback"] = feedback
            print(f"  ✓ AI 建议:")
            print(f"    {feedback.get('content', '')[:200]}{'...' if len(feedback.get('content', '')) > 200 else ''}")

    except Exception as e:
        print(f"[反馈] 检测异常: {e}")

    # 7. 清空 Operated_image
    if len(images) > 0:
        print("\n[Step 7] 清空 Operated_image...")
        clear_operated_image_folder()

    result["success"] = True

    print("\n" + "="*50)
    print("[VLM分析] 处理完成")
    print("="*50)

    return result


def handle_qa_switch() -> dict:
    """
    处理问答切换请求（外部系统调用）

    当外部系统检测到用户从 mico=1 切换到 mico=0 时调用此函数。

    Returns:
        dict: {
            "success": True/False,
            "question": "用户的问题",
            "answer": "AI的回答"
        }
    """
    result = {
        "success": False,
        "question": "",
        "answer": ""
    }

    print("\n[问答] 处理切换请求...")

    try:
        answer = handle_mode_switch_to_mico0(previous_mico=1)
        result["question"] = "（已清空）"
        result["answer"] = answer
        result["success"] = True
        print(f"[问答] 回答长度: {len(answer)} 字符")

    except Exception as e:
        print(f"[问答] 处理异常: {e}")
        result["answer"] = f"处理异常: {str(e)}"

    return result

def init_system() -> bool:
    """
    系统初始化

    执行步骤：
        1. 检查必要配置（GEMINI_API_KEY）
        2. 加载记忆数据库（object_nodes.json）
        3. 设置语音结束回调
        4. 启动持续语音识别
        5. 设置运行标志

    Returns:
        bool: 初始化是否成功
    """
    global memory_db, running, _init_complete

    print("\n" + "="*60)
    print("IntentRelay 系统初始化")
    print("="*60)

    # 1. 检查必要配置
    print("\n[Step 1] 检查配置...")

    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        print("[错误] GEMINI_API_KEY 未设置，请检查 .env 文件")
        return False
    print(f"✓ GEMINI_API_KEY: 已配置")

    # 2. 加载记忆数据库
    print("\n[Step 2] 加载记忆数据库...")

    memory_path = os.path.join(PROJECT_DIR, "object_nodes.json")
    memory_db = load_memory_from_json(memory_path)

    component_count = sum(1 for d in memory_db.values() if d.get('node_type') == 'COMPONENT')
    overall_count = sum(1 for d in memory_db.values() if d.get('node_type') == 'OVERALL')
    print(f"✓ 记忆数据库: {len(memory_db)} 个节点")
    print(f"  - COMPONENT: {component_count}")
    print(f"  - OVERALL: {overall_count}")

    # 3. 设置语音结束回调
    print("\n[Step 3] 设置语音回调...")
    set_speech_end_callback(on_speech_end)
    print("✓ 语音回调已设置")

    # 4. 启动持续语音识别
    print("\n[Step 4] 启动语音识别...")

    try:
        start_continuous_speech()
        print("  正在初始化，请稍候...")
        time.sleep(2)

        if not is_speech_running():
            print("[错误] 语音识别启动失败")
            return False

        print("✓ 语音识别启动成功")

    except Exception as e:
        print(f"[错误] 语音识别启动异常: {e}")
        return False

    # 5. 设置运行标志
    running = True
    _init_complete = True

    print("\n" + "="*60)
    print("系统初始化完成")
    print("="*60)

    return True


# ==================== 监控函数 ====================

def monitor_loop():
    """
    后台监控循环 - 定期保存记忆数据库
    """
    global running, memory_db

    save_interval = 60  # 每60秒保存一次

    while running:
        try:
            time.sleep(save_interval)
            if running:
                save_memory()
        except Exception as e:
            print(f"[监控] 异常: {e}")


def save_memory():
    """保存记忆数据库到文件"""
    global memory_db

    memory_path = os.path.join(PROJECT_DIR, "object_nodes.json")
    save_memory_to_json(memory_db, memory_path)


# ==================== 对外提供的处理函数 ====================

def load_images_from_operated_folder() -> list:
    """
    从 Operated_image 文件夹加载所有图像

    Returns:
        list: [(PIL.Image, 文件名), ...] 图像列表
    """
    images = []

    # 确保 Operated_image 文件夹存在
    if not os.path.exists(OPERATED_IMAGE_DIR):
        os.makedirs(OPERATED_IMAGE_DIR)
        return images

    # 支持的图片格式
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp']

    # 获取所有图片文件并按名称排序
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(OPERATED_IMAGE_DIR, f"*{ext}")))
        image_files.extend(glob.glob(os.path.join(OPERATED_IMAGE_DIR, f"*{ext.upper()}")))

    image_files.sort()  # 按文件名排序

    # 加载每张图片
    for image_path in image_files:
        try:
            image = Image.open(image_path).convert("RGB")
            filename = os.path.basename(image_path)
            images.append((image, filename))
        except Exception as e:
            print(f"[图像加载] 跳过无效文件 {image_path}: {e}")

    return images


def clear_operated_image_folder():
    """
    清空 Operated_image 文件夹中的所有图片

    分析完成后调用此函数，准备下一次分析。
    """
    if not os.path.exists(OPERATED_IMAGE_DIR):
        return

    # 支持的图片格式
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp']

    # 删除所有图片文件
    deleted_count = 0
    for ext in image_extensions:
        for image_path in glob.glob(os.path.join(OPERATED_IMAGE_DIR, f"*{ext}")):
            try:
                os.remove(image_path)
                deleted_count += 1
            except Exception as e:
                print(f"[清空] 删除失败 {image_path}: {e}")

        for image_path in glob.glob(os.path.join(OPERATED_IMAGE_DIR, f"*{ext.upper()}")):
            try:
                os.remove(image_path)
                deleted_count += 1
            except Exception as e:
                print(f"[清空] 删除失败 {image_path}: {e}")

    print(f"[清空] Operated_image 已清空，删除 {deleted_count} 张图片")


def handle_vlm_analysis(trigger_types: list = None) -> dict:
    """
    处理 VLM 分析请求（外部系统调用）

    流程：
        1. 从 Operated_image 文件夹读取所有图像
        2. 获取累积语音文本
        3. 调用多图像 VLM 分析
        4. 解析结果并存入记忆
        5. 检测重复，触发 AI 反馈（如果达到阈值）
        6. 清空 Operated_image 文件夹
        7. 清空语音文本

    Args:
        trigger_types: 触发类型列表，如 ["语音输入触发", "眼动焦点注视单一物体超过五秒钟"]

    Returns:
        dict: {
            "success": True/False,
            "image_count": 3,            # 分析的图像数量
            "vlm_result": {...},         # VLM 分析结果 JSON
            "node": {...},               # 记忆节点（如果创建了）
            "node_type": "component",    # 节点类型
            "feedback": {...},           # AI 反馈（如果触发）
            "repeat_count": 3            # 重复计数
        }
    """
    global memory_db

    result = {
        "success": False,
        "image_count": 0,
        "vlm_result": None,
        "node": None,
        "node_type": None,
        "feedback": None,
        "repeat_count": 0
    }

    # 设置默认触发类型
    trigger_types = trigger_types or ["语音输入触发"]

    print("\n" + "="*50)
    print("[VLM分析] 开始处理")
    print("="*50)

    # 1. 从 Operated_image 读取所有图像
    print("\n[Step 1] 读取 Operated_image 图像...")
    images = load_images_from_operated_folder()
    result["image_count"] = len(images)

    if len(images) == 0:
        print("[VLM分析] Operated_image 中无图像，无法分析")
        print("  请确保外部系统已将视频帧放入 Operated_image 文件夹")
        return result

    print(f"  ✓ 已加载 {len(images)} 张图像")

    # 2. 检查是否有语音内容
    print("\n[Step 2] 检查语音内容...")
    transcript_text = get_accumulated_text()
    has_voice = has_speech_text()

    if has_voice:
        print(f"  ✓ 语音文本: '{transcript_text[:100]}{'...' if len(transcript_text) > 100 else ''}'")
    else:
        print("  ○ 无语音内容（但仍然进行分析）")

    # 3. 编码所有图像为 Base64
    print("\n[Step 3] 编码图像...")
    image_bytes_list = []
    for image, filename in images:
        image_bytes = encode_image_to_base64(image)
        image_bytes_list.append(image_bytes)
        print(f"  ✓ {filename}")

    # 4. 调用多图像 VLM 分析
    print("\n[Step 4] VLM 分析（多图像模式）...")
    print(f"  触发类型: {trigger_types}")

    try:
        vlm_response = vlm_chat_multi_images(
            image_bytes_list=image_bytes_list,
            trigger_types=trigger_types,
            transcript_text=transcript_text
        )

        if not vlm_response:
            print("[VLM分析] VLM 无返回结果")
            return result

        print(f"  ✓ VLM 返回成功")

    except Exception as e:
        print(f"[VLM分析] VLM 调用异常: {e}")
        return result

    # 5. 解析 JSON
    print("\n[Step 5] 解析 JSON...")
    vlm_json = extract_and_parse_json(vlm_response)

    if not vlm_json:
        print("[VLM分析] JSON 解析失败")
        print(f"  原始响应: {vlm_response[:200]}...")
        return result

    result["vlm_result"] = vlm_json
    print(f"  ✓ 解析成功:")
    print(f"    - type: {vlm_json.get('type')}")
    print(f"    - label: {vlm_json.get('label')}")
    print(f"    - User intent: {vlm_json.get('User intent')}")

    # 6. 存入记忆
    print("\n[Step 6] 存入记忆...")
    try:
        # 使用第一张图像作为代表图像存储
        first_image = images[0][0]

        node, node_type = process_vlm_result(
            vlm_result=vlm_json,
            memory_db=memory_db,
            component_image=first_image
        )

        if node:
            memory_db[node.node_id] = node.model_dump()
            result["node"] = node.model_dump()
            result["node_type"] = node_type
            print(f"  ✓ 已存储: {node_type}")
            print(f"    - node_id: {node.node_id}")

            # 保存记忆
            save_memory()

    except Exception as e:
        print(f"[记忆] 存储异常: {e}")

    # 7. 重复检测 + AI反馈
    print("\n[Step 7] 重复检测...")
    try:
        parsed, should_feedback, count = check_vlm_output(vlm_response, trigger_types[0])
        result["repeat_count"] = count
        print(f"  ✓ 重复计数: {count}")

        if should_feedback:
            print(f"  ⚠ 重复超过阈值，生成 AI 反馈...")
            component_name = vlm_json.get("label", "unknown")
            feedback = generate_ai_feedback(component_name, parsed, memory_db)
            result["feedback"] = feedback
            print(f"  ✓ AI 建议:")
            print(f"    {feedback.get('content', '')[:200]}{'...' if len(feedback.get('content', '')) > 200 else ''}")

    except Exception as e:
        print(f"[反馈] 检测异常: {e}")

    # 8. 清空 Operated_image 文件夹
    print("\n[Step 8] 清空 Operated_image...")
    clear_operated_image_folder()

    # 9. 清空语音文本，准备新一轮
    print("\n[Step 9] 清空语音文本...")
    clear_accumulated_text()

    result["success"] = True

    print("\n" + "="*50)
    print("[VLM分析] 处理完成")
    print("="*50)

    return result


def handle_qa_switch() -> dict:
    """
    处理问答切换请求（外部系统调用）

    当外部系统检测到用户从 mico=1 切换到 mico=0 时调用此函数。

    流程：
        1. 获取累积语音文本并清空
        2. 调用 LLM 进行问答
        3. 返回回答

    Returns:
        dict: {
            "success": True/False,
            "question": "用户的问题",
            "answer": "AI的回答"
        }
    """
    result = {
        "success": False,
        "question": "",
        "answer": ""
    }

    print("\n[问答] 处理切换请求...")

    try:
        # 调用 handle_mode_switch_to_mico0
        answer = handle_mode_switch_to_mico0(previous_mico=1)

        # 获取问题文本（已经被清空了，但可以从日志获取）
        result["question"] = "（已清空）"  # 实际问题在调用前获取
        result["answer"] = answer
        result["success"] = True

        print(f"[问答] 回答长度: {len(answer)} 字符")

    except Exception as e:
        print(f"[问答] 处理异常: {e}")
        result["answer"] = f"处理异常: {str(e)}"

    return result


def handle_image_generation(
    mode: int,
    component_name: str = None,
    image_paths: list = None,
    component_image_mapping: dict = None,
    overall_image_index: int = None
) -> dict:
    """
    处理图像生成请求（外部系统调用）

    Args:
        mode: 生成模式
            - 1: 部件生成（需要 component_name 和参考图）
            - 2: 整体生成（需要部件图片列表）
        component_name: 部件名称（mode=1 时必需）
        image_paths: 图片路径列表
            - mode=1: 单张参考图
            - mode=2: 多张部件图
        component_image_mapping: 部件图片索引映射（mode=2时使用）
            格式: {"车架": 0, "车轮": 1, ...}
        overall_image_index: 整体图片索引（mode=2时使用）

    Returns:
        dict: {
            "success": True/False,
            "prompt": "生成的提示词",
            "image_paths": ["生成的图片路径列表"]
        }
    """
    global memory_db

    result = {
        "success": False,
        "prompt": "",
        "image_paths": []
    }

    print(f"\n[图像生成] mode={mode}")

    try:
        if mode == 1:
            # 部件生成
            if not component_name:
                print("[错误] 部件生成需要 component_name")
                return result

            print(f"[图像生成] 部件: {component_name}")

            if image_paths and len(image_paths) > 0:
                # 有参考图，使用带参考图生成
                paths = generate_component_with_prompt(
                    component_name=component_name,
                    image_path=image_paths[0],
                    memory_db=memory_db
                )
                result["image_paths"] = paths
            else:
                # 无参考图，只生成提示词
                prompt = process_generate_request(
                    t=1,
                    component_name=component_name,
                    memory_db=memory_db
                )
                result["prompt"] = prompt
                print("[图像生成] 无参考图，只返回提示词")

        elif mode == 2:
            # 整体生成
            print(f"[图像生成] 整体，部件图: {len(image_paths) if image_paths else 0} 张")

            paths = generate_overall_with_prompt(
                image_paths=image_paths or [],
                memory_db=memory_db,
                component_image_mapping=component_image_mapping,
                overall_image_index=overall_image_index
            )
            result["image_paths"] = paths

        else:
            print(f"[错误] 无效的 mode: {mode}")
            return result

        if result["image_paths"]:
            result["success"] = True
            print(f"[图像生成] 成功，输出: {result['image_paths']}")

            # 更新记忆中的图片
            batch_update_images(memory_db)
            save_memory()

    except Exception as e:
        print(f"[图像生成] 异常: {e}")
        result["success"] = False

    return result


def get_memory_status() -> dict:
    """
    获取记忆状态（外部系统调用）

    Returns:
        dict: {
            "total_nodes": 10,
            "component_count": 8,
            "overall_count": 1,
            "components": ["handle", "wheel", ...],
            "overall_exists": True,
            "speech_running": True,
            "accumulated_text": "当前累积的语音文本",
            "has_speech_text": True
        }
    """
    global memory_db

    # 统计节点
    component_count = sum(1 for d in memory_db.values() if d.get('node_type') == 'COMPONENT')
    overall_count = sum(1 for d in memory_db.values() if d.get('node_type') == 'OVERALL')

    # 获取部件名称列表
    components = []
    for node_id, data in memory_db.items():
        if data.get('node_type') == 'COMPONENT':
            components.append(data.get('component_name', 'unknown'))

    # 获取整体节点
    overall = get_overall_node(memory_db)
    overall_exists = overall is not None

    return {
        "total_nodes": len(memory_db),
        "component_count": component_count,
        "overall_count": overall_count,
        "components": components,
        "overall_exists": overall_exists,
        "speech_running": is_speech_running(),
        "accumulated_text": get_accumulated_text(),
        "has_speech_text": has_speech_text()
    }


def get_current_speech_text() -> str:
    """
    获取当前累积的语音文本（不清空）

    外部系统可以定期调用此函数来显示实时转写结果。

    Returns:
        str: 当前累积的语音文本
    """
    return get_accumulated_text()


def clear_speech_text():
    """
    清空语音累积文本（外部系统调用）

    用于在特定情况下手动清空累积文本。
    """
    clear_accumulated_text()
    print("[语音] 累积文本已清空")


def get_components_info() -> dict:
    """
    获取所有部件的结构/功能/待确定信息

    Returns:
        dict: {
            "structure_info": ["履带：齿轮连接结构", ...],
            "function_info": ["履带：平稳行走", ...],
            "uncertain_info": ["履带：外形风格不确定", ...]
        }
    """
    global memory_db

    return get_components_info(trigger=1, memory_db=memory_db)


# ==================== 辅助函数 ====================

def decode_base64_image(image_bytes: str) -> Image.Image:
    """
    从 Base64 字符串解码图像

    Args:
        image_bytes: Base64 编码的图像字符串

    Returns:
        PIL.Image: 解码后的图像对象
    """
    img_data = base64.b64decode(image_bytes)
    return Image.open(io.BytesIO(img_data)).convert("RGB")


def encode_image(image: Image.Image) -> str:
    """
    将 PIL Image 编码为 Base64 字符串

    Args:
        image: PIL Image 对象

    Returns:
        str: Base64 编码的图像字符串
    """
    return encode_image_to_base64(image)


# ==================== 模式切换接口 ====================

def switch_mico_mode(mode: int) -> dict:
    """
    切换 mico 模式（外部系统调用）

    mico=0: 分贝检测模式，自动触发 VLM 分析
    mico=1: 持续发送模式，用于 LLM 问答

    Args:
        mode: 目标模式（0 或 1）

    Returns:
        dict: {
            "success": True,
            "mode": 0/1,
            "text": "切换时获取的文本"（仅从 mico=1 切换到 mico=0 时有效）
        }
    """
    result = {
        "success": True,
        "mode": mode,
        "text": ""
    }

    current_mode = get_mico_mode()

    if mode == 0 and current_mode == 1:
        # 从 mico=1 切换到 mico=0
        # 获取累积文本传给 LLM
        text = get_text_and_clear()
        result["text"] = text
        print(f"\n[模式切换] mico=1 → mico=0")
        print(f"[模式切换] 获取文本: '{text}'")

        # 这里可以调用 LLM 问答
        # handle_qa_switch() 或其他处理

    set_mico_mode(mode)
    print(f"[模式切换] 当前模式: mico={mode}")

    return result


def get_current_mico_mode() -> int:
    """获取当前 mico 模式"""
    return get_mico_mode()


# ==================== 系统停止 ====================

def stop_system():
    """
    停止系统（外部系统调用）

    执行步骤：
        1. 设置停止标志
        2. 停止语音识别
        3. 保存记忆数据库
        4. 清理资源
    """
    global running

    print("\n" + "="*60)
    print("系统正在停止...")
    print("="*60)

    running = False

    # 停止语音识别
    try:
        stop_continuous_speech()
        print("✓ 语音识别已停止")
    except Exception as e:
        print(f"语音识别停止异常: {e}")

    # 保存记忆
    try:
        save_memory()
        print("✓ 记忆数据库已保存")
    except Exception as e:
        print(f"记忆保存异常: {e}")

    print("\n系统已停止")


# ==================== 主函数 ====================

def main():
    """
    主入口

    启动后台服务，等待外部系统调用。
    """
    # 初始化
    if not init_system():
        print("\n[系统] 初始化失败，退出")
        return

    print("\n" + "="*60)
    print("IntentRelay 后台服务已启动")
    print("="*60)
    print("\n系统状态:")
    status = get_memory_status()
    print(f"  - 记忆节点: {status['total_nodes']}")
    print(f"  - 语音识别: {'运行中' if status['speech_running'] else '已停止'}")
    print(f"  - 语音触发: 已启用（分贝检测）")

    print("\n等待语音触发或外部系统调用...")
    print("="*60)

    # 启动监控线程（仅定期保存）
    global monitor_thread
    monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
    monitor_thread.start()

    # 主线程保持运行
    try:
        while running:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n收到 Ctrl+C 信号")
        stop_system()

    except Exception as e:
        print(f"\n主循环异常: {e}")
        stop_system()


# ==================== 入口 ====================

if __name__ == "__main__":
    main()