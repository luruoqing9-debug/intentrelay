"""
api.py - HTTP API 接口层

遵循 main.py 的业务逻辑，不过度拆分也不过度整合。

启动方式：python api.py
服务地址：http://localhost:5000

文件夹说明：
├── original_image/      → 参考图（始终只有一张，上传新图时自动替换旧的）
│                          部件生成：部件参考图
│                          整体生成：粗糙结构参考图
├── Operated_image/      → 操作记录图（临时，VLM分析后清空）
├── generated_images/    → 生成的图片（临时，存入记忆后移动到 processed_images）
├── processed_images/    → 已存记忆的图片（永久保留，前端使用此路径）

接口分类：
├── 系统控制
│   ├── GET  /health           - 健康检查
│   ├── POST /init             - 初始化系统
│   ├── POST /stop             - 停止系统
│
├── VLM 分析（使用 Operated_image，分析后清空）
│   ├── POST /vlm_analysis     - 语音触发（可选图片）
│   ├── POST /vlm_analysis_images - 眼动/手势触发（必需图片）
│
├── AI 反馈
│   ├── POST /ai_feedback      - 按需调用（should_feedback=true 时）
│   ├── POST /qa_switch        - 问答处理
│
├── 图像生成（自动读取 original_image，输出到 generated_images）
│   ├── POST /generate_prompt  - 生成提示词（自动检测粗糙参考图）
│   ├── POST /generate_image   - 调用 ComfyUI 生成图像
│                          mode=1（部件）：自动读取 original_image/ 参考图
│                          mode=2（整体）：自动读取 processed_images/ 部件图 + original_image/ 粗糙参考图
│
├── 状态查询
│   ├── GET  /memory_status    - 查询系统状态
│   ├── GET  /components_list  - 获取部件列表
│   ├── GET  /components_info  - 获取部件详情
│
├── 语音管理
│   ├── GET  /speech_text      - 获取累积语音文本
│   ├── POST /speech_clear     - 清空语音文本
│
├── mico 模式管理
│   ├── GET  /mico_mode        - 获取当前模式
│   ├── POST /mico_switch      - 切换模式
"""

import os
import sys
import json

# 禁用警告
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from flask import Flask, request, jsonify
from flask_cors import CORS

print("[api.py] 正在导入模块...")

# ==================== 导入 main.py 的函数 ====================

from main import (
    # 系统控制
    init_system,
    stop_system,
    memory_db,
    save_memory,

    # 核心业务流程
    handle_vlm_analysis,
    handle_vlm_analysis_with_text,
    handle_qa_switch,

    # 状态查询
    get_memory_status,
    get_components_info,

    # 语音管理
    get_current_speech_text,
    clear_speech_text,

    # mico 模式管理
    switch_mico_mode,
    get_current_mico_mode
)

# 导入 AI 反馈生成函数（供前端按需调用）
from Feedback import generate_ai_feedback

# 导入提示词生成函数
from generate import process_generate_request

# 导入图像生成函数
from Generate_image import (
    generate_component_image,
    generate_overall_image,
    get_original_image,
    get_processed_component_images
)

# 导入记忆更新函数
from Memory import batch_update_images

print("[api.py] 模块导入完成")


# ==================== 创建 Flask 应用 ====================

app = Flask(__name__)
CORS(app)


# ==================== 全局状态 ====================

_system_initialized = False


# ==================== 系统控制接口 ====================

@app.route('/health', methods=['GET'])
def api_health():
    """
    健康检查

    输入: 无

    输出:
        {
            "status": "ok",
            "system_initialized": true/false,
            "speech_running": true/false,
            "memory_loaded": true/false,
            "mico_mode": 0/1
        }
    """
    return jsonify({
        "status": "ok",
        "system_initialized": _system_initialized,
        "speech_running": len(memory_db) > 0 and _system_initialized,
        "memory_loaded": len(memory_db) > 0,
        "mico_mode": get_current_mico_mode()
    })


@app.route('/init', methods=['POST'])
def api_init():
    """
    初始化系统

    输入: 无

    输出:
        {
            "success": true/false,
            "message": "初始化成功" 或 错误信息,
            "memory_status": {...}  // 初始化后的记忆状态
        }
    """
    global _system_initialized

    try:
        if _system_initialized:
            return jsonify({
                "success": True,
                "message": "系统已初始化",
                "memory_status": get_memory_status()
            })

        success = init_system()
        if success:
            _system_initialized = True
            return jsonify({
                "success": True,
                "message": "系统初始化成功",
                "memory_status": get_memory_status()
            })
        else:
            return jsonify({
                "success": False,
                "message": "系统初始化失败"
            }), 500

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/stop', methods=['POST'])
def api_stop():
    """
    停止系统

    输入: 无

    输出:
        {
            "success": true
        }
    """
    global _system_initialized

    try:
        stop_system()
        _system_initialized = False
        return jsonify({"success": True})

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ==================== 核心业务流程接口 ====================

@app.route('/vlm_analysis', methods=['POST'])
def api_vlm_analysis():
    """
    VLM 分析（语音触发）

    文件夹说明：
    - Operated_image/ → 可选的图像输入（如有则使用，无则纯语音分析）
    - 如果使用了 Operated_image/ 的图片，分析后会清空该文件夹

    流程：
    1. 检查语音文本长度
    2. 从 Operated_image/ 读取图片（如有）
    3. 有图片 → 多图像 VLM 分析；无图片 → 纯语音 VLM 分析
    4. 存入记忆（不存图片，只存文本描述）
    5. 重复检测
    6. 清空 Operated_image/（如果使用了图片）

    输入:
        {
            "transcript_text": "用户说的话",     // 必需：语音文本
            "trigger_types": ["语音输入触发"]    // 可选：默认为语音输入触发
        }

    输出:
        {
            "success": true/false,
            "image_count": 0,                 // 分析的图片数量（0=纯语音）
            "vlm_result": {...},              // VLM 分析结果
            "node_type": "component/overall", // 存入记忆的节点类型
            "repeat_count": 3,                // 重复计数
            "should_feedback": true,          // 是否需要 AI 反馈
            "parsed_vlm": {...}               // 供 /ai_feedback 使用
        }
    """
    try:
        data = request.json or {}
        transcript_text = data.get('transcript_text', '')
        trigger_types = data.get('trigger_types', ['语音输入触发'])

        if not transcript_text:
            return jsonify({
                "success": False,
                "error": "缺少 transcript_text 参数"
            }), 400

        # 调用 main.py 的完整流程
        result = handle_vlm_analysis_with_text(
            trigger_types=trigger_types,
            transcript_text=transcript_text
        )

        return jsonify(result)

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/vlm_analysis_images', methods=['POST'])
def api_vlm_analysis_images():
    """
    多图像 VLM 分析（眼动/手势触发）

    文件夹说明：
    - Operated_image/ → VLM 分析输入（眼动追踪/摄像头捕获的视频帧）
    - 分析完成后会自动清空 Operated_image/

    流程：
    1. 从 Operated_image/ 读取所有图片（按文件名排序）
    2. VLM 分析图片 + 累积语音文本
    3. 存入记忆（不存图片，只存文本描述）
    4. 重复检测
    5. 清空 Operated_image/

    输入:
        {
            "trigger_types": ["眼动焦点注视单一物体超过五秒钟"]  // 必需
        }

    trigger_types 可选值：
        - "眼动焦点注视单一物体超过五秒钟"（眼动触发）
        - "手部坐标与物体坐标重叠"（手势触发）

    输出:
        {
            "success": true/false,
            "image_count": 3,                 // 分析的图片数量
            "vlm_result": {...},              // VLM 分析结果
            "node_type": "component/overall", // 存入记忆的节点类型
            "repeat_count": 3,                // 重复计数
            "should_feedback": true,          // 是否需要 AI 反馈
            "parsed_vlm": {...}               // 供 /ai_feedback 使用
        }
    """
    try:
        data = request.json or {}
        trigger_types = data.get('trigger_types')

        # 必须传入 trigger_types
        if not trigger_types:
            return jsonify({
                "success": False,
                "error": "缺少 trigger_types 参数（眼动触发或手势触发）"
            }), 400

        # 调用 main.py 的完整流程
        result = handle_vlm_analysis(trigger_types=trigger_types)

        return jsonify(result)

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/generate_prompt', methods=['POST'])
def api_generate_prompt():
    """
    生成图像提示词（前端可修改后再调用 /generate_image）

    说明：
    - 从记忆数据库（object_nodes.json）获取部件的已有描述信息
    - mode=2（整体生成）时，系统会自动：
      - 从 processed_images/ 读取部件数量（排除 overall.png）
      - 从 original_image/ 检测是否有粗糙参考图

    输入:
        {
            "mode": 1 或 2,                        // 必需：1=部件，2=整体
            "component_name": "履带",              // mode=1 时必需：部件名称
            "component_image_mapping": {...}       // mode=2 时可选：部件图片索引映射（不传则自动生成）
        }

    mode=1 输入示例（部件生成）:
        {
            "mode": 1,
            "component_name": "履带"
        }

    mode=2 输入示例（整体生成）:
        {
            "mode": 2
            // 不需要其他参数，系统自动读取部件图和粗糙参考图
        }

    输出:
        {
            "success": true,
            "prompt": "圆润流线型履带设计，表面光滑...",
            "has_reference_image": true/false,  // mode=2时返回：是否有粗糙参考图
            "overall_image_index": 3,           // mode=2且有粗糙参考图时返回
            "component_count": 3,               // mode=2时返回：部件数量
            "component_images": ["履带.png", ...]  // mode=2时返回：部件图片文件名列表
        }
    """
    try:
        data = request.json or {}
        mode = data.get('mode')
        component_name = data.get('component_name')
        component_image_mapping = data.get('component_image_mapping')

        if mode is None:
            return jsonify({
                "success": False,
                "error": "缺少 mode 参数（1=部件，2=整体）"
            }), 400

        # mode=2 整体生成时，自动检测
        overall_image_index = None
        has_reference_image = False
        component_count = 0
        component_images = []

        if mode == 2:
            # 自动获取部件图片（排除 overall.png）
            component_image_paths = get_processed_component_images(exclude_overall=True)
            component_count = len(component_image_paths)
            component_images = [os.path.basename(p) for p in component_image_paths]

            print(f"[generate_prompt] 检测到 {component_count} 个部件图片")
            for name in component_images:
                print(f"  - {name}")

            # 自动生成 component_image_mapping（文件名 -> 索引）
            if not component_image_mapping:
                component_image_mapping = {}
                for i, path in enumerate(component_image_paths):
                    name = os.path.splitext(os.path.basename(path))[0]
                    component_image_mapping[name] = i
                print(f"[generate_prompt] 自动生成索引映射: {component_image_mapping}")

            # 检查 original_image/ 是否有粗糙参考图
            reference_image = get_original_image()
            if reference_image:
                has_reference_image = True
                overall_image_index = component_count  # 粗糙参考图放在末尾
                print(f"[generate_prompt] 粗糙参考图索引: {overall_image_index}")
            else:
                print("[generate_prompt] 无粗糙参考图")

        # 调用 generate.py 生成提示词
        prompt = process_generate_request(
            t=mode,
            component_name=component_name,
            trigger_generate=1,
            memory_db=memory_db,
            component_image_mapping=component_image_mapping,
            overall_image_index=overall_image_index
        )

        if not prompt:
            return jsonify({
                "success": False,
                "error": "提示词生成失败"
            }), 500

        # 构建返回结果
        result = {
            "success": True,
            "prompt": prompt
        }

        # mode=2 时返回详细信息
        if mode == 2:
            result["has_reference_image"] = has_reference_image
            result["component_count"] = component_count
            result["component_images"] = component_images
            if overall_image_index is not None:
                result["overall_image_index"] = overall_image_index

        return jsonify(result)

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/generate_image', methods=['POST'])
def api_generate_image():
    """
    调用 ComfyUI 生成图像（使用提示词）

    前端流程：
    1. 用户将参考图放入 original_image/ 文件夹
    2. 用户对着麦克风说话描述设计意图（自动触发 VLM 分析）
    3. 调用 /generate_prompt 获取提示词
    4. 用户可修改提示词
    5. 调用此接口生成图像

    文件夹说明：
    - original_image/   → 参考图来源（前端上传，始终只有一张）
    - generated_images/ → 生成图片存放位置

    输入:
        mode=1（部件生成）:
        {
            "mode": 1,
            "prompt": "提示词",
            "component_name": "履带"  // 必需：部件名称（用于保存文件名）
            // 不需要 image_paths，系统自动从 original_image/ 读取参考图
        }

        mode=2（整体生成）:
        {
            "mode": 2,
            "prompt": "提示词",
            "save_name": "overall"  // 可选，默认为 "overall"
            // 部件图片系统自动从 processed_images/ 读取（排除 overall.png）
            // 粗糙参考图系统自动从 original_image/ 读取
        }

    输出:
        {
            "success": true/false,
            "image_paths": ["processed_images/履带.png"],  // 最终存储路径
            "reference_image_used": "original_image/xxx.png",  // 使用的参考图路径
            "component_images_used": ["processed_images/履带.png", ...]  // mode=2时返回使用的部件图
        }
    """
    try:
        data = request.json or {}
        mode = data.get('mode')
        prompt = data.get('prompt', '')
        save_name = data.get('save_name')
        component_name = data.get('component_name')

        # 项目目录（用于路径转换）
        project_dir = os.path.dirname(os.path.abspath(__file__))
        processed_images_dir = os.path.join(project_dir, "processed_images")

        if mode is None:
            return jsonify({
                "success": False,
                "error": "缺少 mode 参数（1=部件，2=整体）"
            }), 400

        if not prompt:
            return jsonify({
                "success": False,
                "error": "缺少 prompt 参数"
            }), 400

        # 调用 ComfyUI 生成图像
        if mode == 1:
            # 部件生成：自动从 original_image/ 读取参考图
            reference_image = get_original_image()

            if not reference_image:
                return jsonify({
                    "success": False,
                    "error": "original_image/ 中没有参考图片，请先上传参考图"
                }), 400

            # 部件名称用于保存文件名
            if not save_name and not component_name:
                return jsonify({
                    "success": False,
                    "error": "部件生成需要 component_name 或 save_name"
                }), 400

            final_save_name = save_name or component_name

            print(f"[generate_image] 部件生成: {final_save_name}")
            print(f"[generate_image] 参考图: {reference_image}")

            saved_paths = generate_component_image(
                prompt=prompt,
                image_path=reference_image,
                save_name=final_save_name
            )

            if not saved_paths:
                return jsonify({
                    "success": False,
                    "error": "图像生成失败"
                }), 500

            # 更新记忆（图片移动到 processed_images）
            batch_update_images(memory_db)
            save_memory()

            # 转换路径：generated_images → processed_images
            final_paths = []
            for path in saved_paths:
                filename = os.path.basename(path)
                final_path = os.path.join(processed_images_dir, filename)
                final_paths.append(final_path)

            return jsonify({
                "success": True,
                "image_paths": final_paths,
                "reference_image_used": reference_image
            })

        elif mode == 2:
            # 整体生成：自动读取 processed_images/ 的部件图 + original_image/ 的粗糙参考图

            # 自动获取所有部件图片（排除 overall.png）
            component_image_paths = get_processed_component_images(exclude_overall=True)

            if len(component_image_paths) == 0:
                return jsonify({
                    "success": False,
                    "error": "processed_images/ 中没有部件图片，请先生成部件"
                }), 400

            # 获取粗糙参考图
            reference_image = get_original_image()

            # 组合图片数组：部件图 + 粗糙参考图（如果有）
            final_image_paths = list(component_image_paths)  # 复制一份

            if reference_image:
                final_image_paths.append(reference_image)
                print(f"[generate_image] 整体生成: {len(component_image_paths)} 张部件图 + 1 张粗糙参考图")
                print(f"[generate_image] 粗糙参考图: {reference_image}")
            else:
                print(f"[generate_image] 整体生成: {len(component_image_paths)} 张部件图（无粗糙参考图）")

            # 默认保存名为 overall
            final_save_name = save_name or "overall"

            saved_paths = generate_overall_image(
                prompt=prompt,
                image_paths=final_image_paths,
                save_name=final_save_name
            )

            if not saved_paths:
                return jsonify({
                    "success": False,
                    "error": "图像生成失败"
                }), 500

            # 更新记忆（图片移动到 processed_images）
            batch_update_images(memory_db)
            save_memory()

            # 转换路径：generated_images → processed_images
            final_paths = []
            for path in saved_paths:
                filename = os.path.basename(path)
                final_path = os.path.join(processed_images_dir, filename)
                final_paths.append(final_path)

            result = {
                "success": True,
                "image_paths": final_paths,
                "component_count": len(component_image_paths),
                "component_images_used": component_image_paths  # 返回使用的部件图路径
            }

            if reference_image:
                result["reference_image_used"] = reference_image
                result["overall_image_index"] = len(component_image_paths)  # 粗糙参考图的索引

            return jsonify(result)

        else:
            return jsonify({
                "success": False,
                "error": "无效的 mode（只能是 1 或 2）"
            }), 400

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/qa_switch', methods=['POST'])
def api_qa_switch():
    """
    问答处理（mico=1→0 切换时调用）

    流程：获取累积语音 → 调用 LLM → 返回回答

    输入: 无

    输出:
        {
            "success": true/false,
            "question": "用户的问题",
            "answer": "AI的回答"
        }
    """
    try:
        result = handle_qa_switch()
        return jsonify(result)

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/ai_feedback', methods=['POST'])
def api_ai_feedback():
    """
    AI 反馈生成（前端按需调用）

    当 /vlm_analysis 返回 should_feedback=True 时，
    前端可以调用此接口获取 AI 建议。

    输入:
        {
            "component_name": "履带",    // 必需
            "parsed_vlm": {...}          // 必需：/vlm_analysis 返回的 parsed_vlm
        }

    输出:
        {
            "success": true,
            "feedback": {
                "content": "AI建议内容",
                "scores": {
                    "Novelty": 85,
                    "Value": 90,
                    "Feasibility": 95,
                    "Context-specific": 98
                },
                "total_score": 92.0
            }
        }
    """
    try:
        data = request.json or {}
        component_name = data.get('component_name', '')
        parsed_vlm = data.get('parsed_vlm', {})

        if not component_name:
            return jsonify({
                "success": False,
                "error": "缺少 component_name 参数"
            }), 400

        if not parsed_vlm:
            return jsonify({
                "success": False,
                "error": "缺少 parsed_vlm 参数（需要从 /vlm_analysis 返回值中获取）"
            }), 400

        # 调用 Feedback.py 的 AI 反馈生成函数
        feedback = generate_ai_feedback(component_name, parsed_vlm, memory_db)

        return jsonify({
            "success": True,
            "feedback": feedback
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ==================== 状态查询接口 ====================

@app.route('/memory_status', methods=['GET'])
def api_memory_status():
    """
    查询记忆数据库状态

    输入: 无

    输出:
        {
            "total_nodes": 5,
            "component_count": 4,
            "overall_count": 1,
            "components": ["履带", "底座", ...],
            "overall_exists": true/false,
            "speech_running": true/false,
            "accumulated_text": "...",
            "has_speech_text": true/false
        }
    """
    return jsonify(get_memory_status())


@app.route('/components_list', methods=['GET'])
def api_components_list():
    """
    获取所有部件名称列表

    输入: 无

    输出:
        {
            "components": [
                {"name": "履带", "node_id": "..."},
                {"name": "底座", "node_id": "..."}
            ]
        }
    """
    components = []
    for node_id, data in memory_db.items():
        if data.get('node_type') == 'COMPONENT':
            components.append({
                "name": data.get('component_name', 'unknown'),
                "node_id": node_id
            })

    return jsonify({"components": components})


@app.route('/components_info', methods=['GET'])
def api_components_info():
    """
    获取所有部件的结构/功能/待确定信息

    输入: 无

    输出:
        {
            "structure_info": ["履带：齿轮连接结构", ...],
            "function_info": ["履带：平稳行走", ...],
            "uncertain_info": ["履带：外形风格不确定", ...]
        }
    """
    return jsonify(get_components_info())


# ==================== 语音管理接口 ====================

@app.route('/speech_text', methods=['GET'])
def api_speech_text():
    """
    获取当前累积的语音文本（不清空）

    输入: 无

    输出:
        {
            "text": "当前累积的语音文本",
            "has_text": true/false
        }
    """
    text = get_current_speech_text()
    has_text = len(text.strip()) > 0

    return jsonify({
        "text": text,
        "has_text": has_text
    })


@app.route('/speech_clear', methods=['POST'])
def api_speech_clear():
    """
    清空累积的语音文本

    输入: 无

    输出:
        {
            "success": true
        }
    """
    try:
        clear_speech_text()
        return jsonify({"success": True})

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ==================== mico 模式管理接口 ====================

@app.route('/mico_mode', methods=['GET'])
def api_mico_mode():
    """
    获取当前 mico 模式

    输入: 无

    输出:
        {
            "mode": 0 或 1      // 0=分贝检测，1=持续发送
        }
    """
    mode = get_current_mico_mode()
    return jsonify({"mode": mode})


@app.route('/mico_switch', methods=['POST'])
def api_mico_switch():
    """
    切换 mico 模式

    mico=0: 分贝检测模式，自动触发 VLM 分析
    mico=1: 持续发送模式，用于 LLM 问答

    输入:
        {
            "mode": 0 或 1    // 必需
        }

    输出:
        {
            "success": true,
            "previous_mode": 之前的模式,
            "current_mode": 当前模式,
            "text": "切换时累积的文本"（从1切换到0时）
        }
    """
    try:
        data = request.json or {}
        new_mode = data.get('mode')

        if new_mode is None:
            return jsonify({
                "success": False,
                "error": "缺少 mode 参数"
            }), 400

        # 调用 main.py 的模式切换
        result = switch_mico_mode(new_mode)

        return jsonify(result)

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ==================== 错误处理 ====================

@app.errorhandler(404)
def not_found(e):
    return jsonify({
        "success": False,
        "error": "接口不存在"
    }), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({
        "success": False,
        "error": "服务器内部错误"
    }), 500


# ==================== 启动服务 ====================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("IntentRelay HTTP API 服务")
    print("="*70)

    print("\n【系统控制】")
    print("  GET  /health              - 健康检查")
    print("  POST /init                - 初始化系统")
    print("  POST /stop                - 停止系统")

    print("\n【VLM 分析】（使用 Operated_image，分析后清空）")
    print("  POST /vlm_analysis        - 语音触发（可选图片）")
    print("  POST /vlm_analysis_images - 眼动/手势触发（必需图片）")

    print("\n【AI 反馈】")
    print("  POST /ai_feedback         - 按需调用")
    print("  POST /qa_switch           - 问答处理")

    print("\n【图像生成】（自动读取 original_image，输出到 generated_images）")
    print("  POST /generate_prompt     - 生成提示词（自动检测粗糙参考图）")
    print("  POST /generate_image      - 调用 ComfyUI 生成")
    print("    mode=1（部件）：自动读取 original_image/ 参考图")
    print("    mode=2（整体）：部件图由前端传入 + 自动读取粗糙参考图")

    print("\n【状态查询】")
    print("  GET  /memory_status       - 查询系统状态")
    print("  GET  /components_list     - 获取部件列表")
    print("  GET  /components_info     - 获取部件详情")

    print("\n【语音管理】")
    print("  GET  /speech_text         - 获取累积语音文本")
    print("  POST /speech_clear        - 清空语音文本")

    print("\n【mico 模式管理】")
    print("  GET  /mico_mode           - 获取当前模式")
    print("  POST /mico_switch         - 切换模式")

    print("\n服务地址: http://localhost:5000")
    print("="*70)

    # 自动初始化系统（启动语音识别）
    print("\n[自动初始化] 正在启动系统...")
    if init_system():
        _system_initialized = True
        print("[自动初始化] 系统初始化成功，语音识别已启动")
        print("\n>>> 现在可以对着麦克风说话，静音1秒后会自动触发 VLM 分析 <<<")
    else:
        print("[自动初始化] 系统初始化失败，请检查配置")
        print("  - 确保 .env 文件中设置了 GEMINI_API_KEY")
        print("  - 确保网络可以访问 Gemini API")

    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True
    )