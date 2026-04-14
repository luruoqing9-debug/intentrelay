"""
api.py - HTTP API 接口层

把每个功能拆分成独立的 HTTP 接口，
前端可以单独调用每个功能，有完整的控制权。

启动方式：python api.py
服务地址：http://localhost:5000

接口分类：
├── 系统控制
│   ├── GET  /health           - 健康检查
│   ├── POST /init             - 初始化系统
│   ├── POST /stop             - 停止系统
│
├── VLM 分析
│   ├── POST /vlm_analysis_text    - 纯语音 VLM 分析
│   ├── POST /vlm_analysis_images  - 图像+语音 VLM 分析
│   ├── POST /vlm_parse_json       - 解析 VLM JSON 结果
│
├── 记忆管理
│   ├── GET  /memory_status    - 查询记忆状态
│   ├── GET  /components_list  - 获取部件列表
│   ├── GET  /components_info  - 获取部件结构/功能信息
│   ├── POST /memory_save      - 保存记忆到文件
│   ├── POST /vlm_result_save  - 存入 VLM 结果到记忆
│
├── AI 反馈
│   ├── POST /repeat_check     - 重复检测
│   ├── POST /ai_feedback      - 生成 AI 建议
│   ├── POST /user_feedback    - 处理用户反馈
│   ├── GET  /feedback_weights - 获取评分权重
│   ├── POST /weights_reset    - 重置评分权重
│
├── 图像生成
│   ├── POST /prompt_component - 生成部件提示词
│   ├── POST /prompt_overall   - 生成整体提示词
│   ├── POST /image_generate   - 调用 ComfyUI 生成图像
│
├── LLM 问答
│   ├── POST /qa_answer        - LLM 回答问题
│
├── 语音模式
│   ├── GET  /mico_mode        - 获取当前 mico 模式
│   ├── POST /mico_switch      - 切换 mico 模式
│   ├── GET  /speech_text      - 获取累积语音文本
│   ├── POST /speech_clear     - 清空语音文本
│   ├── GET  /speech_status    - 检查语音识别状态
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

# ==================== 导入各模块的函数 ====================

# main.py - 系统控制和整合逻辑
from main import (
    init_system,
    handle_vlm_analysis,
    handle_vlm_analysis_with_text,
    handle_image_generation,
    handle_qa_switch,
    stop_system,
    load_images_from_operated_folder,
    clear_operated_image_folder,
    save_memory,
    memory_db,
    OPERATED_IMAGE_DIR
)

# speech.py - 语音相关
from speech import (
    get_accumulated_text,
    get_text_and_clear,
    clear_accumulated_text,
    has_speech_text,
    is_speech_running,
    get_mico_mode,
    set_mico_mode
)

# Memory.py - 记忆管理
from Memory import (
    process_vlm_result,
    get_all_components,
    get_overall_node
)

# Feedback.py - AI 反馈
from Feedback import (
    check_vlm_output,
    generate_ai_feedback,
    process_user_feedback,
    get_current_weights,
    reset_weights,
    reset_repeat_count,
    get_repeat_count
)

# generate.py - 提示词生成
from generate import (
    process_generate_request,
    get_components_info
)

# Generate_image.py - 图像生成
from Generate_image import (
    generate_component_image,
    generate_overall_image,
    ComfyUIClient,
    COMPONENT_WORKFLOW_PATH,
    OVERALL_WORKFLOW_PATH
)

# record.py - 编码和解析
from record import (
    extract_and_parse_json,
    encode_image_to_base64,
    vlm_chat_multi_images,
    vlm_chat_text_only,
    load_memory_from_json,
    save_memory_to_json
)

print("[api.py] 模块导入完成")


# ==================== 创建 Flask 应用 ====================

app = Flask(__name__)
CORS(app)


# ==================== 全局状态 ====================

_system_initialized = False
MEMORY_PATH = os.path.join(os.path.dirname(__file__), "object_nodes.json")


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
            "memory_loaded": true/false
        }
    """
    return jsonify({
        "status": "ok",
        "system_initialized": _system_initialized,
        "speech_running": is_speech_running(),
        "memory_loaded": len(memory_db) > 0
    })


@app.route('/init', methods=['POST'])
def api_init():
    """
    初始化系统

    输入: 无

    输出:
        {
            "success": true/false,
            "message": "初始化成功" 或 错误信息
        }
    """
    global _system_initialized

    try:
        if _system_initialized:
            return jsonify({
                "success": True,
                "message": "系统已初始化"
            })

        success = init_system()
        if success:
            _system_initialized = True
            return jsonify({
                "success": True,
                "message": "系统初始化成功"
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


# ==================== VLM 分析接口 ====================

@app.route('/vlm_analysis_text', methods=['POST'])
def api_vlm_analysis_text():
    """
    纯语音 VLM 分析

    输入:
        {
            "transcript_text": "用户说的话",     // 必需
            "trigger_types": ["语音输入触发"]    // 可选
        }

    输出:
        {
            "success": true/false,
            "vlm_response": "VLM原始响应字符串",
            "vlm_json": {                       // 解析后的JSON
                "type": "component",
                "label": "履带",
                "User intent": "Appearance design",
                "User Speaking": "...",
                "Behavior description": "..."
            }
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

        # 调用 VLM 分析
        vlm_response = vlm_chat_text_only(
            trigger_types=trigger_types,
            transcript_text=transcript_text
        )

        # 解析 JSON
        vlm_json = extract_and_parse_json(vlm_response)

        return jsonify({
            "success": True,
            "vlm_response": vlm_response,
            "vlm_json": vlm_json
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/vlm_analysis_images', methods=['POST'])
def api_vlm_analysis_images():
    """
    图像+语音 VLM 分析

    注意：图像需先放入 Operated_image 文件夹

    输入:
        {
            "trigger_types": ["语音输入触发"]    // 可选
        }

    输出:
        {
            "success": true/false,
            "image_count": 3,
            "vlm_response": "VLM原始响应",
            "vlm_json": {...}
        }
    """
    try:
        data = request.json or {}
        trigger_types = data.get('trigger_types', ['语音输入触发'])

        # 读取图像
        images = load_images_from_operated_folder()

        if len(images) == 0:
            return jsonify({
                "success": False,
                "error": "Operated_image 文件夹中没有图像"
            }), 400

        # 编码图像
        image_bytes_list = []
        for image, filename in images:
            image_bytes = encode_image_to_base64(image)
            image_bytes_list.append(image_bytes)

        # 获取语音文本
        transcript_text = get_accumulated_text()

        # 调用多图像 VLM 分析
        vlm_response = vlm_chat_multi_images(
            image_bytes_list=image_bytes_list,
            trigger_types=trigger_types,
            transcript_text=transcript_text
        )

        # 解析 JSON
        vlm_json = extract_and_parse_json(vlm_response)

        return jsonify({
            "success": True,
            "image_count": len(images),
            "transcript_text": transcript_text,
            "vlm_response": vlm_response,
            "vlm_json": vlm_json
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/vlm_parse_json', methods=['POST'])
def api_vlm_parse_json():
    """
    解析 VLM 响应的 JSON

    输入:
        {
            "vlm_response": "VLM返回的JSON字符串"
        }

    输出:
        {
            "success": true/false,
            "vlm_json": {
                "type": "component",
                "label": "履带",
                "User intent": "Appearance design",
                ...
            }
        }
    """
    try:
        data = request.json or {}
        vlm_response = data.get('vlm_response', '')

        if not vlm_response:
            return jsonify({
                "success": False,
                "error": "缺少 vlm_response 参数"
            }), 400

        vlm_json = extract_and_parse_json(vlm_response)

        if vlm_json is None:
            return jsonify({
                "success": False,
                "error": "JSON 解析失败"
            }), 400

        return jsonify({
            "success": True,
            "vlm_json": vlm_json
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ==================== 记忆管理接口 ====================

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
            "overall_exists": true/false
        }
    """
    global memory_db

    component_count = sum(1 for d in memory_db.values() if d.get('node_type') == 'COMPONENT')
    overall_count = sum(1 for d in memory_db.values() if d.get('node_type') == 'OVERALL')

    components = []
    for node_id, data in memory_db.items():
        if data.get('node_type') == 'COMPONENT':
            components.append(data.get('component_name', 'unknown'))

    overall = get_overall_node(memory_db)

    return jsonify({
        "total_nodes": len(memory_db),
        "component_count": component_count,
        "overall_count": overall_count,
        "components": components,
        "overall_exists": overall is not None
    })


@app.route('/components_list', methods=['GET'])
def api_components_list():
    """
    获取所有部件名称列表

    输入: 无

    输出:
        {
            "components": ["履带", "底座", "探照灯", ...]
        }
    """
    global memory_db

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
    global memory_db

    info = get_components_info(trigger=1, memory_db=memory_db)
    return jsonify(info)


@app.route('/memory_save', methods=['POST'])
def api_memory_save():
    """
    手动保存记忆数据库到文件

    输入: 无

    输出:
        {
            "success": true
        }
    """
    try:
        save_memory()
        return jsonify({"success": True})

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/memory_reload', methods=['POST'])
def api_memory_reload():
    """
    从文件重新加载记忆数据库

    输入: 无

    输出:
        {
            "success": true/false,
            "node_count": 重新加载后的节点数量
        }
    """
    global memory_db

    try:
        memory_db = load_memory_from_json(MEMORY_PATH)
        return jsonify({
            "success": True,
            "node_count": len(memory_db)
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/vlm_result_save', methods=['POST'])
def api_vlm_result_save():
    """
    将 VLM 结果存入记忆数据库

    输入:
        {
            "vlm_json": {...},           // 必需：VLM 分析结果
            "image_path": "path.png"     // 可选：部件图片路径
        }

    输出:
        {
            "success": true/false,
            "node_id": "新节点的ID",
            "node_type": "component/overall"
        }
    """
    global memory_db

    try:
        data = request.json or {}
        vlm_json = data.get('vlm_json')

        if not vlm_json:
            return jsonify({
                "success": False,
                "error": "缺少 vlm_json 参数"
            }), 400

        # 获取图片（如果有）
        component_image = None
        image_path = data.get('image_path')
        if image_path and os.path.exists(image_path):
            from PIL import Image
            component_image = Image.open(image_path).convert("RGB")

        # 存入记忆
        node, node_type = process_vlm_result(
            vlm_result=vlm_json,
            memory_db=memory_db,
            component_image=component_image
        )

        if node:
            memory_db[node.node_id] = node.model_dump()
            save_memory()

            return jsonify({
                "success": True,
                "node_id": node.node_id,
                "node_type": node_type
            })
        else:
            return jsonify({
                "success": False,
                "error": "存入记忆失败"
            }), 500

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ==================== AI 反馈接口 ====================

@app.route('/repeat_check', methods=['POST'])
def api_repeat_check():
    """
    重复检测

    输入:
        {
            "vlm_response": "VLM原始响应",    // 必需
            "trigger_type": "语音输入触发"   // 必需
        }

    输出:
        {
            "success": true,
            "parsed": {...},                 // 解析后的JSON
            "should_feedback": true/false,   // 是否应该生成反馈
            "repeat_count": 3                // 当前重复计数
        }
    """
    try:
        data = request.json or {}
        vlm_response = data.get('vlm_response', '')
        trigger_type = data.get('trigger_type', '语音输入触发')

        if not vlm_response:
            return jsonify({
                "success": False,
                "error": "缺少 vlm_response 参数"
            }), 400

        parsed, should_feedback, count = check_vlm_output(vlm_response, trigger_type)

        return jsonify({
            "success": True,
            "parsed": parsed,
            "should_feedback": should_feedback,
            "repeat_count": count
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/repeat_count', methods=['GET'])
def api_repeat_count():
    """
    获取指定部件的当前重复计数

    输入:
        {
            "component_name": "履带"   // 必需，通过 query 参数传递
        }

    输出:
        {
            "component_name": "履带",
            "repeat_count": 3
        }
    """
    try:
        component_name = request.args.get('component_name', '')

        if not component_name:
            return jsonify({
                "success": False,
                "error": "缺少 component_name 参数"
            }), 400

        count = get_repeat_count(component_name)

        return jsonify({
            "component_name": component_name,
            "repeat_count": count
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/repeat_reset', methods=['POST'])
def api_repeat_reset():
    """
    重置重复计数

    输入:
        {
            "component_name": "履带"   // 可选，不传则重置所有
        }

    输出:
        {
            "success": true
        }
    """
    try:
        data = request.json or {}
        component_name = data.get('component_name')

        reset_repeat_count(component_name)

        return jsonify({"success": True})

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/ai_feedback', methods=['POST'])
def api_ai_feedback():
    """
    生成 AI 反馈建议

    输入:
        {
            "component_name": "履带",     // 必需
            "vlm_output": {...}           // 必需：VLM分析结果
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
    global memory_db

    try:
        data = request.json or {}
        component_name = data.get('component_name', '')
        vlm_output = data.get('vlm_output', {})

        if not component_name:
            return jsonify({
                "success": False,
                "error": "缺少 component_name 参数"
            }), 400

        if not vlm_output:
            return jsonify({
                "success": False,
                "error": "缺少 vlm_output 参数"
            }), 400

        feedback = generate_ai_feedback(component_name, vlm_output, memory_db)

        return jsonify({
            "success": True,
            "feedback": feedback
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/user_feedback', methods=['POST'])
def api_user_feedback():
    """
    处理用户反馈，更新评分权重

    输入:
        {
            "user_feedback": "我觉得可以更创新一些"
        }

    输出:
        {
            "success": true,
            "dimension_changes": {"Novelty": 0.1, ...},
            "updated_weights": {"Novelty": 1.1, ...},
            "analysis": "用户偏好分析"
        }
    """
    try:
        data = request.json or {}
        user_feedback_text = data.get('user_feedback', '')

        if not user_feedback_text:
            return jsonify({
                "success": False,
                "error": "缺少 user_feedback 参数"
            }), 400

        result = process_user_feedback(user_feedback_text)

        return jsonify({
            "success": True,
            **result
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/feedback_weights', methods=['GET'])
def api_feedback_weights():
    """
    获取当前的评分权重

    输入: 无

    输出:
        {
            "weights": {
                "Novelty": 1.0,
                "Value": 1.0,
                "Feasibility": 1.0,
                "Context-specific": 1.0
            }
        }
    """
    weights = get_current_weights()
    return jsonify({"weights": weights})


@app.route('/weights_reset', methods=['POST'])
def api_weights_reset():
    """
    重置评分权重为默认值

    输入: 无

    输出:
        {
            "success": true,
            "weights": {"Novelty": 1.0, ...}
        }
    """
    try:
        reset_weights()
        weights = get_current_weights()

        return jsonify({
            "success": True,
            "weights": weights
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ==================== 提示词生成接口 ====================

@app.route('/prompt_component', methods=['POST'])
def api_prompt_component():
    """
    生成部件图像提示词

    输入:
        {
            "component_name": "履带"    // 必需
        }

    输出:
        {
            "success": true,
            "prompt": "生成的提示词"
        }
    """
    global memory_db

    try:
        data = request.json or {}
        component_name = data.get('component_name', '')

        if not component_name:
            return jsonify({
                "success": False,
                "error": "缺少 component_name 参数"
            }), 400

        prompt = process_generate_request(
            t=1,
            component_name=component_name,
            trigger_generate=1,
            memory_db=memory_db
        )

        return jsonify({
            "success": True,
            "prompt": prompt
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/prompt_overall', methods=['POST'])
def api_prompt_overall():
    """
    生成整体图像提示词

    输入:
        {
            "component_image_mapping": {"履带": 0, "底座": 1},  // 可选
            "overall_image_index": 3                             // 可选
        }

    输出:
        {
            "success": true,
            "prompt": "生成的提示词（包含 [@图N] 引用）"
        }
    """
    global memory_db

    try:
        data = request.json or {}
        component_image_mapping = data.get('component_image_mapping')
        overall_image_index = data.get('overall_image_index')

        prompt = process_generate_request(
            t=2,
            trigger_generate=1,
            memory_db=memory_db,
            component_image_mapping=component_image_mapping,
            overall_image_index=overall_image_index
        )

        return jsonify({
            "success": True,
            "prompt": prompt
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ==================== 图像生成接口 ====================

@app.route('/image_generate', methods=['POST'])
def api_image_generate():
    """
    调用 ComfyUI 生成图像

    输入:
        {
            "mode": 1 或 2,                        // 必需：1=部件，2=整体
            "prompt": "提示词",                    // 必需
            "image_paths": ["path1.png", ...],     // mode=1时单张，mode=2时多张
            "save_name": "履带"                    // 可选：保存的文件名
        }

    输出:
        {
            "success": true,
            "image_paths": ["生成的图片路径"]
        }
    """
    try:
        data = request.json or {}
        mode = data.get('mode')
        prompt = data.get('prompt', '')
        image_paths = data.get('image_paths', [])
        save_name = data.get('save_name', 'generated')

        if mode is None:
            return jsonify({
                "success": False,
                "error": "缺少 mode 参数"
            }), 400

        if not prompt:
            return jsonify({
                "success": False,
                "error": "缺少 prompt 参数"
            }), 400

        if mode == 1:
            # 部件生成
            if len(image_paths) == 0:
                return jsonify({
                    "success": False,
                    "error": "mode=1 需要提供参考图片"
                }), 400

            result_paths = generate_component_image(
                prompt=prompt,
                image_path=image_paths[0],
                workflow_path=COMPONENT_WORKFLOW_PATH,
                save_name=save_name
            )

        elif mode == 2:
            # 整体生成
            result_paths = generate_overall_image(
                prompt=prompt,
                image_paths=image_paths,
                workflow_path=OVERALL_WORKFLOW_PATH,
                save_name=save_name
            )

        else:
            return jsonify({
                "success": False,
                "error": "mode 只能是 1 或 2"
            }), 400

        return jsonify({
            "success": True,
            "image_paths": result_paths
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ==================== LLM 问答接口 ====================

@app.route('/qa_answer', methods=['POST'])
def api_qa_answer():
    """
    LLM 回答用户问题

    输入:
        {
            "question": "用户的问题"    // 必需
        }

    输出:
        {
            "success": true,
            "question": "用户的问题",
            "answer": "AI的回答"
        }
    """
    try:
        data = request.json or {}
        question = data.get('question', '')

        if not question:
            return jsonify({
                "success": False,
                "error": "缺少 question 参数"
            }), 400

        answer = handle_qa_switch()

        return jsonify({
            "success": True,
            "question": question,
            "answer": answer.get("answer", "")
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ==================== 语音模式接口 ====================

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
    mode = get_mico_mode()
    return jsonify({"mode": mode})


@app.route('/mico_switch', methods=['POST'])
def api_mico_switch():
    """
    切换 mico 模式

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

        previous_mode = get_mico_mode()

        # 如果从1切换到0，获取累积文本
        captured_text = ""
        if previous_mode == 1 and new_mode == 0:
            captured_text = get_text_and_clear()

        # 切换模式
        set_mico_mode(new_mode)

        return jsonify({
            "success": True,
            "previous_mode": previous_mode,
            "current_mode": new_mode,
            "text": captured_text
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


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
    text = get_accumulated_text()
    has_text = has_speech_text()

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
        clear_accumulated_text()
        return jsonify({"success": True})

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/speech_status', methods=['GET'])
def api_speech_status():
    """
    检查语音识别是否运行

    输入: 无

    输出:
        {
            "running": true/false
        }
    """
    running = is_speech_running()
    return jsonify({"running": running})


# ==================== 操作文件夹接口 ====================

@app.route('/operated_images', methods=['GET'])
def api_operated_images():
    """
    获取 Operated_image 文件夹中的图片列表

    输入: 无

    输出:
        {
            "images": ["0.png", "1.png", ...],
            "count": 3
        }
    """
    try:
        images = load_images_from_operated_folder()
        filenames = [filename for image, filename in images]

        return jsonify({
            "images": filenames,
            "count": len(images)
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/operated_images_clear', methods=['POST'])
def api_operated_images_clear():
    """
    清空 Operated_image 文件夹

    输入: 无

    输出:
        {
            "success": true,
            "deleted_count": 3
        }
    """
    try:
        clear_operated_image_folder()
        return jsonify({"success": True})

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

    print("\n【VLM 分析】")
    print("  POST /vlm_analysis_text       - 纯语音 VLM 分析")
    print("  POST /vlm_analysis_images     - 图像+语音 VLM 分析")
    print("  POST /vlm_parse_json          - 解析 VLM JSON 结果")

    print("\n【记忆管理】")
    print("  GET  /memory_status       - 查询记忆状态")
    print("  GET  /components_list     - 获取部件列表")
    print("  GET  /components_info     - 获取部件结构/功能信息")
    print("  POST /memory_save         - 保存记忆到文件")
    print("  POST /memory_reload       - 从文件重新加载记忆")
    print("  POST /vlm_result_save     - 存入 VLM 结果到记忆")

    print("\n【AI 反馈】")
    print("  POST /repeat_check        - 重复检测")
    print("  GET  /repeat_count        - 获取重复计数")
    print("  POST /repeat_reset        - 重置重复计数")
    print("  POST /ai_feedback         - 生成 AI 建议")
    print("  POST /user_feedback       - 处理用户反馈")
    print("  GET  /feedback_weights    - 获取评分权重")
    print("  POST /weights_reset       - 重置评分权重")

    print("\n【提示词生成】")
    print("  POST /prompt_component    - 生成部件提示词")
    print("  POST /prompt_overall      - 生成整体提示词")

    print("\n【图像生成】")
    print("  POST /image_generate      - 调用 ComfyUI 生成图像")

    print("\n【LLM 问答】")
    print("  POST /qa_answer           - LLM 回答问题")

    print("\n【语音模式】")
    print("  GET  /mico_mode           - 获取当前 mico 模式")
    print("  POST /mico_switch         - 切换 mico 模式")
    print("  GET  /speech_text         - 获取累积语音文本")
    print("  POST /speech_clear        - 清空语音文本")
    print("  GET  /speech_status       - 检查语音识别状态")

    print("\n【操作文件夹】")
    print("  GET  /operated_images     - 获取图片列表")
    print("  POST /operated_images_clear - 清空图片文件夹")

    print("\n服务地址: http://localhost:5000")
    print("="*70)

    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True
    )