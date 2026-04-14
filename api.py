"""
api.py - HTTP API 接口层

把 main.py 的 Python 函数包装成 HTTP 接口，
让前端（Web/Unity/其他程序）可以通过网络调用。

使用方式：
1. 启动服务：python api.py
2. 前端发送 HTTP 请求到 http://localhost:5000

接口列表：
- POST /vlm_analysis         → VLM 分析
- POST /image_generation     → 图像生成
- POST /qa                   → LLM 问答
- GET  /memory_status        → 查询记忆状态
- POST /switch_mode          → 切换语音模式
- GET  /components_info      → 获取部件信息
- POST /clear_speech         → 清空语音文本
- POST /stop                 → 停止系统
"""

import os
import sys

# 禁用警告
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from flask import Flask, request, jsonify
from flask_cors import CORS  # 允许跨域请求

print("[api.py] 正在导入模块...")

# 导入 main.py 的函数
from main import (
    init_system,
    handle_vlm_analysis,
    handle_vlm_analysis_with_text,
    handle_image_generation,
    handle_qa_switch,
    get_memory_status,
    switch_mico_mode,
    get_components_info,
    clear_speech_text,
    stop_system,
    get_current_speech_text,
    get_current_mico_mode,
    is_speech_running,
    has_speech_text
)

print("[api.py] 模块导入完成")


# ==================== 创建 Flask 应用 ====================

app = Flask(__name__)
CORS(app)  # 允许前端跨域访问


# ==================== 系统初始化 ====================

print("\n[api.py] 正在初始化系统...")
init_success = init_system()

if not init_success:
    print("[api.py] 系统初始化失败，请检查配置")
    sys.exit(1)

print("[api.py] 系统初始化成功")


# ==================== API 接口定义 ====================


# ----- 1. VLM 分析 -----
@app.route('/vlm_analysis', methods=['POST'])
def api_vlm_analysis():
    """
    VLM 分析接口

    输入 (JSON):
        {
            "trigger_types": ["语音输入触发"],  // 可选
            "transcript_text": "用户说的话"      // 必需
        }

    输出 (JSON):
        {
            "success": True/False,
            "vlm_result": {...},
            "feedback": {...},
            "repeat_count": 3
        }
    """
    try:
        data = request.json or {}

        trigger_types = data.get('trigger_types', ['语音输入触发'])
        transcript_text = data.get('transcript_text', '')

        if not transcript_text:
            return jsonify({
                "success": False,
                "error": "缺少 transcript_text 参数"
            }), 400

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


@app.route('/vlm_analysis_with_images', methods=['POST'])
def api_vlm_analysis_with_images():
    """
    VLM 分析接口（带图像）

    注意：图像需要先放入 Operated_image 文件夹

    输入 (JSON):
        {
            "trigger_types": ["语音输入触发"]
        }

    输出 (JSON):
        {
            "success": True/False,
            "image_count": 3,
            "vlm_result": {...},
            "feedback": {...}
        }
    """
    try:
        data = request.json or {}
        trigger_types = data.get('trigger_types', ['语音输入触发'])

        result = handle_vlm_analysis(trigger_types=trigger_types)

        return jsonify(result)

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ----- 2. 图像生成 -----
@app.route('/image_generation', methods=['POST'])
def api_image_generation():
    """
    图像生成接口

    输入 (JSON):
        {
            "mode": 1 或 2,                    // 必需：1=部件，2=整体
            "component_name": "履带",          // mode=1 时必需
            "image_paths": ["path1.png"],      // 图片路径列表
            "component_image_mapping": {...}   // mode=2 时可选
        }

    输出 (JSON):
        {
            "success": True/False,
            "prompt": "生成的提示词",
            "image_paths": ["生成的图片路径"]
        }
    """
    try:
        data = request.json or {}

        mode = data.get('mode')
        if mode is None:
            return jsonify({
                "success": False,
                "error": "缺少 mode 参数"
            }), 400

        component_name = data.get('component_name')
        image_paths = data.get('image_paths')
        component_image_mapping = data.get('component_image_mapping')
        overall_image_index = data.get('overall_image_index')

        result = handle_image_generation(
            mode=mode,
            component_name=component_name,
            image_paths=image_paths,
            component_image_mapping=component_image_mapping,
            overall_image_index=overall_image_index
        )

        return jsonify(result)

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ----- 3. LLM 问答 -----
@app.route('/qa', methods=['POST'])
def api_qa():
    """
    LLM 问答接口

    当用户从问答模式切换回来时调用

    输入: 无

    输出 (JSON):
        {
            "success": True/False,
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


# ----- 4. 查询记忆状态 -----
@app.route('/memory_status', methods=['GET'])
def api_memory_status():
    """
    查询记忆状态

    输入: 无

    输出 (JSON):
        {
            "total_nodes": 5,
            "component_count": 4,
            "overall_count": 1,
            "components": ["履带", "底座", ...],
            "overall_exists": True,
            "speech_running": True,
            "accumulated_text": "当前语音",
            "has_speech_text": True
        }
    """
    try:
        result = get_memory_status()
        return jsonify(result)

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ----- 5. 切换语音模式 -----
@app.route('/switch_mode', methods=['POST'])
def api_switch_mode():
    """
    切换语音模式

    输入 (JSON):
        {
            "mode": 0 或 1    // 0=分贝检测模式，1=持续发送模式
        }

    输出 (JSON):
        {
            "success": True,
            "mode": 0/1,
            "text": "切换时获取的文本"
        }
    """
    try:
        data = request.json or {}
        mode = data.get('mode')

        if mode is None:
            return jsonify({
                "success": False,
                "error": "缺少 mode 参数"
            }), 400

        result = switch_mico_mode(mode)
        return jsonify(result)

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/current_mode', methods=['GET'])
def api_current_mode():
    """
    获取当前语音模式

    输入: 无

    输出 (JSON):
        {
            "mode": 0 或 1
        }
    """
    try:
        mode = get_current_mico_mode()
        return jsonify({"mode": mode})

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ----- 6. 获取部件信息 -----
@app.route('/components_info', methods=['GET'])
def api_components_info():
    """
    获取部件结构/功能/待确定信息

    输入: 无

    输出 (JSON):
        {
            "structure_info": ["履带：齿轮连接结构", ...],
            "function_info": ["履带：平稳行走", ...],
            "uncertain_info": ["履带：外形风格不确定", ...]
        }
    """
    try:
        result = get_components_info()
        return jsonify(result)

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ----- 7. 获取当前语音文本 -----
@app.route('/speech_text', methods=['GET'])
def api_speech_text():
    """
    获取当前累积的语音文本（不清空）

    输入: 无

    输出 (JSON):
        {
            "text": "当前累积的语音文本",
            "has_text": True/False
        }
    """
    try:
        text = get_current_speech_text()
        has_text = has_speech_text()
        return jsonify({
            "text": text,
            "has_text": has_text
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/clear_speech', methods=['POST'])
def api_clear_speech():
    """
    清空语音累积文本

    输入: 无

    输出 (JSON):
        {
            "success": True
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


# ----- 8. 检查语音识别状态 -----
@app.route('/speech_status', methods=['GET'])
def api_speech_status():
    """
    检查语音识别是否运行

    输入: 无

    输出 (JSON):
        {
            "running": True/False
        }
    """
    try:
        running = is_speech_running()
        return jsonify({"running": running})

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ----- 9. 停止系统 -----
@app.route('/stop', methods=['POST'])
def api_stop():
    """
    停止系统

    输入: 无

    输出 (JSON):
        {
            "success": True
        }
    """
    try:
        stop_system()
        return jsonify({"success": True})

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ----- 10. 健康检查 -----
@app.route('/health', methods=['GET'])
def api_health():
    """
    健康检查接口

    输入: 无

    输出 (JSON):
        {
            "status": "ok",
            "speech_running": True/False,
            "memory_loaded": True/False
        }
    """
    return jsonify({
        "status": "ok",
        "speech_running": is_speech_running(),
        "memory_loaded": True
    })


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
    print("\n" + "="*60)
    print("IntentRelay HTTP API 服务已启动")
    print("="*60)
    print("\n接口列表:")
    print("  POST /vlm_analysis           - VLM 分析（纯语音）")
    print("  POST /vlm_analysis_with_images - VLM 分析（带图像）")
    print("  POST /image_generation       - 图像生成")
    print("  POST /qa                     - LLM 问答")
    print("  GET  /memory_status          - 查询记忆状态")
    print("  POST /switch_mode            - 切换语音模式")
    print("  GET  /current_mode           - 获取当前模式")
    print("  GET  /components_info        - 获取部件信息")
    print("  GET  /speech_text            - 获取语音文本")
    print("  POST /clear_speech           - 清空语音文本")
    print("  GET  /speech_status          - 检查语音识别状态")
    print("  POST /stop                   - 停止系统")
    print("  GET  /health                 - 健康检查")

    print("\n服务地址: http://localhost:5000")
    print("\n等待前端请求...")
    print("="*60)

    # 启动 Flask 服务
    app.run(
        host='0.0.0.0',  # 允许外部访问
        port=5000,
        debug=False,
        threaded=True  # 支持多线程
    )