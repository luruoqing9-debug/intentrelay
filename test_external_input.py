"""
test_external_input.py - 模拟外部输入测试

测试 main.py 提供的对外接口，无需启动完整系统。
"""

import os
import sys
import json

# 禁用警告
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()

# 导入必要模块
print("[测试] 正在导入模块...")

from record import (
    load_memory_from_json,
    save_memory_to_json,
    vlm_chat_text_only,
    extract_and_parse_json
)

from Memory import process_vlm_result
from Feedback import check_vlm_output, generate_ai_feedback
from generate import process_generate_request, get_components_info
from Generate_image import (
    generate_component_image,
    generate_overall_image,
    generate_component_with_prompt,
    generate_overall_with_prompt,
    ComfyUIClient,
    load_workflow_template,
    get_images_from_folder,
    COMPONENT_WORKFLOW_PATH,
    OVERALL_WORKFLOW_PATH
)

print("[测试] 模块导入完成\n")

# 项目路径
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MEMORY_PATH = os.path.join(PROJECT_DIR, "object_nodes.json")
OPERATED_IMAGE_DIR = os.path.join(PROJECT_DIR, "Operated_image")
GENERATED_IMAGE_DIR = os.path.join(PROJECT_DIR, "generated_images")


# ==================== 初始化 ====================

def init_test():
    """初始化测试环境"""
    # 加载记忆数据库
    global memory_db
    memory_db = load_memory_from_json(MEMORY_PATH)

    # 显示当前部件列表
    print("="*50)
    print("当前记忆数据库状态")
    print("="*50)

    components = []
    for node_id, data in memory_db.items():
        if data.get('node_type') == 'COMPONENT':
            components.append(data.get('component_name', 'unknown'))

    print(f"部件数量: {len(components)}")
    print(f"部件列表: {components[:10]}{'...' if len(components) > 10 else ''}")
    print()


# ==================== 测试1: 纯语音 VLM 分析 ====================

def test_vlm_analysis_with_text(transcript_text: str, trigger_types: list = None):
    """
    测试纯语音 VLM 分析（模拟 handle_vlm_analysis_with_text）

    Args:
        transcript_text: 用户说的话
        trigger_types: 触发类型列表
    """
    print("\n" + "="*50)
    print("[测试1] 纯语音 VLM 分析")
    print("="*50)
    print(f"输入语音: '{transcript_text}'")

    trigger_types = trigger_types or ["语音输入触发"]

    # 1. 调用 VLM 分析
    print("\n[Step 1] VLM 分析...")
    vlm_response = vlm_chat_text_only(
        trigger_types=trigger_types,
        transcript_text=transcript_text
    )

    if not vlm_response:
        print("[失败] VLM 无返回")
        return None

    print(f"[成功] VLM 返回:\n{vlm_response[:300]}{'...' if len(vlm_response) > 300 else ''}")

    # 2. 解析 JSON
    print("\n[Step 2] 解析 JSON...")
    vlm_json = extract_and_parse_json(vlm_response)

    if not vlm_json:
        print("[失败] JSON 解析失败")
        return None

    print(f"[成功] 解析结果:")
    print(f"  - type: {vlm_json.get('type')}")
    print(f"  - label: {vlm_json.get('label')}")
    print(f"  - User intent: {vlm_json.get('User intent')}")
    print(f"  - User Speaking: {vlm_json.get('User Speaking')}")
    print(f"  - Behavior description: {vlm_json.get('Behavior description')}")

    # 3. 存入记忆（可选）
    print("\n[Step 3] 存入记忆（可选）...")
    try:
        node, node_type = process_vlm_result(
            vlm_result=vlm_json,
            memory_db=memory_db,
            component_image=None  # 无图像
        )
        if node:
            memory_db[node.node_id] = node.model_dump()
            save_memory_to_json(memory_db, MEMORY_PATH)
            print(f"[成功] 已存储为 {node_type}")
    except Exception as e:
        print(f"[跳过] 存入记忆: {e}")

    # 4. 重复检测
    print("\n[Step 4] 重复检测...")
    parsed, should_feedback, count = check_vlm_output(vlm_response, trigger_types[0])
    print(f"  重复计数: {count}")

    # 5. AI 反馈（如果触发）
    if should_feedback:
        print("\n[Step 5] 生成 AI 反馈...")
        component_name = vlm_json.get("label", "unknown")
        feedback = generate_ai_feedback(component_name, parsed, memory_db)
        print(f"  [AI 建议]:")
        print(f"  内容: {feedback.get('content', '')}")
        print(f"  分数: {feedback.get('scores', {})}")
        print(f"  总分: {feedback.get('total_score', 0)}")

    return vlm_json


# ==================== 测试2: 部件图像提示词生成 ====================

def test_component_prompt_generation(component_name: str):
    """
    测试部件图像提示词生成（模拟 handle_image_generation mode=1）

    Args:
        component_name: 部件名称
    """
    print("\n" + "="*50)
    print("[测试2] 部件图像提示词生成")
    print("="*50)
    print(f"部件名称: '{component_name}'")

    # 生成提示词
    print("\n[Step 1] 生成提示词...")
    prompt = process_generate_request(
        t=1,
        component_name=component_name,
        trigger_generate=1,
        memory_db=memory_db
    )

    if prompt:
        print(f"[成功] 生成的提示词:")
        print(f"  {prompt}")
    else:
        print("[失败] 未生成提示词")

    return prompt


# ==================== 测试3: 整体图像提示词生成 ====================

def test_overall_prompt_generation(
    component_image_mapping: dict = None,
    overall_image_index: int = None
):
    """
    测试整体图像提示词生成（模拟 handle_image_generation mode=2）

    Args:
        component_image_mapping: 部件图片索引映射，如 {"履带": 0, "把手": 1}
        overall_image_index: 整体图片索引
    """
    print("\n" + "="*50)
    print("[测试3] 整体图像提示词生成")
    print("="*50)
    print(f"部件图片映射: {component_image_mapping}")
    print(f"整体图片索引: {overall_image_index}")

    # 生成提示词
    print("\n[Step 1] 生成提示词...")
    prompt = process_generate_request(
        t=2,
        trigger_generate=1,
        memory_db=memory_db,
        component_image_mapping=component_image_mapping,
        overall_image_index=overall_image_index
    )

    if prompt:
        print(f"[成功] 生成的提示词:")
        print(f"  {prompt}")
    else:
        print("[失败] 未生成提示词")

    return prompt


# ==================== 测试4: 部件信息提取 ====================

def test_components_info():
    """测试部件结构/功能/待确定信息提取"""
    print("\n" + "="*50)
    print("[测试4] 部件信息提取")
    print("="*50)

    info = get_components_info(trigger=1, memory_db=memory_db)

    print(f"\n[结构信息] ({len(info['structure_info'])} 条):")
    for item in info['structure_info'][:5]:
        print(f"  - {item}")
    if len(info['structure_info']) > 5:
        print(f"  ... 共 {len(info['structure_info'])} 条")

    print(f"\n[功能信息] ({len(info['function_info'])} 条):")
    for item in info['function_info'][:5]:
        print(f"  - {item}")
    if len(info['function_info']) > 5:
        print(f"  ... 共 {len(info['function_info'])} 条")

    print(f"\n[待确定信息] ({len(info['uncertain_info'])} 条):")
    for item in info['uncertain_info'][:5]:
        print(f"  - {item}")
    if len(info['uncertain_info']) > 5:
        print(f"  ... 共 {len(info['uncertain_info'])} 条")

    return info


# ==================== 测试5: AI 反馈生成 ====================

def test_ai_feedback(component_name: str, user_intent: str, behavior_desc: str, user_speaking: str = ""):
    """
    测试 AI 反馈生成

    Args:
        component_name: 部件名称
        user_intent: 用户意图
        behavior_desc: 行为描述
        user_speaking: 用户说的话
    """
    print("\n" + "="*50)
    print("[测试5] AI 反馈生成")
    print("="*50)
    print(f"部件: {component_name}")
    print(f"意图: {user_intent}")
    print(f"行为: {behavior_desc}")
    print(f"说话: {user_speaking}")

    # 构造 VLM 输出格式
    vlm_output = {
        "User intent": user_intent,
        "Behavior description": behavior_desc,
        "User Speaking": user_speaking,
        "label": component_name
    }

    # 生成反馈
    print("\n[Step 1] 生成 AI 反馈...")
    feedback = generate_ai_feedback(component_name, vlm_output, memory_db)

    print(f"\n[AI 建议]:")
    print(f"  内容: {feedback.get('content', '')}")
    print(f"  各维度分数: {feedback.get('scores', {})}")
    print(f"  总分: {feedback.get('total_score', 0)}")

    return feedback


# ==================== 测试6: 部件图像生成（调用 ComfyUI）====================

def test_component_image_generation(component_name: str, image_path: str = None):
    """
    测试部件图像生成（模拟 handle_image_generation mode=1）

    Args:
        component_name: 部件名称
        image_path: 参考图片路径（如不提供，从 Operated_image 中读取第一张）
    """
    print("\n" + "="*50)
    print("[测试6] 部件图像生成（调用 ComfyUI）")
    print("="*50)
    print(f"部件名称: '{component_name}'")

    # 获取参考图片
    if image_path is None:
        images = get_images_from_folder(OPERATED_IMAGE_DIR)
        if len(images) == 0:
            print("[失败] Operated_image 中无图片，请先放入参考图片")
            return None
        image_path = images[0]
        print(f"使用 Operated_image 中的图片: {image_path}")
    else:
        print(f"参考图片: {image_path}")

    # 调用生成
    print("\n[Step 1] 生成提示词...")
    prompt = process_generate_request(
        t=1,
        component_name=component_name,
        trigger_generate=1,
        memory_db=memory_db
    )
    print(f"  提示词: {prompt}")

    print("\n[Step 2] 调用 ComfyUI 生成图像...")
    try:
        saved_paths = generate_component_image(
            prompt=prompt,
            image_path=image_path,
            workflow_path=COMPONENT_WORKFLOW_PATH,
            output_dir=GENERATED_IMAGE_DIR,
            save_name=component_name,
            timeout=120
        )

        if saved_paths:
            print(f"[成功] 生成的图片:")
            for path in saved_paths:
                print(f"  - {path}")
        else:
            print("[失败] 未生成图片")

        return saved_paths

    except Exception as e:
        print(f"[失败] 生成异常: {e}")
        return None


# ==================== 测试7: 整体图像生成（调用 ComfyUI）====================

def test_overall_image_generation(image_paths: list = None, component_image_mapping: dict = None):
    """
    测试整体图像生成（模拟 handle_image_generation mode=2）

    Args:
        image_paths: 部件图片路径列表（如不提供，从 Operated_image 中读取）
        component_image_mapping: 部件图片索引映射
    """
    print("\n" + "="*50)
    print("[测试7] 整体图像生成（调用 ComfyUI）")
    print("="*50)

    # 获取部件图片
    if image_paths is None:
        image_paths = get_images_from_folder(OPERATED_IMAGE_DIR, max_images=9)
        print(f"使用 Operated_image 中的图片: {len(image_paths)} 张")
        for i, path in enumerate(image_paths):
            print(f"  [{i}] {path}")

    if len(image_paths) == 0:
        print("[失败] 无部件图片，请先放入 Operated_image 文件夹")
        return None

    # 构建部件映射（如果没有提供）
    if component_image_mapping is None:
        # 获取部件名称列表
        components = []
        for node_id, data in memory_db.items():
            if data.get('node_type') == 'COMPONENT':
                components.append(data.get('component_name', 'unknown'))

        component_image_mapping = {}
        for i in range(min(len(image_paths), len(components))):
            component_image_mapping[components[i]] = i

        print(f"自动构建部件映射: {component_image_mapping}")

    print(f"部件映射: {component_image_mapping}")

    # 调用生成
    print("\n[Step 1] 生成提示词...")
    prompt = process_generate_request(
        t=2,
        trigger_generate=1,
        memory_db=memory_db,
        component_image_mapping=component_image_mapping
    )
    print(f"  提示词: {prompt[:200]}{'...' if len(prompt) > 200 else ''}")

    print("\n[Step 2] 调用 ComfyUI 生成图像...")
    try:
        saved_paths = generate_overall_image(
            prompt=prompt,
            image_paths=image_paths,
            workflow_path=OVERALL_WORKFLOW_PATH,
            output_dir=GENERATED_IMAGE_DIR,
            save_name="overall",
            timeout=120
        )

        if saved_paths:
            print(f"[成功] 生成的图片:")
            for path in saved_paths:
                print(f"  - {path}")
        else:
            print("[失败] 未生成图片")

        return saved_paths

    except Exception as e:
        print(f"[失败] 生成异常: {e}")
        return None


# ==================== 测试8: 检查 ComfyUI 连接 ====================

def test_comfyui_connection():
    """测试 ComfyUI 服务连接"""
    print("\n" + "="*50)
    print("[测试0] ComfyUI 服务连接检查")
    print("="*50)

    import requests

    try:
        client = ComfyUIClient()
        response = requests.get(f"{client.base_url}/system_stats", timeout=5)

        if response.status_code == 200:
            print(f"[成功] ComfyUI 运行正常 ({client.base_url})")
            return True
        else:
            print(f"[失败] ComfyUI 响应异常: {response.status_code}")
            return False

    except Exception as e:
        print(f"[失败] ComfyUI 未运行: {e}")
        print("  请确保 ComfyUI 服务已启动 (localhost:8000)")
        return False


# ==================== 主测试入口 ====================

def run_all_tests():
    """运行所有测试（不含图像生成）"""
    init_test()

    # 测试1: 纯语音 VLM 分析
    print("\n" + "="*60)
    print("开始测试 1/5")
    print("="*60)
    test_vlm_analysis_with_text(
        transcript_text="我想把履带的外形做得更圆润一些，不要太笨重"
    )

    # 测试2: 部件图像提示词生成
    print("\n" + "="*60)
    print("开始测试 2/5")
    print("="*60)
    test_component_prompt_generation("履带")

    # 测试3: 整体图像提示词生成
    print("\n" + "="*60)
    print("开始测试 3/5")
    print("="*60)
    test_overall_prompt_generation(
        component_image_mapping={"履带": 0, "把手": 1, "摄像头": 2},
        overall_image_index=3
    )

    # 测试4: 部件信息提取
    print("\n" + "="*60)
    print("开始测试 4/5")
    print("="*60)
    test_components_info()

    # 测试5: AI 反馈生成
    print("\n" + "="*60)
    print("开始测试 5/5")
    print("="*60)
    test_ai_feedback(
        component_name="履带",
        user_intent="Appearance design",
        behavior_desc="用户正在查看履带的外形设计",
        user_speaking="这个履带看起来有点笨重，我想让它更轻便"
    )

    print("\n" + "="*60)
    print("所有基础测试完成")
    print("="*60)


def run_all_tests_with_image():
    """运行所有测试（包含图像生成）"""
    init_test()

    # 先检查 ComfyUI
    if not test_comfyui_connection():
        print("\n[警告] ComfyUI 未运行，跳过图像生成测试")
        run_all_tests()
        return

    # 测试1: 纯语音 VLM 分析
    print("\n" + "="*60)
    print("开始测试 1/7")
    print("="*60)
    test_vlm_analysis_with_text(
        transcript_text="我想把履带的外形做得更圆润一些，不要太笨重"
    )

    # 测试2: 部件图像提示词生成
    print("\n" + "="*60)
    print("开始测试 2/7")
    print("="*60)
    test_component_prompt_generation("履带")

    # 测试3: 整体图像提示词生成
    print("\n" + "="*60)
    print("开始测试 3/7")
    print("="*60)
    test_overall_prompt_generation(
        component_image_mapping={"履带": 0, "把手": 1, "摄像头": 2},
        overall_image_index=3
    )

    # 测试4: 部件信息提取
    print("\n" + "="*60)
    print("开始测试 4/7")
    print("="*60)
    test_components_info()

    # 测试5: AI 反馈生成
    print("\n" + "="*60)
    print("开始测试 5/7")
    print("="*60)
    test_ai_feedback(
        component_name="履带",
        user_intent="Appearance design",
        behavior_desc="用户正在查看履带的外形设计",
        user_speaking="这个履带看起来有点笨重，我想让它更轻便"
    )

    # 测试6: 部件图像生成
    print("\n" + "="*60)
    print("开始测试 6/7")
    print("="*60)
    test_component_image_generation("履带")

    # 测试7: 整体图像生成
    print("\n" + "="*60)
    print("开始测试 7/7")
    print("="*60)
    test_overall_image_generation()

    print("\n" + "="*60)
    print("所有测试完成（含图像生成）")
    print("="*60)


# ==================== 单独测试入口 ====================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="模拟外部输入测试")
    parser.add_argument("--all", action="store_true", help="运行所有基础测试（不含图像生成）")
    parser.add_argument("--full", action="store_true", help="运行所有测试（包含图像生成）")
    parser.add_argument("--vlm", type=str, help="测试 VLM 分析，传入语音文本")
    parser.add_argument("--component", type=str, help="测试部件提示词生成，传入部件名")
    parser.add_argument("--overall", action="store_true", help="测试整体提示词生成")
    parser.add_argument("--info", action="store_true", help="测试部件信息提取")
    parser.add_argument("--feedback", type=str, nargs=4, metavar=("部件", "意图", "行为", "说话"),
                        help="测试 AI 反馈生成")
    parser.add_argument("--gen-component", type=str, metavar="部件名",
                        help="测试部件图像生成，传入部件名")
    parser.add_argument("--gen-overall", action="store_true",
                        help="测试整体图像生成")
    parser.add_argument("--check-comfyui", action="store_true",
                        help="检查 ComfyUI 服务连接")

    args = parser.parse_args()

    init_test()

    if args.full:
        run_all_tests_with_image()

    elif args.all:
        run_all_tests()

    elif args.vlm:
        test_vlm_analysis_with_text(args.vlm)

    elif args.component:
        test_component_prompt_generation(args.component)

    elif args.overall:
        test_overall_prompt_generation(
            component_image_mapping={"履带": 0, "把手": 1},
            overall_image_index=2
        )

    elif args.info:
        test_components_info()

    elif args.feedback:
        test_ai_feedback(
            component_name=args.feedback[0],
            user_intent=args.feedback[1],
            behavior_desc=args.feedback[2],
            user_speaking=args.feedback[3]
        )

    elif args.gen_component:
        test_component_image_generation(args.gen_component)

    elif args.gen_overall:
        test_overall_image_generation()

    elif args.check_comfyui:
        test_comfyui_connection()

    else:
        # 默认运行所有测试
        print("未指定测试，运行所有测试（含图像生成）...")
        run_all_tests_with_image()