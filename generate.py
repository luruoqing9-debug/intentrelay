"""
generate.py - 图像生成模块
包含：部件/整体图像生成提示词生成
"""

import sys
import os
import json

# 禁用警告
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
import warnings
warnings.filterwarnings('ignore')

from typing import Literal
from record import client, extract_and_parse_json, text_encoder
import numpy as np

# ========== 生成模式定义 ==========

GenerateMode = Literal[1, 2]
# 1: 部件生成
# 2: 整体生成


# ========== 部件名称匹配 ==========

def find_component_in_memory(component_name: str, memory_db: dict) -> dict:
    """
    在记忆数据库中查找部件，使用 LLM 判断名称是否指向同一部件。

    Args:
        component_name: 用户输入的部件名称
        memory_db: 记忆数据库

    Returns:
        找到的部件记忆数据，没找到返回 None
    """
    # 获取所有部件节点
    component_nodes = {}
    for node_id, data in memory_db.items():
        if data.get('node_type') == 'COMPONENT':
            stored_name = data.get('component_name', '')
            component_nodes[stored_name] = data

    if not component_nodes:
        print("[Generate] No components in memory")
        return None

    # 使用 LLM 判断名称匹配
    stored_names = list(component_nodes.keys())
    match_prompt = f'''
请判断用户输入的部件名称是否与以下已存储的部件名称中的某一个指向同一个部件。

用户输入：{component_name}

已存储部件：{json.dumps(stored_names, ensure_ascii=False)}

判断规则：
1. 如果名称完全相同，返回该名称
2. 如果是同义词或方言变体（如"车座"和"车座子"、"把手"和"手柄"），返回最匹配的存储名称
3. 如果没有匹配的，返回 "无匹配"

只输出结果，不要解释。输出格式：
{{"match": "匹配的部件名称 或 无匹配"}}
'''

    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[match_prompt]
    )

    result = extract_and_parse_json(response.text)
    if result is None:
        print("[Generate] Failed to parse LLM response")
        return None

    match_name = result.get('match', '无匹配')
    print(f"[Generate] LLM match result: '{component_name}' -> '{match_name}'")

    if match_name == '无匹配' or match_name not in component_nodes:
        return None

    return component_nodes[match_name]


def get_component_memory_text(component_data: dict) -> str:
    """
    将部件记忆数据转换为文本描述。

    Args:
        component_data: 部件节点数据

    Returns:
        部件记忆文本描述
    """
    component_name = component_data.get('component_name', '未知部件')

    # 提取各类描述
    appearance_list = component_data.get('appearance_descriptions', [])
    function_list = component_data.get('function_descriptions', [])
    structure_list = component_data.get('structure_descriptions', [])

    appearance_text = "".join([d.get('content', '') for d in appearance_list])
    function_text = "".join([d.get('content', '') for d in function_list])
    structure_text = "".join([d.get('content', '') for d in structure_list])

    return f'''
部件名称：{component_name}
外形描述：{appearance_text if appearance_text else '暂无'}
功能描述：{function_text if function_text else '暂无'}
结构描述：{structure_text if structure_text else '暂无'}
'''


def get_overall_memory_text(memory_db: dict) -> str:
    """
    获取整体产品的记忆文本。

    Args:
        memory_db: 记忆数据库

    Returns:
        整体记忆文本描述
    """
    overall_nodes = []
    for node_id, data in memory_db.items():
        if data.get('node_type') == 'OVERALL':
            overall_nodes.append(data)

    if not overall_nodes:
        return "暂无整体设计记忆"

    # 合并所有整体节点信息
    overall_text = ""
    for node in overall_nodes:
        design_background = node.get('design_background', '')
        if design_background:
            overall_text += f"设计背景：{design_background}\n"

        appearance_list = node.get('appearance_descriptions', [])
        function_list = node.get('function_descriptions', [])
        structure_list = node.get('structure_descriptions', [])

        for d in appearance_list:
            overall_text += f"整体外形：{d.get('content', '')}\n"
        for d in function_list:
            overall_text += f"整体功能：{d.get('content', '')}\n"
        for d in structure_list:
            overall_text += f"整体结构：{d.get('content', '')}\n"

    return overall_text if overall_text else "暂无整体设计记忆"


def get_all_components_text(memory_db: dict) -> str:
    """
    获取所有部件的记忆文本。

    Args:
        memory_db: 记忆数据库

    Returns:
        所有部件记忆文本描述
    """
    all_text = ""
    for node_id, data in memory_db.items():
        if data.get('node_type') == 'COMPONENT':
            all_text += get_component_memory_text(data) + "\n"

    return all_text if all_text else "暂无部件设计记忆"


# ========== 提示词生成 ==========

def generate_component_prompt(
    component_name: str,
    memory_db: dict,
    trigger_generate: Literal[0, 1] = 1
) -> str:
    """
    生成部件图像的提示词（t=1）。

    Args:
        component_name: 用户输入的部件名称
        memory_db: 记忆数据库
        trigger_generate: 是否触发生成（1=触发，0=不触发）

    Returns:
        图像生成提示词，不触发时返回空字符串
    """
    if trigger_generate == 0:
        print("[Generate] Not triggered (trigger_generate=0)")
        return ""

    print(f"[Generate] Component mode (t=1), searching for '{component_name}'...")

    # 查找部件记忆
    component_data = find_component_in_memory(component_name, memory_db)

    # 获取整体记忆
    overall_text = get_overall_memory_text(memory_db)

    if component_data:
        # 找到部件，结合整体记忆和部件记忆
        component_text = get_component_memory_text(component_data)

        prompt_source = f'''
## 整体产品信息
{overall_text}

## 目标部件信息
{component_text}
'''
        print(f"[Generate] Found component '{component_data.get('component_name')}'")
    else:
        # 没找到部件，只用整体记忆
        prompt_source = f'''
## 整体产品信息
{overall_text}

## 目标部件
部件名称：{component_name}
（该部件暂无设计记忆，请根据整体风格进行设计）
'''
        print(f"[Generate] Component '{component_name}' not found in memory")

    # 使用 LLM 生成图像提示词
    generate_prompt = f'''
你是一个专业的产品设计图像生成提示词编写助手。请根据以下设计记忆信息，为图像生成模型编写一个详细、准确的提示词。

{prompt_source}

## 提示词编写要求
1. 描述清晰、具体，便于图像生成模型理解
2. 包含材质、颜色、形状、细节等视觉特征
3. 保持与整体产品风格的一致性
4. 使用简洁的语言，不超过100字
5. 直接输出提示词，不要有任何解释或前缀

请输出图像生成提示词：
'''

    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[generate_prompt]
    )

    image_prompt = response.text.strip()
    print(f"[Generate] Generated prompt: '{image_prompt}'")

    return image_prompt


def generate_overall_prompt(
    memory_db: dict,
    trigger_generate: Literal[0, 1] = 1,
    component_image_mapping: dict = None,
    overall_image_index: int = None
) -> str:
    """
    生成整体图像的提示词（t=2）。

    Args:
        memory_db: 记忆数据库
        trigger_generate: 是否触发生成（1=触发，0=不触发）
        component_image_mapping: 部件图片索引映射，格式如 {"车架": 1, "车轮": 2, ...}
        overall_image_index: 整体图片的索引，如 7（整体图是模型制作的粗糙图，作为结构参考）

    Returns:
        图像生成提示词，不触发时返回空字符串
    """
    if trigger_generate == 0:
        print("[Generate] Not triggered (trigger_generate=0)")
        return ""

    print("[Generate] Overall mode (t=2), combining all memories...")

    # 获取整体记忆
    overall_text = get_overall_memory_text(memory_db)

    # 获取所有部件记忆并构建图片引用信息
    all_components_text = get_all_components_text(memory_db)

    # 构建图片引用信息
    image_reference_text = ""
    if component_image_mapping:
        component_refs = []
        for comp_name, img_idx in component_image_mapping.items():
            component_refs.append(f"- {comp_name}：[@图{img_idx}]")
        image_reference_text += f"## 部件图片索引\n" + "\n".join(component_refs) + "\n\n"
        print(f"[Generate] Component image mapping: {component_image_mapping}")

    if overall_image_index is not None:
        image_reference_text += f"## 整体图片索引\n- 整体结构参考图：[@图{overall_image_index}]（此图较粗糙，仅作结构和外形参考）\n\n"
        print(f"[Generate] Overall image index: {overall_image_index}")

    prompt_source = f'''
{image_reference_text}## 整体产品信息
{overall_text}

## 所有部件信息
{all_components_text}
'''

    # 使用 LLM 生成图像提示词
    generate_prompt = f'''
你是一个专业的产品设计图像生成提示词编写助手。

你需要根据以下信息，编写一个用于生成高质量整体产品图像的提示词。

{prompt_source}

## 关键要求
1. **必须引用图片**：提示词中必须使用 [@图N] 格式引用对应的部件图片（如 [@图0]、[@图1]）
2. **部件图为主**：部件图片是高质量参考，应详细描述各部件的特征
3. **整体图为辅**：如有整体图片索引，它只是粗糙的结构参考，用于确定组装方式和整体布局
4. **输出目标**：生成的是整体产品的最终效果图

## 提示词格式示例
[@图 0]至[@图 3]中显示的组件组合在一起，形成了一个现代、智能的社区移动送货机器人，其特点是友好、时尚和未来主义风格。
货物存储：主体配备了[@图 0]的多隔间耐候存储模块，具有透明的观察窗和安全的电子锁。
交互界面：[@图 1]的彩色触摸屏界面以符合人体工程学的高度集成在侧面，专为直观的二维码扫描和用户交互而设计。
移动底盘：底座是一个坚固、低矮的底盘，如[@图 2]所示，具有全向车轮，可在各种住宅通道和表面导航。
总体布局：车辆的空间结构遵循[@图 3]的布局设计，确保储物舱、交互区和底盘之间的比例平衡。


## 编写规则
1. 开头说明所有部件共同构成什么产品
2. 分句描述各部件特征，每句引用对应的部件图片
3. 如有整体图片索引，提及它作为结构参考
4. 结尾描述整体效果和画面要求
5. 不超过200字
6. 直接输出提示词，不要有任何解释或前缀

请输出图像生成提示词：
'''

    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[generate_prompt]
    )

    image_prompt = response.text.strip()
    print(f"[Generate] Generated prompt: '{image_prompt}'")

    return image_prompt


# ========== 统一生成接口 ==========

def process_generate_request(
    t: GenerateMode,
    component_name: str = None,
    trigger_generate: Literal[0, 1] = 1,
    memory_db: dict = None,
    memory_path: str = None,
    component_image_mapping: dict = None,
    overall_image_index: int = None
) -> str:
    """
    统一图像生成接口。

    Args:
        t: 生成模式（1=部件生成，2=整体生成）
        component_name: 部件名称（t=1 时必需）
        trigger_generate: 是否触发生成（1=触发，0=不触发）
        memory_db: 记忆数据库（为 None 则从文件加载）
        memory_path: 记忆文件路径
        component_image_mapping: 部件图片索引映射（t=2时使用），格式如 {"车架": 1, "车轮": 2}
        overall_image_index: 整体图片索引（t=2时使用），如 7

    Returns:
        图像生成提示词，不触发时返回空字符串
    """
    # 加载记忆数据库
    if memory_db is None:
        if memory_path is None:
            memory_path = os.path.join(os.path.dirname(__file__), "object_nodes.json")

        if os.path.exists(memory_path):
            with open(memory_path, 'r', encoding='utf-8') as f:
                memory_db = json.load(f)
        else:
            memory_db = {}
            print(f"[Generate] No memory file at '{memory_path}'")

    # 根据模式生成
    if t == 1:
        # 部件生成
        if component_name is None:
            print("[Generate] Error: component_name required for t=1")
            return ""

        return generate_component_prompt(component_name, memory_db, trigger_generate)

    elif t == 2:
        # 整体生成
        return generate_overall_prompt(
            memory_db,
            trigger_generate,
            component_image_mapping,
            overall_image_index
        )

    else:
        print(f"[Generate] Error: Invalid t value: {t}")
        return ""


# ========== 部件结构/功能信息提取 ==========

def get_components_structure_info(
    trigger: Literal[0, 1] = 1,
    memory_db: dict = None,
    memory_path: str = None
) -> list:
    """
    提取所有部件的结构信息。

    Args:
        trigger: 是否触发（1=触发，0=不触发）
        memory_db: 记忆数据库（为 None 则从文件加载）
        memory_path: 记忆文件路径

    Returns:
        list: 部件结构信息列表，格式如 ["履带：采用齿轮连接结构", "把手：螺纹连接", ...]
        不触发时返回空列表
    """
    if trigger == 0:
        print("[Structure Info] Not triggered (trigger=0)")
        return []

    # 加载记忆数据库
    if memory_db is None:
        if memory_path is None:
            memory_path = os.path.join(os.path.dirname(__file__), "object_nodes.json")

        if os.path.exists(memory_path):
            with open(memory_path, 'r', encoding='utf-8') as f:
                memory_db = json.load(f)
        else:
            memory_db = {}
            print(f"[Structure Info] No memory file at '{memory_path}'")
            return []

    # 检索所有部件的结构描述
    structure_list = []
    for node_id, data in memory_db.items():
        if data.get('node_type') == 'COMPONENT':
            component_name = data.get('component_name', '未知部件')
            structure_descriptions = data.get('structure_descriptions', [])

            # 提取有内容的结构描述
            for desc in structure_descriptions:
                content = desc.get('content', '')
                if content and content.strip():
                    structure_list.append(f"{component_name}：{content}")

    print(f"[Structure Info] Found {len(structure_list)} structure entries")
    return structure_list


def get_components_function_info(
    trigger: Literal[0, 1] = 1,
    memory_db: dict = None,
    memory_path: str = None
) -> list:
    """
    提取所有部件的功能信息。

    Args:
        trigger: 是否触发（1=触发，0=不触发）
        memory_db: 记忆数据库（为 None 则从文件加载）
        memory_path: 记忆文件路径

    Returns:
        list: 部件功能信息列表，格式如 ["履带：用于在社区间平稳行走", "把手：方便握持", ...]
        不触发时返回空列表
    """
    if trigger == 0:
        print("[Function Info] Not triggered (trigger=0)")
        return []

    # 加载记忆数据库
    if memory_db is None:
        if memory_path is None:
            memory_path = os.path.join(os.path.dirname(__file__), "object_nodes.json")

        if os.path.exists(memory_path):
            with open(memory_path, 'r', encoding='utf-8') as f:
                memory_db = json.load(f)
        else:
            memory_db = {}
            print(f"[Function Info] No memory file at '{memory_path}'")
            return []

    # 检索所有部件的功能描述
    function_list = []
    for node_id, data in memory_db.items():
        if data.get('node_type') == 'COMPONENT':
            component_name = data.get('component_name', '未知部件')
            function_descriptions = data.get('function_descriptions', [])

            # 提取有内容的功能描述
            for desc in function_descriptions:
                content = desc.get('content', '')
                if content and content.strip():
                    function_list.append(f"{component_name}：{content}")

    print(f"[Function Info] Found {len(function_list)} function entries")
    return function_list


def get_components_uncertain_info(
    trigger: Literal[0, 1] = 1,
    memory_db: dict = None,
    memory_path: str = None
) -> list:
    """
    提取所有部件的待确定信息（status=0的描述）。

    Args:
        trigger: 是否触发（1=触发，0=不触发）
        memory_db: 记忆数据库（为 None 则从文件加载）
        memory_path: 记忆文件路径

    Returns:
        list: 部件待确定信息列表，格式如 ["履带：还没想好外形设计的风格", ...]
        不触发时返回空列表
    """
    if trigger == 0:
        print("[Uncertain Info] Not triggered (trigger=0)")
        return []

    # 加载记忆数据库
    if memory_db is None:
        if memory_path is None:
            memory_path = os.path.join(os.path.dirname(__file__), "object_nodes.json")

        if os.path.exists(memory_path):
            with open(memory_path, 'r', encoding='utf-8') as f:
                memory_db = json.load(f)
        else:
            memory_db = {}
            print(f"[Uncertain Info] No memory file at '{memory_path}'")
            return []

    # 检索所有部件中 status=0 的描述
    uncertain_list = []
    for node_id, data in memory_db.items():
        if data.get('node_type') == 'COMPONENT':
            component_name = data.get('component_name', '未知部件')

            # 检查三类描述中 status=0 的内容
            for desc_type in ['appearance_descriptions', 'function_descriptions', 'structure_descriptions']:
                descriptions = data.get(desc_type, [])
                for desc in descriptions:
                    status = desc.get('status', 1)  # 默认为确定状态
                    content = desc.get('content', '')
                    if status == 0 and content and content.strip():
                        uncertain_list.append(f"{component_name}：{content}")

    print(f"[Uncertain Info] Found {len(uncertain_list)} uncertain entries")
    return uncertain_list


def get_components_info(
    trigger: Literal[0, 1] = 1,
    memory_db: dict = None,
    memory_path: str = None
) -> dict:
    """
    统一接口：同时获取部件结构信息、功能信息和待确定信息。

    Args:
        trigger: 是否触发（1=触发，0=不触发）
        memory_db: 记忆数据库（为 None 则从文件加载）
        memory_path: 记忆文件路径

    Returns:
        dict: {
            "structure_info": ["履带：采用齿轮连接结构", ...],
            "function_info": ["履带：用于在社区间平稳行走", ...],
            "uncertain_info": ["履带：还没想好外形设计的风格", ...]
        }
        不触发时返回空列表
    """
    structure_info = get_components_structure_info(trigger, memory_db, memory_path)
    function_info = get_components_function_info(trigger, memory_db, memory_path)
    uncertain_info = get_components_uncertain_info(trigger, memory_db, memory_path)

    return {
        "structure_info": structure_info,
        "function_info": function_info,
        "uncertain_info": uncertain_info
    }


# ========== 测试 ==========

if __name__ == "__main__":
    print("=== Generate Module Test ===")

    # 测试部件生成
    print("\n--- Test 1: Component generation (t=1) ---")
    prompt1 = process_generate_request(
        t=1,
        component_name="履带",
        trigger_generate=1
    )
    print(f"Result: {prompt1}")

    # 测试整体生成（带图片索引）
    print("\n--- Test 2: Overall generation with image mapping (t=2) ---")
    prompt2 = process_generate_request(
        t=2,
        trigger_generate=1,
        component_image_mapping={"车架": 1, "车轮": 2, "车座": 3, "车篮": 4, "水杯架": 5, "车把手": 6},
        overall_image_index=7
    )
    print(f"Result: {prompt2}")

    # 测试整体生成（不带图片索引）
    print("\n--- Test 2b: Overall generation without image mapping ---")
    prompt2b = process_generate_request(
        t=2,
        trigger_generate=1
    )
    print(f"Result: {prompt2b}")

    # 测试部件结构/功能/待确定信息
    print("\n--- Test 3: Components info ---")
    info = get_components_info(trigger=1)
    print(f"Structure info: {info['structure_info']}")
    print(f"Function info: {info['function_info']}")
    print(f"Uncertain info: {info['uncertain_info']}")

    # 测试不触发
    print("\n--- Test 4: Not triggered ---")
    info2 = get_components_info(trigger=0)
    print(f"Result: {info2}")