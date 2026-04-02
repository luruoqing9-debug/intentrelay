"""
Feedback.py - AI反馈模块
包含：重复检测、反馈触发逻辑
"""

import json
import numpy as np
from typing import Dict, Any, Tuple, Optional

# 从 record.py 导入需要的函数
from record import text_encoder, extract_and_parse_json, client


# ========== 重复检测配置 ==========

REPEAT_SIMILARITY_THRESHOLD = 0.8
REPEAT_COUNT_THRESHOLD = 5


# ========== 重复检测状态存储 ==========

# 存储每个部件的上一次 VLM 输出和计数器
# 结构: { "部件名": {"last_output": {...}, "count": 0} }
_repeat_detection_state: Dict[str, Dict[str, Any]] = {}


# ========== 相似度计算函数 ==========

def _calculate_text_similarity(text1: str, text2: str) -> float:
    """计算两个文本的相似度（使用 embedding cosine similarity）"""
    if not text1 and not text2:
        return 1.0  # 两个都为空，认为相似
    if not text1 or not text2:
        return 0.0  # 一个为空一个不为空，不相似

    emb1 = text_encoder(text1)
    emb2 = text_encoder(text2)
    similarity = float(np.dot(emb1, emb2))
    return similarity


def _calculate_output_similarity(output1: dict, output2: dict) -> float:
    """
    计算两次 VLM 输出的相似度。

    比较内容：trigger_type, label, User Speaking, Behavior description, User intent
    """
    similarities = []

    # 1. trigger_type 比较（完全匹配）
    trigger1 = output1.get("trigger_type", "")
    trigger2 = output2.get("trigger_type", "")
    if trigger1 == trigger2:
        similarities.append(1.0)
    else:
        similarities.append(0.0)

    # 2. label 比较（文本相似度）
    label_sim = _calculate_text_similarity(
        output1.get("label", ""),
        output2.get("label", "")
    )
    similarities.append(label_sim)

    # 3. User Speaking 比较（特殊规则：空也认为相似）
    speaking1 = output1.get("User Speaking", "")
    speaking2 = output2.get("User Speaking", "")
    if not speaking1 and not speaking2:
        speaking_sim = 1.0
    else:
        speaking_sim = _calculate_text_similarity(speaking1, speaking2)
    similarities.append(speaking_sim)

    # 4. Behavior description 比较（文本相似度）
    behavior_sim = _calculate_text_similarity(
        output1.get("Behavior description", ""),
        output2.get("Behavior description", "")
    )
    similarities.append(behavior_sim)

    # 5. User intent 比较（完全匹配）
    intent1 = output1.get("User intent", "")
    intent2 = output2.get("User intent", "")
    if intent1 == intent2:
        similarities.append(1.0)
    else:
        similarities.append(0.0)

    # 返回平均相似度
    return sum(similarities) / len(similarities)


# ========== 重复检测主函数 ==========

def check_vlm_output(
    vlm_response: str,
    trigger_type: str
) -> Tuple[Optional[dict], bool, int]:
    """
    解析 VLM 输出并检测重复，返回解析结果和是否触发反馈。

    Args:
        vlm_response: VLM 返回的 JSON 字符串
        trigger_type: 当前触发类型

    Returns:
        Tuple[dict|None, bool, int]:
            - 解析后的 VLM 输出（解析失败则为 None）
            - 是否触发 AI 反馈
            - 当前重复计数
    """
    # 1. 解析 JSON
    parsed_output = extract_and_parse_json(vlm_response)
    if parsed_output is None:
        print("[Feedback] Failed to parse VLM response")
        return None, False, 0

    # 2. 添加 trigger_type 到输出中（用于比较）
    parsed_output["trigger_type"] = trigger_type

    # 3. 获取部件名（label）
    component_name = parsed_output.get("label", "unknown")

    # 4. 调用重复检测
    should_trigger, count = check_repeat_and_update(component_name, parsed_output)

    return parsed_output, should_trigger, count


def check_repeat_and_update(
    component_name: str,
    current_output: dict
) -> Tuple[bool, int]:
    """
    检查当前输出是否与上一次相似，更新状态，返回是否触发反馈。

    Args:
        component_name: 部件名称（来自 VLM 输出的 label）
        current_output: 当前 VLM 输出（包含 trigger_type, label, User Speaking,
                        Behavior description, User intent）

    Returns:
        Tuple[bool, int]: (是否触发AI反馈, 当前计数)
    """
    global _repeat_detection_state

    # 初始化该部件的状态
    if component_name not in _repeat_detection_state:
        _repeat_detection_state[component_name] = {
            "last_output": current_output,
            "count": 1
        }
        print(f"[Feedback] New component '{component_name}', count: 1")
        return False, 1

    state = _repeat_detection_state[component_name]
    last_output = state["last_output"]

    # 计算相似度
    similarity = _calculate_output_similarity(last_output, current_output)
    print(f"[Feedback] '{component_name}' similarity: {similarity:.4f}")

    if similarity > REPEAT_SIMILARITY_THRESHOLD:
        # 相似，计数器 +1
        state["count"] += 1
        state["last_output"] = current_output
        print(f"[Feedback] Similar! Count: {state['count']}")

        # 检查是否超过阈值
        if state["count"] > REPEAT_COUNT_THRESHOLD:
            print(f"[Feedback] ⚠️ Threshold exceeded! Triggering AI feedback for '{component_name}'")
            return True, state["count"]

        return False, state["count"]
    else:
        # 不相似，计数器清零
        state["count"] = 1
        state["last_output"] = current_output
        print(f"[Feedback] Different! Count reset to 1")
        return False, 1


def get_repeat_count(component_name: str) -> int:
    """获取指定部件的当前重复计数"""
    if component_name in _repeat_detection_state:
        return _repeat_detection_state[component_name]["count"]
    return 0


def reset_repeat_count(component_name: str = None):
    """
    重置重复计数。

    Args:
        component_name: 指定部件名，为 None 则重置所有
    """
    global _repeat_detection_state

    if component_name is None:
        _repeat_detection_state = {}
        print("[Feedback] All counts reset")
    elif component_name in _repeat_detection_state:
        _repeat_detection_state[component_name]["count"] = 0
        print(f"[Feedback] '{component_name}' count reset")


# ========== AI 反馈生成函数 ==========

def get_component_memory(component_name: str, memory_db: dict = None) -> dict:
    """
    从记忆数据库中获取指定部件的记忆信息。

    Args:
        component_name: 部件名称
        memory_db: 记忆数据库（为 None 则从文件加载）

    Returns:
        部件记忆信息 dict，包含 appearance, function, structure 描述
    """
    import os

    # 如果没有传入 memory_db，从文件加载
    if memory_db is None:
        memory_path = os.path.join(os.path.dirname(__file__), "object_nodes.json")
        if os.path.exists(memory_path):
            with open(memory_path, 'r', encoding='utf-8') as f:
                memory_db = json.load(f)
        else:
            return {}

    # 查找匹配的部件节点
    for node_id, data in memory_db.items():
        if data.get('node_type') == 'COMPONENT':
            stored_name = data.get('component_name', '')
            # 名字匹配（忽略大小写）
            if stored_name.lower() == component_name.lower():
                return {
                    "component_name": stored_name,
                    "appearance": [d.get("content", "") for d in data.get("appearance_descriptions", [])],
                    "function": [d.get("content", "") for d in data.get("function_descriptions", [])],
                    "structure": [d.get("content", "") for d in data.get("structure_descriptions", [])],
                    "design_background": data.get("design_background")
                }

    # 没找到，返回空
    return {"component_name": component_name, "appearance": [], "function": [], "structure": []}


def generate_ai_feedback(component_name: str, vlm_output: dict, memory_db: dict = None) -> str:
    """
    生成 AI 主动反馈内容。

    根据部件记忆和当前 VLM 输出，针对用户意图生成建议。

    Args:
        component_name: 部件名称
        vlm_output: VLM 分析结果（包含 User intent, Behavior description 等）
        memory_db: 记忆数据库（为 None 则从文件加载）

    Returns:
        AI 反馈文本
    """
    from record import client

    # 1. 获取部件记忆
    component_memory = get_component_memory(component_name, memory_db)

    # 2. 提取当前信息
    user_intent = vlm_output.get("User intent", "")
    behavior_description = vlm_output.get("Behavior description", "")
    user_speaking = vlm_output.get("User Speaking", "")

    # 3. 根据 User intent 构建不同的提示
    intent_prompts = {
        "Appearance design": "用户正在关注产品的外观设计，包括外形、尺寸、材质、颜色、表面纹理等。请基于已有记忆提供外观设计相关的建议。",
        "Functional concept": "用户正在思考产品的功能目标和使用方式。请基于已有记忆提供功能设计相关的建议。",
        "Structural design": "用户正在考虑产品的结构关系、部件连接、布局等。请基于已有记忆提供结构设计相关的建议。",
        "Still-uncertain Idea Exploration": "用户正在进行探索性思考，想法还不够确定。请帮助用户梳理思路，提供引导性问题或建议。",
        "Design Background supplement": "用户正在补充设计背景信息，如目标用户、使用场景等。请帮助用户完善背景信息。"
    }

    intent_context = intent_prompts.get(user_intent, "用户正在思考设计相关的问题。")

    # 4. 构建 prompt
    prompt = f'''
你是一个专业的设计助手，正在协助用户进行产品设计。用户已经连续多次关注同一个部件，你需要主动提供有价值的建议。

## 当前关注部件
部件名称：{component_name}

## 部件已有记忆
- 外形描述：{component_memory.get("appearance", [])}
- 功能描述：{component_memory.get("function", [])}
- 结构描述：{component_memory.get("structure", [])}

## 用户当前状态
- 用户意图类型：{user_intent}
- 行为描述：{behavior_description}
- 用户说的话：{user_speaking if user_speaking else "（用户没有说话）"}

## 你的任务
{intent_context}

## 输出要求
1. 简洁友好，不超过100字
2. 基于已有记忆直接给出具体建议，不要泛泛而谈
3. 如果记忆信息不足，提出具体的设计问题引导用户思考
4. 用中文回复
5. 直接输出建议，不要有任何前缀、问候语或"我建议"等开场白

请直接输出建议内容：
'''

    # 5. 调用 LLM 生成反馈
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt]
        )
        feedback = response.text.strip()
        print(f"[AI Feedback] Generated for '{component_name}': {feedback[:50]}...")
        return feedback
    except Exception as e:
        print(f"[AI Feedback] Error generating feedback: {e}")
        # 异常时也直接给建议，不询问
        if component_memory.get("appearance") or component_memory.get("function"):
            return f"基于已记录的{component_name}设计信息，建议进一步明确细节需求。"
        return f"关于{component_name}的设计，可以从外形、功能或结构方面进一步深化。"