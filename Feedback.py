"""
Feedback.py - AI反馈模块
包含：重复检测、反馈触发逻辑
"""

import json
import numpy as np
from typing import Dict, Any, Tuple, Optional

# 从 record.py 导入需要的函数
from record import text_encoder, extract_and_parse_json


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


# ========== AI 反馈生成函数（待实现）==========

def generate_ai_feedback(component_name: str, vlm_output: dict) -> str:
    """
    生成 AI 主动反馈内容。

    TODO: 根据部件名和 VLM 输出生成针对性的反馈

    Args:
        component_name: 部件名称
        vlm_output: VLM 分析结果

    Returns:
        AI 反馈文本
    """
    # TODO: 实现反馈生成逻辑
    feedback = f"[AI Feedback] 我注意到你一直在关注 '{component_name}'，需要我提供一些建议吗？"
    return feedback