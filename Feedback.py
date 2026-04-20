"""
Feedback.py - AI反馈模块
包含：重复检测、反馈触发逻辑
"""

import sys
import os

# 禁用警告
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
import warnings
warnings.filterwarnings('ignore')

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


# ========== 评分权重配置 ==========

EVALUATION_WEIGHTS = {
    "Novelty": 1.0,
    "Value": 1.0,
    "Feasibility": 1.0,
    "Context-specific": 1.0
}


def generate_ai_feedback(component_name: str, vlm_output: dict, memory_db: dict = None) -> dict:
    """
    生成 AI 主动反馈内容（3个建议 + 评分）。

    根据部件记忆和当前 VLM 输出，针对用户意图生成3个建议并评分。

    Args:
        component_name: 部件名称
        vlm_output: VLM 分析结果（包含 User intent, Behavior description 等）
        memory_db: 记忆数据库（为 None 则从文件加载）

    Returns:
        dict: {
            "suggestions": [
                {
                    "content": "建议内容",
                    "scores": {"Novelty": 85, "Value": 90, "Feasibility": 75, "Context-specific": 88},
                    "total_score": 84.5
                },
                ...
            ]
        }
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
        "Appearance design": "用户正在关注产品的外观设计，包括外形、尺寸、材质、颜色、表面纹理等。",
        "Functional concept": "用户正在思考产品的功能目标和使用方式。",
        "Structural design": "用户正在考虑产品的结构关系、部件连接、布局等。",
        "Still-uncertain Idea Exploration": "用户正在进行探索性思考，想法还不够确定。",
        "Design Background supplement": "用户正在补充设计背景信息，如目标用户、使用场景等。"
    }

    intent_context = intent_prompts.get(user_intent, "用户正在思考设计相关的问题。")

    # 4. 构建生成建议的 prompt
    generate_prompt = f'''
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
请生成3个内容侧重不同的设计建议，每个建议不超过80字。
直接输出建议，不要有任何前缀、问候语或"我建议"等开场白。

输出格式（严格遵守JSON格式）：
{{
    "suggestions": [
        "建议1内容",
        "建议2内容",
        "建议3内容"
    ]
}}
'''

    # 5. 调用 LLM 生成3个建议
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[generate_prompt]
        )
        suggestions_response = extract_and_parse_json(response.text)
        if suggestions_response is None or "suggestions" not in suggestions_response:
            suggestions_response = {"suggestions": [
                f"基于{component_name}的外形特征，可以进一步优化细节设计。",
                f"考虑{component_name}的功能需求，建议提升实用性。",
                f"从结构角度，可以优化{component_name}的连接方式。"
            ]}
        suggestions = suggestions_response.get("suggestions", [])
        print(f"[AI Feedback] Generated {len(suggestions)} suggestions for '{component_name}'")
    except Exception as e:
        print(f"[AI Feedback] Error generating suggestions: {e}")
        suggestions = [
            f"基于{component_name}的外形特征，可以进一步优化细节设计。",
            f"考虑{component_name}的功能需求，建议提升实用性。",
            f"从结构角度，可以优化{component_name}的连接方式。"
        ]

    # 6. 对每个建议进行评分
    scored_suggestions = []
    for suggestion in suggestions:
        scores = evaluate_suggestion(
            component_name=component_name,
            user_intent=user_intent,
            behavior_description=behavior_description,
            user_speaking=user_speaking,
            ai_suggestion=suggestion
        )
        # 计算加权总分
        total_score = (
            scores["Novelty"] * EVALUATION_WEIGHTS["Novelty"] +
            scores["Value"] * EVALUATION_WEIGHTS["Value"] +
            scores["Feasibility"] * EVALUATION_WEIGHTS["Feasibility"] +
            scores["Context-specific"] * EVALUATION_WEIGHTS["Context-specific"]
        ) / sum(EVALUATION_WEIGHTS.values())

        scored_suggestions.append({
            "content": suggestion,
            "scores": scores,
            "total_score": round(total_score, 1)
        })

    # 7. 找出得分最高的建议
    best_suggestion = max(scored_suggestions, key=lambda x: x["total_score"])

    return best_suggestion


def evaluate_suggestion(
    component_name: str,
    user_intent: str,
    behavior_description: str,
    user_speaking: str,
    ai_suggestion: str
) -> dict:
    """
    对单个建议进行四维度评分。

    Args:
        component_name: 部件名称
        user_intent: 用户意图
        behavior_description: 行为描述
        user_speaking: 用户说的话
        ai_suggestion: AI建议内容

    Returns:
        dict: {"Novelty": 85, "Value": 90, "Feasibility": 75, "Context-specific": 88}
    """
    from record import client

    evaluate_prompt = f'''
你是一个专业的设计专家评估器，负责对"Intentrelay设计助手"给出的 AI 反馈进行打分。请根据以下四个维度进行百分制评价。

请结合以下内容进行评估：
当前设计对象：{component_name}
当前设计意图：{user_intent}
当前设计描述：{behavior_description}
当前用户表述：{user_speaking if user_speaking else "（用户没有说话）"}
AI建议：{ai_suggestion}

## 评分规则
请根据反馈质量在以下连续区间内给出具体分数：
[70 - 100分] 卓越：符合高分案例描述，提供了突破性、参数级或精准对位的建议。
[30 - 70分] 合格：符合中等案例描述，提供了常规补充或逻辑延伸建议。
[0 - 30分] 低效：符合低等案例描述，属于复读、空泛或不相关的反馈。

## 详细打分案例参考

### ① 新颖性 (Novelty)
AI 给出的内容是否包含当前尚未被用户明确提出、但对当前设计有意义的新信息或新方向。

[0 - 30分]：完全不新颖——AI 只是重复用户刚刚说过的话，或者转述当前状态，没有新增内容。
[30 - 70分]：中等新颖性——AI 在用户原有想法上做了一点补充。
[70 - 100分]：高新颖性——AI 提出了当前没有被提到、并且能明显拓展用户思路的新信息、新方向。

### ② 价值性 (Value)
反馈是否真的有助于当前设计任务，能改善功能、外形、结构，辅助探索。

[0 - 30分]：低价值——反馈过于空泛，对当前任务几乎没有帮助。
[30 - 70分]：中等价值——反馈能帮助用户意识到某个问题，或者提供一个有参考意义的方向，但推进作用有限。
[70 - 100分]：高价值——反馈抓住了当前设计中的关键问题或关键突破点，能够明显推动任务进展。

### ③ 可执行性 (Feasibility)
反馈是否能够转化成明确的下一步修改动作。

[0 - 30分]：低可执行性——反馈只是抽象评价，没有明确可操作的修改方向。
[30 - 70分]：中等可执行性——反馈指出了修改对象和大致方向，但具体怎么修改仍然不够明确。
[70 - 100分]：高可执行性——反馈已经能直接指导用户做出下一步修改。

### ④ 上下文相关性 (Context-specific)
给出的反馈是否贴合当前设计对象、阶段与意图。

[0 - 30分]：低相关性——反馈和当前行为明显不对应。
[30 - 70分]：中等相关性——反馈和产品整体方向有关，但没有紧贴当前的具体设计行为。
[70 - 100分]：高相关性——反馈紧密对应当前正在修改的对象、目标或刚发生的动作。

## 输出要求
请输出JSON格式，包含四个维度的分数（0-100的整数）：
{{
    "Novelty": 分数,
    "Value": 分数,
    "Feasibility": 分数,
    "Context-specific": 分数
}}
'''

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[evaluate_prompt]
        )
        scores = extract_and_parse_json(response.text)
        if scores is None:
            scores = {"Novelty": 50, "Value": 50, "Feasibility": 50, "Context-specific": 50}

        # 确保四个维度都有分数
        for key in ["Novelty", "Value", "Feasibility", "Context-specific"]:
            if key not in scores:
                scores[key] = 50
            scores[key] = max(0, min(100, int(scores[key])))

        return scores
    except Exception as e:
        print(f"[AI Feedback] Error evaluating suggestion: {e}")
        return {"Novelty": 50, "Value": 50, "Feasibility": 50, "Context-specific": 50}


# ========== 用户反馈处理函数 ==========

def process_user_feedback(user_feedback: str) -> dict:
    """
    处理用户反馈，分析对各维度的偏好，并更新评分权重。

    Args:
        user_feedback: 用户的反馈文本（如"我觉得可以更新颖一些"）

    Returns:
        dict: {
            "dimension_changes": {"Novelty": +0.1, "Feasibility": -0.1, ...},
            "updated_weights": {"Novelty": 1.1, "Value": 1.0, ...},
            "analysis": "LLM分析结果"
        }
    """
    global EVALUATION_WEIGHTS

    # 构建 LLM 分析 prompt
    analyze_prompt = f'''
你是一个反馈分析器，负责分析用户对AI设计建议的反馈，识别用户对不同评价维度的偏好变化。

## 四个评价维度
1. Novelty（新颖性）- 建议的创新程度、新想法
2. Value（价值性）- 建议的实用价值、对设计的帮助
3. Feasibility（可执行性）- 建议的可操作性、落地难度
4. Context-specific（上下文相关性）- 建议与当前设计场景的贴合度

## 用户反馈
"{user_feedback}"

## 分析规则
1. 判断用户反馈与哪些维度相关，不相关的维度标记为 0
2. 如果用户表达对某维度的偏好/希望加强，该维度标记为 +0.1
3. 如果用户表达对某维度的不在意/希望降低，该维度标记为 -0.1
4. 如果用户说降低某维度的同时暗示要提高另一维度，两者都需标记
5. 如果反馈与所有维度都不相关，全部标记为 0

## 常见表达示例
- "可以更新颖一些" → Novelty: +0.1，其他: 0
- "不需要搞那些花里胡哨的创新，用最稳妥、经典的方案就行。" → Novelty: -0.1，其他: 0
- "希望能更实用一些" → Value: +0.1，其他: 0
- "现在只是头脑风暴阶段，没必要考虑是否有用，纯发散即可。" → Value: -0.1，其他: 0
- "要能落地执行" → Feasibility: +0.1，其他: 0
- "不用太考虑可行性，给一些天马行空的想法" → Feasibility: -0.1, Novelty: +0.1
- "更贴合我的设计场景" → Context-specific: +0.1，其他: 0
- "跳出当前的业务框架，给我一些通用的、跨行业的设计灵感。" → Context-specific: -0.1，其他: 0
- "好的，谢谢" → 全部: 0（与维度无关）

## 输出要求
请输出JSON格式：
{{
    "Novelty": 变化值（-0.1, 0, 或 +0.1）,
    "Value": 变化值（-0.1, 0, 或 +0.1）,
    "Feasibility": 变化值（-0.1, 0, 或 +0.1）,
    "Context-specific": 变化值（-0.1, 0, 或 +0.1）,
    "analysis": "简要分析用户的偏好倾向"
}}
'''

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[analyze_prompt]
        )
        # 调试：打印 LLM 原始返回
        print(f"[User Feedback] LLM raw response: {response.text[:200] if response.text else 'Empty'}...")

        result = extract_and_parse_json(response.text)

        if result is None:
            print("[User Feedback] Failed to parse LLM response")
            return {
                "dimension_changes": {},
                "updated_weights": EVALUATION_WEIGHTS.copy(),
                "analysis": "解析失败"
            }

        # 提取变化值
        dimension_changes = {}
        for key in ["Novelty", "Value", "Feasibility", "Context-specific"]:
            change = result.get(key, 0)
            # 确保变化值在合理范围内
            if isinstance(change, (int, float)):
                dimension_changes[key] = round(float(change), 1)
            else:
                dimension_changes[key] = 0.0

        # 更新权重
        for key, change in dimension_changes.items():
            EVALUATION_WEIGHTS[key] = round(EVALUATION_WEIGHTS[key] + change, 2)
            # 权重下限为 0.1，避免归零
            if EVALUATION_WEIGHTS[key] < 0.1:
                EVALUATION_WEIGHTS[key] = 0.1

        print(f"[User Feedback] Weight changes: {dimension_changes}")
        print(f"[User Feedback] Updated weights: {EVALUATION_WEIGHTS}")

        return {
            "dimension_changes": dimension_changes,
            "updated_weights": EVALUATION_WEIGHTS.copy(),
            "analysis": result.get("analysis", "")
        }

    except Exception as e:
        print(f"[User Feedback] Error processing feedback: {e}")
        return {
            "dimension_changes": {},
            "updated_weights": EVALUATION_WEIGHTS.copy(),
            "analysis": f"处理出错: {str(e)}"
        }


def get_current_weights() -> dict:
    """获取当前的评分权重"""
    return EVALUATION_WEIGHTS.copy()


def reset_weights():
    """重置评分权重为默认值"""
    global EVALUATION_WEIGHTS
    EVALUATION_WEIGHTS = {
        "Novelty": 1.0,
        "Value": 1.0,
        "Feasibility": 1.0,
        "Context-specific": 1.0
    }
    print("[User Feedback] Weights reset to default")


# ========== 记忆问答交互 ==========

# 全局状态：当前问题列表和正在回答的问题索引
_question_list = []
_current_question_index = 0


def memory_qa_round(memory_db: dict, user_answer: str = None) -> dict:
    """
    多轮问答：AI生成最多3个问题，逐个询问用户。

    用户只需要输入回答内容，不需要输入部件名和描述类型！
    AI 会记住当前正在问哪个问题，自动匹配用户的回答。

    询问两类问题：
    1. 完全没有描述但重要：某个部件的某个维度是空的
    2. 描述冲突：同一部件的不同描述矛盾

    不询问 status=0 的内容（用户正在探索的）。

    Args:
        memory_db: 记忆数据库
        user_answer: 用户对当前问题的回答（纯文本，如 "科技风格，金属材质")
            如果为 None，表示开始新一轮问答，AI会生成问题列表

    Returns:
        dict: {
            "has_question": true/false,
            "questions": [
                {
                    "target": "履带",
                    "desc_type": "外形",
                    "issue_type": "缺失",
                    "question": "履带的外形风格您希望是什么？"
                },
                ...最多3个
            ],
            "current_index": 0,  // 当前正在问第几个问题（前端可用 questions[current_index] 获取当前问题）
            "update_result": "已更新" 或 null,
            "remaining_count": 2  // 剩余多少问题没问
        }
    """
    global _question_list, _current_question_index
    print("\n[Memory QA] 开始问答轮次...")

    # 1. 如果有用户回答，先更新记忆
    update_result = None
    if user_answer and user_answer.strip() and _question_list and _current_question_index < len(_question_list):
        current_question = _question_list[_current_question_index]
        target = current_question.get("target")
        desc_type = current_question.get("desc_type")

        # 特殊处理：SKIP 表示跳过当前问题
        if user_answer.strip() == "SKIP":
            print(f"[Memory QA] 用户跳过问题[{_current_question_index}]: target='{target}', desc_type='{desc_type}'")
            update_result = f"已跳过 {target} 的 {desc_type} 问题"
            _current_question_index += 1
            print(f"[Memory QA] 移到下一个问题，当前索引: {_current_question_index}")
        else:
            print(f"[Memory QA] 用户回答: '{user_answer}'")
            print(f"[Memory QA] 当前问题[{_current_question_index}]: target='{target}', desc_type='{desc_type}'")

            # 导入更新函数
            from Memory import add_description_from_answer
            success, message = add_description_from_answer(
                memory_db=memory_db,
                target_name=target,
                desc_type=desc_type,
                answer=user_answer.strip()
            )
            update_result = message if success else f"更新失败: {message}"
            print(f"[Memory QA] 更新记忆: {message}")

            # 移到下一个问题
            _current_question_index += 1
            print(f"[Memory QA] 移到下一个问题，当前索引: {_current_question_index}")

    elif user_answer and user_answer.strip() and (not _question_list or _current_question_index >= len(_question_list)):
        print("[Memory QA] 收到回答但没有待回答的问题")
        update_result = "没有待回答的问题，请先开始新一轮问答"
        # 重置状态
        _question_list = []
        _current_question_index = 0

    # 2. 判断当前状态
    # 如果问题列表不为空，且索引在范围内，返回当前问题
    if _question_list and _current_question_index < len(_question_list):
        remaining_count = len(_question_list) - _current_question_index
        current_question = _question_list[_current_question_index]

        print(f"[Memory QA] 当前问题[{_current_question_index}]: {current_question}")
        print(f"[Memory QA] 剩余 {remaining_count} 个问题")

        return {
            "has_questions": True,
            "questions": _question_list,
            "current_index": _current_question_index,
            "update_result": update_result,
            "remaining_count": remaining_count
        }

    # 3. 如果问题列表不为空，但索引超出了，说明这轮问答结束
    # 不自动生成新问题，等待用户主动开始新一轮
    if _question_list and _current_question_index >= len(_question_list):
        print("[Memory QA] 这轮问答结束，清空问题列表")
        _question_list = []
        _current_question_index = 0

        return {
            "has_questions": False,
            "questions": [],
            "current_index": 0,
            "update_result": update_result,
            "remaining_count": 0
        }

    # 4. 如果问题列表为空，且没有用户回答（新一轮开始），生成问题列表
    if not _question_list and not user_answer:
        # 提取当前记忆状态
        components_data = {}
        overall_data = {}

        for node_id, data in memory_db.items():
            if data.get('node_type') == 'COMPONENT':
                component_name = data.get('component_name', '未知部件')

                # 确定的描述 (status=1)
                appearance_confirmed = [d.get('content', '') for d in data.get('appearance_descriptions', []) if d.get('status', 1) == 1 and d.get('content', '')]
                function_confirmed = [d.get('content', '') for d in data.get('function_descriptions', []) if d.get('status', 1) == 1 and d.get('content', '')]
                structure_confirmed = [d.get('content', '') for d in data.get('structure_descriptions', []) if d.get('status', 1) == 1 and d.get('content', '')]

                # 探索中的描述 (status=0) - 不询问
                appearance_exploring = [d.get('content', '') for d in data.get('appearance_descriptions', []) if d.get('status', 1) == 0 and d.get('content', '')]
                function_exploring = [d.get('content', '') for d in data.get('function_descriptions', []) if d.get('status', 1) == 0 and d.get('content', '')]
                structure_exploring = [d.get('content', '') for d in data.get('structure_descriptions', []) if d.get('status', 1) == 0 and d.get('content', '')]

                components_data[component_name] = {
                    "appearance_confirmed": appearance_confirmed,
                    "function_confirmed": function_confirmed,
                    "structure_confirmed": structure_confirmed,
                    "appearance_exploring": appearance_exploring,
                    "function_exploring": function_exploring,
                    "structure_exploring": structure_exploring
                }

            elif data.get('node_type') == 'OVERALL':
                design_background = data.get('design_background', '')

                overall_appearances_confirmed = [d.get('content', '') for d in data.get('overall_appearances', []) if d.get('status', 1) == 1 and d.get('content', '')]
                overall_functions_confirmed = [d.get('content', '') for d in data.get('overall_functions', []) if d.get('status', 1) == 1 and d.get('content', '')]
                overall_structures_confirmed = [d.get('content', '') for d in data.get('overall_structures', []) if d.get('status', 1) == 1 and d.get('content', '')]

                overall_data = {
                    "design_background": design_background,
                    "appearance_confirmed": overall_appearances_confirmed,
                    "function_confirmed": overall_functions_confirmed,
                    "structure_confirmed": overall_structures_confirmed
                }

        # 构建记忆摘要
        memory_summary = "当前记忆内容：\n"

        if overall_data:
            memory_summary += "\n【整体】\n"
            memory_summary += f"  设计背景：{overall_data['design_background'] if overall_data['design_background'] else '暂无'}\n"
            memory_summary += f"  外形（已确定）：{', '.join(overall_data['appearance_confirmed']) if overall_data['appearance_confirmed'] else '暂无'}\n"
            memory_summary += f"  功能（已确定）：{', '.join(overall_data['function_confirmed']) if overall_data['function_confirmed'] else '暂无'}\n"
            memory_summary += f"  结构（已确定）：{', '.join(overall_data['structure_confirmed']) if overall_data['structure_confirmed'] else '暂无'}\n"

        if components_data:
            memory_summary += "\n【部件】\n"
            for name, info in components_data.items():
                memory_summary += f"  {name}：\n"
                memory_summary += f"    外形（已确定）：{', '.join(info['appearance_confirmed']) if info['appearance_confirmed'] else '暂无'}\n"
                memory_summary += f"    功能（已确定）：{', '.join(info['function_confirmed']) if info['function_confirmed'] else '暂无'}\n"
                memory_summary += f"    结构（已确定）：{', '.join(info['structure_confirmed']) if info['structure_confirmed'] else '暂无'}\n"

        print(f"[Memory QA] 提取到 {len(components_data)} 个部件")

        # 使用 LLM 分析并生成问题列表（最多3个）
        analyze_prompt = f'''
你是一个产品设计助手。请分析当前记忆内容，找出需要询问用户的问题。

## 当前记忆内容
{memory_summary}

## 你需要找的问题类型（只找这两类）

### 类型1：完全没有描述但重要
某个部件或整体的某个维度**完全没有任何已确定的描述**（列表是空的），但这对设计很重要。

例如：
- 履带的外形（已确定）是"暂无" → 需要问：履带的外形风格您希望是什么？
- 整体的设计背景是"暂无" → 需要问：这个产品的目标用户和使用场景是什么？

### 类型2：描述冲突
同一部件的不同已确定描述之间可能存在矛盾。

例如：
- 履带外形说"轻便小巧"，但结构说"笨重的大型框架" → 冲突！需要问用户确认

## 不要询问的内容

- **正在探索中的内容（status=0）**：这些是用户正在思考的，不要打断！
- 已有确定描述的维度：不需要重复询问

## 你的任务

1. 找出所有符合上述两类问题的情况
2. 按重要性排序（设计背景 > 部件外形 > 部件功能 > 部件结构 > 冲突）
3. 返回**最多3个**最重要的问题
4. 如果没有需要询问的问题，返回 has_questions=false

## 输出格式（严格遵守JSON格式）
 {{
    "has_questions": true或false,
    "questions": [
        {{
            "target": "部件名或整体",
            "desc_type": "外形/功能/结构/背景",
            "issue_type": "缺失/冲突",
            "issue": "简述缺失什么或有什么冲突",
            "question": "向用户提出的问题（简洁明确，一句话）"
        }},
        ...最多3个
    ],
    "total_count": 总问题数量（0-3的整数）
}}

如果 has_questions=false，questions 字段为空列表。
'''

        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[analyze_prompt]
            )
            result = extract_and_parse_json(response.text)

            if result is None:
                print("[Memory QA] LLM 解析失败")
                return {
                    "has_questions": False,
                    "questions": [],
                    "current_index": 0,
                    "update_result": update_result,
                    "remaining_count": 0
                }

            # 更新全局状态
            _question_list = result.get("questions", [])
            _current_question_index = 0
            total_count = len(_question_list)

            print(f"[Memory QA] 生成了 {total_count} 个问题")
            for i, q in enumerate(_question_list):
                print(f"  [{i}] {q.get('target')} - {q.get('desc_type')}: {q.get('question')}")

            return {
                "has_questions": result.get("has_questions", False) and total_count > 0,
                "questions": _question_list,
                "current_index": _current_question_index,
                "update_result": update_result,
                "remaining_count": total_count
            }

        except Exception as e:
            print(f"[Memory QA] 异常: {e}")
            return {
                "has_questions": False,
                "questions": [],
                "current_index": 0,
                "update_result": update_result,
                "remaining_count": 0,
                "error": str(e)
            }

    # 3. 如果还有未问完的问题，返回当前问题
    else:
        remaining_count = len(_question_list) - _current_question_index

        print(f"[Memory QA] 当前问题[{_current_question_index}]")
        print(f"[Memory QA] 剩余 {remaining_count} 个问题")

        return {
            "has_questions": remaining_count > 0,
            "questions": _question_list,
            "current_index": _current_question_index,
            "update_result": update_result,
            "remaining_count": remaining_count
        }


# ========== 旧版批量分析（保留备用） ==========

def analyze_memory_and_generate_questions(memory_db: dict) -> dict:
    """
    整理记忆中所有文字信息，找出缺失或冲突的地方，生成问题询问用户。

    注意：status=0 的内容是用户正在探索的，不需要询问。
    此函数只关注重要信息缺失或描述冲突的情况。

    Args:
        memory_db: 记忆数据库

    Returns:
        dict: {
            "has_issues": true/false,
            "questions": [
                {
                    "id": 1,
                    "target": "履带",
                    "type": "缺失/冲突",
                    "issue_description": "外形描述缺失",
                    "question": "履带的外形风格您希望是什么？"
                },
                ...
            ],
            "memory_summary": "当前有2个部件..."
        }
    """
    print("\n[Memory Analysis] 开始整理记忆...")

    # 1. 提取所有部件信息（只提取 status=1 的确定内容）
    components_data = {}
    overall_data = {}

    for node_id, data in memory_db.items():
        if data.get('node_type') == 'COMPONENT':
            component_name = data.get('component_name', '未知部件')

            # 只提取确定的描述 (status=1)
            appearance_list = [d.get('content', '') for d in data.get('appearance_descriptions', []) if d.get('status', 1) == 1 and d.get('content', '')]
            function_list = [d.get('content', '') for d in data.get('function_descriptions', []) if d.get('status', 1) == 1 and d.get('content', '')]
            structure_list = [d.get('content', '') for d in data.get('structure_descriptions', []) if d.get('status', 1) == 1 and d.get('content', '')]

            components_data[component_name] = {
                "appearance": appearance_list,
                "function": function_list,
                "structure": structure_list
            }

        elif data.get('node_type') == 'OVERALL':
            design_background = data.get('design_background', '')

            # 只提取确定的描述 (status=1)
            overall_appearances = [d.get('content', '') for d in data.get('overall_appearances', []) if d.get('status', 1) == 1 and d.get('content', '')]
            overall_functions = [d.get('content', '') for d in data.get('overall_functions', []) if d.get('status', 1) == 1 and d.get('content', '')]
            overall_structures = [d.get('content', '') for d in data.get('overall_structures', []) if d.get('status', 1) == 1 and d.get('content', '')]

            overall_data = {
                "design_background": design_background,
                "appearance": overall_appearances,
                "function": overall_functions,
                "structure": overall_structures
            }

    # 2. 构建记忆摘要
    memory_summary = "当前记忆内容（已确定的描述）：\n"

    if overall_data:
        memory_summary += "\n整体信息：\n"
        if overall_data.get('design_background'):
            memory_summary += f"  设计背景：{overall_data['design_background']}\n"
        if overall_data.get('appearance'):
            memory_summary += f"  外形：{', '.join(overall_data['appearance'])}\n"
        if overall_data.get('function'):
            memory_summary += f"  功能：{', '.join(overall_data['function'])}\n"
        if overall_data.get('structure'):
            memory_summary += f"  结构：{', '.join(overall_data['structure'])}\n"

    if components_data:
        memory_summary += "\n部件信息：\n"
        for name, info in components_data.items():
            memory_summary += f"  {name}：\n"
            memory_summary += f"    外形：{', '.join(info['appearance']) if info['appearance'] else '暂无'}\n"
            memory_summary += f"    功能：{', '.join(info['function']) if info['function'] else '暂无'}\n"
            memory_summary += f"    结构：{', '.join(info['structure']) if info['structure'] else '暂无'}\n"

    print(f"[Memory Analysis] 提取到 {len(components_data)} 个部件")
    print(f"[Memory Analysis] 记忆摘要已生成")

    # 3. 使用 LLM 分析缺失/冲突并生成问题
    analyze_prompt = f'''
你是一个产品设计助手，正在协助整理设计记忆。请分析以下记忆内容，找出需要用户补充的重要信息。

## 当前记忆内容（用户已确定的描述）
{memory_summary}

## 你的任务
分析每个部件和整体，找出以下问题：

1. **重要信息缺失**：某个部件或整体缺少关键维度的描述
   - 外形缺失：没有描述外形风格、材质、颜色等
   - 功能缺失：没有描述主要功能、用途
   - 结构缺失：没有描述结构关系、连接方式
   - 设计背景缺失：整体没有设计背景（目标用户、使用场景）

2. **描述冲突**：同一部件的不同描述可能存在矛盾
   - 例如：外形说"轻便小巧"，但结构说"笨重的大型框架"

3. 按重要性排序，生成不超过3条问题

## 注意
- status=0 的内容是用户正在探索的，不要作为问题来源
- 只关注用户还没提到的、但对设计很重要的信息
- 问题应该具体、简洁，用户可以直接回答

## 输出格式（严格遵守JSON格式）
 {{
    "has_issues": true或false,
    "questions": [
        {{
            "id": 1,
            "target": "部件名或整体",
            "type": "缺失/冲突",
            "issue_description": "具体描述缺失什么或有什么冲突",
            "question": "向用户提出的问题（简洁明确）"
        }},
        ...
    ],
    "analysis_summary": "简要分析结果"
}}

注意：
- 如果记忆内容完整且无冲突，has_issues 设为 false，questions 为空列表
- 问题数量不超过3条
'''

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[analyze_prompt]
        )
        result = extract_and_parse_json(response.text)

        if result is None:
            print("[Memory Analysis] LLM 解析失败")
            return {
                "has_issues": False,
                "questions": [],
                "memory_summary": memory_summary
            }

        print(f"[Memory Analysis] LLM 分析完成，has_issues: {result.get('has_issues')}")
        print(f"[Memory Analysis] 生成 {len(result.get('questions', []))} 条问题")

        return {
            "has_issues": result.get("has_issues", False),
            "questions": result.get("questions", []),
            "memory_summary": memory_summary,
            "analysis_summary": result.get("analysis_summary", "")
        }

    except Exception as e:
        print(f"[Memory Analysis] 异常: {e}")
        return {
            "has_issues": False,
            "questions": [],
            "memory_summary": memory_summary,
            "error": str(e)
        }


# ========== 不确定内容建议生成 ==========

def get_uncertain_suggestions(memory_db: dict) -> dict:
    """
    提取所有 status=0 的不确定内容，为每个生成简短建议。

    Args:
        memory_db: 记忆数据库

    Returns:
        dict: {
            "has_uncertain": true/false,
            "uncertain_items": [
                {
                    "id": 1,
                    "target": "履带",
                    "type": "外形",
                    "content": "还没想好外形风格",
                    "suggestion": "可考虑科技风格，与整体功能定位匹配"
                },
                ...
            ],
            "count": 3
        }
    """
    print("\n[Uncertain Suggestions] 开始提取不确定内容...")

    # 1. 提取所有 status=0 的内容
    uncertain_items = []

    for node_id, data in memory_db.items():
        if data.get('node_type') == 'COMPONENT':
            component_name = data.get('component_name', '未知部件')

            # 外形描述
            for desc in data.get('appearance_descriptions', []):
                if desc.get('status', 1) == 0 and desc.get('content', '').strip():
                    uncertain_items.append({
                        "target": component_name,
                        "type": "外形",
                        "content": desc.get('content', '')
                    })

            # 功能描述
            for desc in data.get('function_descriptions', []):
                if desc.get('status', 1) == 0 and desc.get('content', '').strip():
                    uncertain_items.append({
                        "target": component_name,
                        "type": "功能",
                        "content": desc.get('content', '')
                    })

            # 结构描述
            for desc in data.get('structure_descriptions', []):
                if desc.get('status', 1) == 0 and desc.get('content', '').strip():
                    uncertain_items.append({
                        "target": component_name,
                        "type": "结构",
                        "content": desc.get('content', '')
                    })

        elif data.get('node_type') == 'OVERALL':
            # 整体外形
            for desc in data.get('overall_appearances', []):
                if desc.get('status', 1) == 0 and desc.get('content', '').strip():
                    uncertain_items.append({
                        "target": "整体",
                        "type": "外形",
                        "content": desc.get('content', '')
                    })

            # 整体功能
            for desc in data.get('overall_functions', []):
                if desc.get('status', 1) == 0 and desc.get('content', '').strip():
                    uncertain_items.append({
                        "target": "整体",
                        "type": "功能",
                        "content": desc.get('content', '')
                    })

            # 整体结构
            for desc in data.get('overall_structures', []):
                if desc.get('status', 1) == 0 and desc.get('content', '').strip():
                    uncertain_items.append({
                        "target": "整体",
                        "type": "结构",
                        "content": desc.get('content', '')
                    })

    print(f"[Uncertain Suggestions] 提取到 {len(uncertain_items)} 条不确定内容")

    if not uncertain_items:
        return {
            "has_uncertain": False,
            "uncertain_items": [],
            "count": 0
        }

    # 2. 提取已确定的上下文信息（用于生成建议参考）
    context_info = []
    for node_id, data in memory_db.items():
        if data.get('node_type') == 'COMPONENT':
            name = data.get('component_name', '未知')
            # 已确定的描述
            for desc in data.get('appearance_descriptions', []):
                if desc.get('status', 1) == 1 and desc.get('content', '').strip():
                    context_info.append(f"[{name} 外形] {desc.get('content', '')}")
            for desc in data.get('function_descriptions', []):
                if desc.get('status', 1) == 1 and desc.get('content', '').strip():
                    context_info.append(f"[{name} 功能] {desc.get('content', '')}")
            for desc in data.get('structure_descriptions', []):
                if desc.get('status', 1) == 1 and desc.get('content', '').strip():
                    context_info.append(f"[{name} 结构] {desc.get('content', '')}")

        elif data.get('node_type') == 'OVERALL':
            if data.get('design_background'):
                context_info.append(f"[整体背景] {data.get('design_background')}")

    context_text = "\n".join(context_info) if context_info else "暂无其他参考信息"

    # 3. 使用 LLM 为每条不确定内容生成简短建议
    items_text = "\n".join([
        f"{i+1}. [{item['target']} {item['type']}] {item['content']}"
        for i, item in enumerate(uncertain_items)
    ])

    suggest_prompt = f'''
你是一个产品设计助手。用户正在进行设计探索，有一些内容尚未确定。
请根据已有的上下文信息，为每条不确定内容提供简短的建议方向。

## 已确定的上下文信息
{context_text}

## 用户不确定的内容
{items_text}

## 任务
为每条不确定内容生成一条简短建议（不超过30字）：
- 建议应该具体、有启发性
- 结合已有上下文信息
- 不要给出最终答案，而是提供思考方向

## 输出格式（严格遵守JSON格式）
 {{
    "suggestions": [
        {{
            "id": 1,
            "target": "部件名",
            "type": "外形/功能/结构",
            "content": "不确定的内容",
            "suggestion": "简短建议（不超过30字）"
        }},
        ...
    ]
}}
'''

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[suggest_prompt]
        )
        result = extract_and_parse_json(response.text)

        if result is None:
            print("[Uncertain Suggestions] LLM 解析失败，使用默认建议")
            # 使用默认建议
            for i, item in enumerate(uncertain_items):
                item['id'] = i + 1
                item['suggestion'] = f"可以参考同类产品的{item['type']}设计"
            return {
                "has_uncertain": True,
                "uncertain_items": uncertain_items,
                "count": len(uncertain_items)
            }

        # 合并 LLM 结果
        suggestions = result.get('suggestions', [])
        for i, item in enumerate(uncertain_items):
            item['id'] = i + 1
            # 匹配 LLM 返回的建议
            for sug in suggestions:
                if sug.get('target') == item['target'] and sug.get('type') == item['type']:
                    item['suggestion'] = sug.get('suggestion', '可以进一步思考')
                    break
            if 'suggestion' not in item:
                item['suggestion'] = f"可以参考同类产品的{item['type']}设计"

        print(f"[Uncertain Suggestions] 生成了 {len(uncertain_items)} 条建议")

        return {
            "has_uncertain": True,
            "uncertain_items": uncertain_items,
            "count": len(uncertain_items)
        }

    except Exception as e:
        print(f"[Uncertain Suggestions] 异常: {e}")
        # 使用默认建议
        for i, item in enumerate(uncertain_items):
            item['id'] = i + 1
            item['suggestion'] = f"可以参考同类产品的{item['type']}设计"
        return {
            "has_uncertain": True,
            "uncertain_items": uncertain_items,
            "count": len(uncertain_items),
            "error": str(e)
        }