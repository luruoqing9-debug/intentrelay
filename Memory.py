"""
Memory.py - 记忆节点与管理模块
包含：节点类定义、记忆管理逻辑
依赖：record.py（编码、VLM、存储函数）
"""

import sys
import os

# 禁用警告
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
import warnings
warnings.filterwarnings('ignore')

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel, Field
from PIL import Image
import numpy as np

# 从 record.py 导入需要的函数
from record import (
    image_encoder, text_encoder,
    get_crop, encode_image_to_base64,
    vlm_chat_mock, vlm_chat_virtual,
    load_memory_from_json, save_memory_to_json,
    extract_and_parse_json,
    llm_merge_names, llm_merge_descriptions,
    llm_analyze_design_info,
    PhysicalTriggerType
)


# ========== 数据类型定义 ==========

class BoundingBox(BaseModel):
    """2D空间中的边界框"""
    xmin: int
    ymin: int
    xmax: int
    ymax: int


NodeType = Literal["OBJECT", "COMPONENT", "OVERALL"]


# ========== 基础节点类 ==========

class BaseMemoryNode(BaseModel):
    """所有记忆节点的抽象基类"""
    node_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="全局唯一标识符(UUID)"
    )
    node_type: NodeType = Field(..., description="节点类型")
    timestamp_created: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="创建时间(UTC)"
    )
    timestamp_last_accessed: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="最后访问时间(UTC)"
    )
    source_snapshot_ids: List[str] = Field(
        default_factory=list,
        description="提供证据的事件节点ID列表"
    )

    def touch(self) -> None:
        """更新最后访问时间"""
        self.timestamp_last_accessed = datetime.now(timezone.utc)


class DescriptionWithStatus(BaseModel):
    """带状态的描述结构"""
    content: str = Field(..., description="描述内容")
    status: int = Field(default=1, ge=0, le=1, description="状态：1=确定，0=不确定")


# ========== 具体节点类 ==========

class ObjectNode(BaseMemoryNode):
    """物理实体节点"""
    node_type: NodeType = "OBJECT"
    label: str = Field(..., description="视觉模型识别的通用标签")
    custom_names: List[str] = Field(default_factory=list, description="用户自定义名称")
    description: Optional[str] = Field(None, description="自然语言描述")
    image_embedding: Optional[List[float]] = Field(None, description="图像向量嵌入")
    text_embeddings: Optional[List[List[float]]] = Field(None, description="文本向量嵌入列表")
    last_known_bbox: Optional[BoundingBox] = Field(None, description="最后已知边界框")
    last_seen_image: Optional[str] = Field(None, description="最后看到的图像(Base64)")


# 用户意图类型
UserIntentType = Literal[
    "Appearance design",
    "Functional concept",
    "Structural design",
    "Still-uncertain Idea Exploration",
    "Design Background supplement"
]


class ComponentNode(BaseMemoryNode):
    """部件记忆节点"""
    node_type: NodeType = "COMPONENT"
    component_name: str = Field(..., description="部件名称")
    appearance_descriptions: List[DescriptionWithStatus] = Field(default_factory=list, description="外形描述")
    structure_descriptions: List[DescriptionWithStatus] = Field(default_factory=list, description="结构描述")
    function_descriptions: List[DescriptionWithStatus] = Field(default_factory=list, description="功能描述")
    component_image: Optional[str] = Field(None, description="部件图片(Base64)")
    image_embedding: Optional[List[float]] = Field(None, description="图像向量嵌入")
    appearance_embeddings: List[List[float]] = Field(default_factory=list, description="外形描述嵌入")
    structure_embeddings: List[List[float]] = Field(default_factory=list, description="结构描述嵌入")
    function_embeddings: List[List[float]] = Field(default_factory=list, description="功能描述嵌入")


class OverallNode(BaseMemoryNode):
    """整体记忆节点"""
    node_type: NodeType = "OVERALL"
    design_background: Optional[str] = Field(None, description="设计背景")
    overall_appearances: List[DescriptionWithStatus] = Field(default_factory=list, description="整体外形描述")
    overall_structures: List[DescriptionWithStatus] = Field(default_factory=list, description="整体结构描述")
    overall_functions: List[DescriptionWithStatus] = Field(default_factory=list, description="整体功能描述")
    overall_image: Optional[str] = Field(None, description="整体图片(Base64)")
    image_embedding: Optional[List[float]] = Field(None, description="图像向量嵌入")
    overall_appearance_embeddings: List[List[float]] = Field(default_factory=list, description="外形描述嵌入")
    overall_structure_embeddings: List[List[float]] = Field(default_factory=list, description="结构描述嵌入")
    overall_function_embeddings: List[List[float]] = Field(default_factory=list, description="功能描述嵌入")
    component_ids: List[str] = Field(default_factory=list, description="关联部件ID列表")


# ========== 描述处理函数 ==========

DESCRIPTION_SIMILARITY_THRESHOLD = 0.85


def make_description(content: str, status: int = 1) -> DescriptionWithStatus:
    """创建带状态的描述对象"""
    return DescriptionWithStatus(content=content, status=status)


def find_or_update_description(
    description_list: List[DescriptionWithStatus],
    new_content: str,
    new_status: int = 1,
    text_embeddings_list: List[List[float]] = None,
    similarity_threshold: float = DESCRIPTION_SIMILARITY_THRESHOLD
) -> Tuple[List[DescriptionWithStatus], List[List[float]], bool]:
    """
    在描述列表中检索相似描述，找到则合并，否则追加。

    Returns:
        (更新后的描述列表, 更新后的嵌入列表, 是否找到匹配)
    """
    if not new_content:
        return description_list, text_embeddings_list or [], False

    if text_embeddings_list is None:
        text_embeddings_list = []

    if len(text_embeddings_list) != len(description_list):
        print(f"[Warning] Embeddings mismatch, recalculating...")
        text_embeddings_list = [text_encoder(d.content).tolist() for d in description_list]

    new_embedding = text_encoder(new_content)

    if not description_list:
        new_desc = make_description(new_content, new_status)
        return [new_desc], [new_embedding.tolist()], False

    embeddings = np.array(text_embeddings_list)
    if len(embeddings) > 0:
        similarities = np.dot(embeddings, new_embedding)
        max_idx = np.argmax(similarities)
        max_similarity = similarities[max_idx]

        print(f"[Desc Search] Max similarity: {max_similarity:.4f}")

        if max_similarity > similarity_threshold:
            print(f"[Desc Match] Merging content...")
            original_content = description_list[max_idx].content
            merged_content = llm_merge_descriptions(original_content, new_content)
            description_list[max_idx].content = merged_content
            description_list[max_idx].status = new_status
            merged_embedding = text_encoder(merged_content)
            text_embeddings_list[max_idx] = merged_embedding.tolist()
            return description_list, text_embeddings_list, True

    print(f"[Desc Append] No match, appending...")
    new_desc = make_description(new_content, new_status)
    updated_embeddings = text_embeddings_list + [new_embedding.tolist()]
    return description_list + [new_desc], updated_embeddings, False


# ========== 记忆管理函数 ==========

def find_or_create_obj(
    frame: Image.Image,
    bbox: BoundingBox,
    trigger_type: PhysicalTriggerType,
    transcript_text: str,
    memory_db: Dict[str, Any],
    similarity_threshold: float = 0.90
) -> ObjectNode:
    """
    查找或创建物体节点。

    Args:
        frame: 视频帧图像
        bbox: 关注区域的边界框（来自眼动追踪或手部检测）
        trigger_type: 物理世界触发类型（眼动注视/手部交互/语音触发）
        transcript_text: 语音转文本结果
        memory_db: 记忆数据库
        similarity_threshold: 相似度阈值

    Returns:
        ObjectNode: 找到或创建的物体节点
    """
    print(f"\n--- Processing: trigger='{trigger_type}', transcript='{transcript_text}' ---")

    crop = get_crop(frame, bbox)
    new_embedding = image_encoder(crop)

    if memory_db:
        object_nodes = {
            node_id: data for node_id, data in memory_db.items()
            if data.get('node_type') == 'OBJECT' and data.get('image_embedding') is not None
        }

        if object_nodes:
            existing_ids = list(object_nodes.keys())
            existing_embeddings = np.array([data['image_embedding'] for data in object_nodes.values()])

            similarities = np.dot(existing_embeddings, new_embedding)
            most_similar_idx = np.argmax(similarities)
            max_similarity = similarities[most_similar_idx]

            print(f"Highest similarity: {max_similarity:.4f}")

            if max_similarity > similarity_threshold:
                print(f"MATCH FOUND! Updating existing object.")
                matched_node_id = existing_ids[most_similar_idx]
                node_data = object_nodes[matched_node_id]
                node = ObjectNode(**node_data)
                node.last_known_bbox = bbox
                node.last_seen_image = encode_image_to_base64(crop)
                node.touch()
                return node

    print("NO MATCH. Creating new object.")
    ret_json = extract_and_parse_json(vlm_chat_mock(encode_image_to_base64(frame), trigger_type, transcript_text))

    if ret_json is None:
        print("[Warning] VLM returned invalid JSON")
        ret_json = {'type': 'component', 'label': 'unknown', 'User Speaking': '', 'Behavior description': ''}

    # 新格式提取
    label = ret_json.get('label', 'unknown')
    user_speaking = ret_json.get('User Speaking', '')
    behavior_desc = ret_json.get('Behavior description', '')

    # 将 Behavior description 作为描述
    description = behavior_desc

    chunks = []
    if label:
        chunks.append(text_encoder(label).tolist())
    if description:
        chunks.append(text_encoder(description).tolist())

    return ObjectNode(
        label=label,
        custom_names=[label],  # 使用 label 作为名称
        description=description,
        image_embedding=new_embedding.tolist(),
        text_embeddings=chunks,
        last_known_bbox=bbox,
        last_seen_image=encode_image_to_base64(crop)
    )


def create_component_memory(
    component_name: str,
    appearance_items: List[dict] = None,
    structure_items: List[dict] = None,
    function_items: List[dict] = None,
    component_image: Image.Image = None,
    memory_db: Dict[str, Any] = None
) -> ComponentNode:
    """
    创建或更新部件记忆节点。

    Args:
        component_name: 部件名称
        appearance_items: [{"description": "描述", "status": 1}, ...]
        structure_items: [{"description": "描述", "status": 1}, ...]
        function_items: [{"description": "描述", "status": 1}, ...]
        component_image: 部件图片
        memory_db: 记忆数据库
    """
    print(f"\n--- Component Memory: '{component_name}' ---")

    if memory_db:
        for node_id, data in memory_db.items():
            if data.get('node_type') == 'COMPONENT':
                existing_name = data.get('component_name', '')
                merged_name, is_same = llm_merge_names(existing_name, component_name)
                if is_same:
                    print(f"Found existing: '{existing_name}' = '{component_name}'")
                    node = ComponentNode(**data)
                    node.component_name = merged_name

                    if appearance_items:
                        for item in appearance_items:
                            desc = item.get("description", "")
                            status = item.get("status", 1)
                            if desc:
                                node.appearance_descriptions, node.appearance_embeddings, _ = find_or_update_description(
                                    node.appearance_descriptions, desc, status, node.appearance_embeddings
                                )

                    if structure_items:
                        for item in structure_items:
                            desc = item.get("description", "")
                            status = item.get("status", 1)
                            if desc:
                                node.structure_descriptions, node.structure_embeddings, _ = find_or_update_description(
                                    node.structure_descriptions, desc, status, node.structure_embeddings
                                )

                    if function_items:
                        for item in function_items:
                            desc = item.get("description", "")
                            status = item.get("status", 1)
                            if desc:
                                node.function_descriptions, node.function_embeddings, _ = find_or_update_description(
                                    node.function_descriptions, desc, status, node.function_embeddings
                                )

                    if component_image:
                        node.component_image = encode_image_to_base64(component_image)
                        node.image_embedding = image_encoder(component_image).tolist()

                    node.touch()
                    return node

    print("Creating new component...")
    image_embedding = None
    image_base64 = None

    if component_image:
        image_embedding = image_encoder(component_image).tolist()
        image_base64 = encode_image_to_base64(component_image)

    # 创建描述列表
    appearance_list = [make_description(item["description"], item.get("status", 1)) for item in (appearance_items or [])]
    structure_list = [make_description(item["description"], item.get("status", 1)) for item in (structure_items or [])]
    function_list = [make_description(item["description"], item.get("status", 1)) for item in (function_items or [])]

    # 计算嵌入
    appearance_emb = [text_encoder(item["description"]).tolist() for item in (appearance_items or [])]
    structure_emb = [text_encoder(item["description"]).tolist() for item in (structure_items or [])]
    function_emb = [text_encoder(item["description"]).tolist() for item in (function_items or [])]

    return ComponentNode(
        component_name=component_name,
        appearance_descriptions=appearance_list,
        structure_descriptions=structure_list,
        function_descriptions=function_list,
        component_image=image_base64,
        image_embedding=image_embedding,
        appearance_embeddings=appearance_emb,
        structure_embeddings=structure_emb,
        function_embeddings=function_emb
    )


def update_overall_memory(
    memory_db: Dict[str, Any],
    design_background: str = None,
    appearance_items: List[dict] = None,
    structure_items: List[dict] = None,
    function_items: List[dict] = None,
    overall_image: Image.Image = None,
    component_ids: List[str] = None
) -> OverallNode:
    """
    更新整体记忆节点。

    Args:
        memory_db: 记忆数据库
        design_background: 设计背景
        appearance_items: [{"description": "描述", "status": 1}, ...]
        structure_items: [{"description": "描述", "status": 1}, ...]
        function_items: [{"description": "描述", "status": 1}, ...]
        overall_image: 整体图片
        component_ids: 关联部件ID列表
    """
    print(f"\n--- Overall Memory ---")

    for node_id, data in memory_db.items():
        if data.get('node_type') == 'OVERALL':
            print("Found existing overall, updating...")
            node = OverallNode(**data)

            if design_background is not None:
                node.design_background = design_background

            if appearance_items:
                for item in appearance_items:
                    desc = item.get("description", "")
                    status = item.get("status", 1)
                    if desc:
                        node.overall_appearances, node.overall_appearance_embeddings, _ = find_or_update_description(
                            node.overall_appearances, desc, status, node.overall_appearance_embeddings
                        )

            if structure_items:
                for item in structure_items:
                    desc = item.get("description", "")
                    status = item.get("status", 1)
                    if desc:
                        node.overall_structures, node.overall_structure_embeddings, _ = find_or_update_description(
                            node.overall_structures, desc, status, node.overall_structure_embeddings
                        )

            if function_items:
                for item in function_items:
                    desc = item.get("description", "")
                    status = item.get("status", 1)
                    if desc:
                        node.overall_functions, node.overall_function_embeddings, _ = find_or_update_description(
                            node.overall_functions, desc, status, node.overall_function_embeddings
                        )

            if overall_image is not None:
                node.overall_image = encode_image_to_base64(overall_image)
                node.image_embedding = image_encoder(overall_image).tolist()

            if component_ids is not None:
                existing_ids = set(node.component_ids)
                new_ids = [cid for cid in component_ids if cid not in existing_ids]
                node.component_ids = node.component_ids + new_ids

            node.touch()
            return node

    print("Creating new overall...")
    image_embedding = None
    image_base64 = None

    if overall_image is not None:
        image_embedding = image_encoder(overall_image).tolist()
        image_base64 = encode_image_to_base64(overall_image)

    appearance_list = [make_description(item["description"], item.get("status", 1)) for item in (appearance_items or [])]
    structure_list = [make_description(item["description"], item.get("status", 1)) for item in (structure_items or [])]
    function_list = [make_description(item["description"], item.get("status", 1)) for item in (function_items or [])]

    appearance_emb = [text_encoder(item["description"]).tolist() for item in (appearance_items or [])]
    structure_emb = [text_encoder(item["description"]).tolist() for item in (structure_items or [])]
    function_emb = [text_encoder(item["description"]).tolist() for item in (function_items or [])]

    return OverallNode(
        design_background=design_background,
        overall_appearances=appearance_list,
        overall_structures=structure_list,
        overall_functions=function_list,
        overall_image=image_base64,
        image_embedding=image_embedding,
        overall_appearance_embeddings=appearance_emb,
        overall_structure_embeddings=structure_emb,
        overall_function_embeddings=function_emb,
        component_ids=component_ids or []
    )


def process_vlm_result(
    vlm_result: Dict[str, Any],
    memory_db: Dict[str, Any],
    component_image: Image.Image = None,
    source: str = "physical"
) -> Tuple[Any, str]:
    """
    处理VLM返回的分析结果，提取外形/功能/结构/设计背景信息。

    Args:
        vlm_result: VLM返回的解析后JSON对象
        memory_db: 记忆数据库
        component_image: 图片（物理世界传入）
        source: 数据来源 "physical" 或 "virtual"

    Returns:
        Tuple: (创建/更新的节点, 节点类型 "component" 或 "overall")
    """
    if vlm_result is None:
        print("[Error] VLM result is None")
        return None, None

    # 1. 提取 VLM 基础字段
    user_speaking = vlm_result.get("User Speaking", "")
    behavior_desc = vlm_result.get("Behavior description", "")
    user_intent = vlm_result.get("User intent", None)

    print(f"[VLM Result] User Speaking: {user_speaking}")
    print(f"[VLM Result] Behavior: {behavior_desc}")
    print(f"[VLM Result] User Intent: {user_intent}")

    # 2. 调用 LLM 分析设计信息
    print("\n[Analyzing design info...]")
    design_info = llm_analyze_design_info(user_speaking, behavior_desc, user_intent)

    if design_info is None:
        design_info = {
            "component": "unknown",
            "appearance": [],
            "function": [],
            "structure": [],
            "design_background": None
        }

    # 3. 从分析结果提取顶层 component 和描述列表
    component = design_info.get("component", "unknown")
    appearance_items = design_info.get("appearance", [])
    function_items = design_info.get("function", [])
    structure_items = design_info.get("structure", [])
    design_background = design_info.get("design_background", None)

    # 4. 根据 component 判断是部件还是整体
    if component == "overall":
        # 整体节点
        print(f"[Overall] Appearance: {appearance_items}")
        print(f"[Overall] Function: {function_items}")
        print(f"[Overall] Structure: {structure_items}")
        if design_background:
            print(f"[Overall] Design Background: {design_background}")

        node = update_overall_memory(
            memory_db=memory_db,
            design_background=design_background,
            appearance_items=appearance_items,
            structure_items=structure_items,
            function_items=function_items,
            overall_image=component_image
        )
        return node, "overall"

    else:
        # 部件节点
        print(f"[Component '{component}'] Appearance: {appearance_items}")
        print(f"[Component '{component}'] Function: {function_items}")
        print(f"[Component '{component}'] Structure: {structure_items}")

        node = create_component_memory(
            component_name=component,
            appearance_items=appearance_items,
            structure_items=structure_items,
            function_items=function_items,
            component_image=component_image,
            memory_db=memory_db
        )
        return node, "component"


def get_all_components(memory_db: Dict[str, Any]) -> List[ComponentNode]:
    """获取所有部件节点"""
    return [ComponentNode(**data) for node_id, data in memory_db.items() if data.get('node_type') == 'COMPONENT']


def get_overall_node(memory_db: Dict[str, Any]) -> Optional[OverallNode]:
    """获取整体节点"""
    for node_id, data in memory_db.items():
        if data.get('node_type') == 'OVERALL':
            return OverallNode(**data)
    return None


def link_component_to_overall(component_id: str, overall_id: str, memory_db: Dict[str, Any]) -> bool:
    """将部件关联到整体"""
    if overall_id not in memory_db or component_id not in memory_db:
        print("Error: Node not found")
        return False

    overall_data = memory_db[overall_id]
    if overall_data.get('node_type') != 'OVERALL':
        print("Error: Target is not Overall")
        return False

    if component_id not in overall_data.get('component_ids', []):
        overall_data['component_ids'] = overall_data.get('component_ids', []) + [component_id]
        overall_data['timestamp_last_accessed'] = datetime.now(timezone.utc).isoformat()
        print(f"Linked {component_id} to {overall_id}")
        return True

    print("Already linked")
    return True


# ========== 图片输入接口 ==========

def update_image(
    image: Image.Image,
    name: str,
    memory_db: Dict[str, Any]
) -> Tuple[Any, str]:
    """
    更新部件或整体节点的图片（直接覆盖）。

    Args:
        image: 生成的图片
        name: 部件名或 "overall"
        memory_db: 记忆数据库

    Returns:
        Tuple: (更新的节点, 节点类型 "component" 或 "overall")
    """
    print(f"\n--- Update Image: '{name}' ---")

    # 处理整体节点
    if name.lower() == "overall":
        for node_id, data in memory_db.items():
            if data.get('node_type') == 'OVERALL':
                print("Found existing overall, updating image...")
                node = OverallNode(**data)
                node.overall_image = encode_image_to_base64(image)
                node.image_embedding = image_encoder(image).tolist()
                node.touch()
                return node, "overall"

        # 没找到则创建新的整体节点（只有图片）
        print("Creating new overall with image...")
        return OverallNode(
            overall_image=encode_image_to_base64(image),
            image_embedding=image_encoder(image).tolist()
        ), "overall"

    # 处理部件节点
    for node_id, data in memory_db.items():
        if data.get('node_type') == 'COMPONENT':
            existing_name = data.get('component_name', '')
            # 名字匹配（允许大小写差异）
            if existing_name.lower() == name.lower():
                print(f"Found existing component '{existing_name}', updating image...")
                node = ComponentNode(**data)
                node.component_image = encode_image_to_base64(image)
                node.image_embedding = image_encoder(image).tolist()
                node.touch()
                return node, "component"

    # 没找到则创建新的部件节点（只有图片和名称）
    print(f"Creating new component '{name}' with image...")
    return ComponentNode(
        component_name=name,
        component_image=encode_image_to_base64(image),
        image_embedding=image_encoder(image).tolist()
    ), "component"


def batch_update_images(
    memory_db: Dict[str, Any],
    generated_folder: str = None,
    processed_folder: str = None
) -> Dict[str, Any]:
    """
    从 generated_images 文件夹批量提取图片并更新记忆节点。
    处理完后图片移动到 processed_images 文件夹。

    Args:
        memory_db: 记忆数据库
        generated_folder: 待处理图片文件夹路径（默认: 项目目录/generated_images）
        processed_folder: 已处理图片文件夹路径（默认: 项目目录/processed_images）

    Returns:
        更新后的记忆数据库
    """
    import os
    import shutil

    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

    if generated_folder is None:
        generated_folder = os.path.join(PROJECT_DIR, "generated_images")
    if processed_folder is None:
        processed_folder = os.path.join(PROJECT_DIR, "processed_images")

    # 创建文件夹（如果不存在）
    os.makedirs(generated_folder, exist_ok=True)
    os.makedirs(processed_folder, exist_ok=True)

    # 支持的图片格式
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif'}

    for filename in os.listdir(generated_folder):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in image_extensions:
            continue

        # 文件名作为部件名（去掉扩展名）
        name = os.path.splitext(filename)[0]
        image_path = os.path.join(generated_folder, filename)

        try:
            image = Image.open(image_path).convert("RGB")
            node, node_type = update_image(image, name, memory_db)
            if node:
                memory_db[node.node_id] = node.model_dump()
                print(f"[Batch] Updated {node_type}: {name}")

                # 移动到已处理文件夹
                dest_path = os.path.join(processed_folder, filename)
                shutil.move(image_path, dest_path)
                print(f"[Batch] Moved '{filename}' to processed_images")

        except Exception as e:
            print(f"[Batch] Error processing '{filename}': {e}")

    return memory_db


# ========== 主程序 ==========

if __name__ == "__main__":
    import os

    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
    IMAGE_FILE_PATH = os.path.join(PROJECT_DIR, "test.png")
    JSON_MEMORY_FILE = "object_nodes.json"

    memory_database = load_memory_from_json(JSON_MEMORY_FILE)

    # ========== Step 1: 处理 generated_images 文件夹中的图片 ==========
    print("\n--- Step 1: Processing generated_images folder ---")
    memory_database = batch_update_images(memory_database)
    save_memory_to_json(memory_database, JSON_MEMORY_FILE)

    # ========== Step 2: VLM分析（可选，需要test.png）==========
    try:
        source_frame = Image.open(IMAGE_FILE_PATH).convert("RGB")
    except FileNotFoundError:
        print(f"[Skip VLM] 找不到文件 '{IMAGE_FILE_PATH}'，跳过VLM分析")
        source_frame = None

    if source_frame:
        # 演示：模拟触发类型和语音文本（实际应由外部模块传入）
        trigger_type = "眼动焦点注视单一物体超过五秒钟"
        transcript_text = "我正在设计社区配送机器人的履带，我希望它能平稳的在社区间行走，履带整体是充满链条的，有科技感的外形，能适应不同的地面环境。"

        print("\n--- Step 2: Analyzing image with VLM ---")
        image_bytes = encode_image_to_base64(source_frame)
        vlm_response = vlm_chat_mock(image_bytes, trigger_type, transcript_text)
        print(f"[VLM] {vlm_response[:200]}...")

        vlm_result = extract_and_parse_json(vlm_response)

        if vlm_result:
            print(f"[Parsed] Type: {vlm_result.get('type')}")

            node, node_type = process_vlm_result(
                vlm_result=vlm_result,
                memory_db=memory_database,
                component_image=source_frame
            )

            if node:
                memory_database[node.node_id] = node.model_dump()
                save_memory_to_json(memory_database, JSON_MEMORY_FILE)

                print(f"\n--- {node_type} node ---")
                print(f"ID: {node.node_id}")
                if node_type == "component":
                    print(f"Name: {node.component_name}")
                    print(f"Appearance: {len(node.appearance_descriptions)}")
                    print(f"Structure: {len(node.structure_descriptions)}")
                    print(f"Function: {len(node.function_descriptions)}")
                elif node_type == "overall":
                    print(f"Background: {node.design_background}")
                    print(f"Appearances: {len(node.overall_appearances)}")
                    print(f"Functions: {len(node.overall_functions)}")
        else:
            print("[Error] Failed to parse VLM response")

    # ========== Summary ==========
    print("\n--- Summary ---")
    component_count = sum(1 for d in memory_database.values() if d.get('node_type') == 'COMPONENT')
    overall_count = sum(1 for d in memory_database.values() if d.get('node_type') == 'OVERALL')
    print(f"Components: {component_count}")
    print(f"Overall nodes: {overall_count}")