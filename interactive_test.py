"""
interactive_test.py - 交互式 API 测试工具

手动控制测试流程，逐步调用各个接口。

使用方法：
1. 先启动 API 服务：python api.py
2. 在另一个终端运行此脚本：python interactive_test.py
"""

import requests
import os

BASE_URL = "http://localhost:5000"

def print_response(response):
    """打印响应结果"""
    try:
        data = response.json()
        print("\n" + "-"*40)
        print("响应结果:")
        print("-"*40)
        print(json.dumps(data, indent=2, ensure_ascii=False))
    except:
        print(f"\n响应状态码: {response.status_code}")
        print(f"响应内容: {response.text}")

import json

def interactive_test():
    """交互式测试"""

    print("\n" + "="*60)
    print("IntentRelay API 交互式测试工具")
    print("="*60)
    print("请确保 API 服务已启动：python api.py")
    print("="*60)

    while True:
        print("\n" + "-"*50)
        print("选择要执行的接口:")
        print("-"*50)
        print("【系统控制】")
        print("  1. GET  /health          - 健康检查")
        print("  2. POST /init            - 初始化系统")
        print("  3. POST /stop            - 停止系统")

        print("\n【图像生成】")
        print("  4. POST /generate_prompt  - 生成提示词")
        print("  5. POST /generate_image   - 调用 ComfyUI 生成")

        print("\n【状态查询】")
        print("  6. GET  /memory_status    - 查询记忆状态")
        print("  7. GET  /components_list  - 获取部件列表")
        print("  8. GET  /components_info  - 获取部件详情")
        print("  9. POST /update_description - 修改部件描述内容")
        print("  10. POST /memory_qa       - 单轮问答：问一句答一句")
        print("  11. GET /uncertain_suggestions - 获取不确定内容建议")

        print("\n【VLM 分析】")
        print("  12. POST /vlm_analysis    - 语音触发 VLM 分析")

        print("\n【AI反馈】")
        print("  13. POST /ai_feedback         - 生成AI设计建议")

        print("\n【用户反馈】")
        print("  14. POST /user_feedback       - 处理用户反馈（更新评分权重）")

        print("\n【mico=1问答】")
        print("  15. POST /qa_switch           - 问答处理（mico=1→0切换时调用）")
        print("  16. POST /generate_from_answer - 从最新AI回答生成图片")

        print("\n【其他】")
        print("  20. 清空屏幕")
        print("  0. 退出")
        print("-"*50)

        choice = input("\n请输入编号: ").strip()

        if choice == "0":
            print("退出测试")
            break

        elif choice == "1":
            print("\n>>> 健康检查")
            r = requests.get(f"{BASE_URL}/health")
            print_response(r)

        elif choice == "2":
            print("\n>>> 初始化系统")
            r = requests.post(f"{BASE_URL}/init")
            print_response(r)

        elif choice == "3":
            print("\n>>> 停止系统")
            r = requests.post(f"{BASE_URL}/stop")
            print_response(r)

        elif choice == "4":
            print("\n>>> 生成提示词")
            mode = input("请输入 mode（1=部件，2=整体）: ").strip()

            if mode == "1":
                component_name = input("请输入部件名称（如 履带）: ").strip()
                r = requests.post(
                    f"{BASE_URL}/generate_prompt",
                    json={"mode": 1, "component_name": component_name}
                )
            elif mode == "2":
                r = requests.post(
                    f"{BASE_URL}/generate_prompt",
                    json={"mode": 2}
                )
            else:
                print("无效的 mode")
                continue
            print_response(r)

        elif choice == "5":
            print("\n>>> 调用 ComfyUI 生成图片")
            mode = input("请输入 mode（1=部件，2=整体）: ").strip()
            prompt = input("请输入提示词: ").strip()

            if mode == "1":
                component_name = input("请输入部件名称（如 履带）: ").strip()
                r = requests.post(
                    f"{BASE_URL}/generate_image",
                    json={
                        "mode": 1,
                        "prompt": prompt,
                        "component_name": component_name
                    }
                )
            elif mode == "2":
                r = requests.post(
                    f"{BASE_URL}/generate_image",
                    json={
                        "mode": 2,
                        "prompt": prompt
                    }
                )
            else:
                print("无效的 mode")
                continue
            print_response(r)

        elif choice == "6":
            print("\n>>> 查询记忆状态")
            r = requests.get(f"{BASE_URL}/memory_status")
            print_response(r)

        elif choice == "7":
            print("\n>>> 获取部件列表")
            r = requests.get(f"{BASE_URL}/components_list")
            print_response(r)

        elif choice == "8":
            print("\n>>> 获取部件详情")
            r = requests.get(f"{BASE_URL}/components_info")
            print_response(r)

        elif choice == "9":
            print("\n>>> 修改部件描述内容")
            target_name = input("请输入部件名称（或'整体'): ").strip()
            desc_type = input("请输入描述类型（结构/功能/不确定点/外形): ").strip()
            old_content = input("请输入原来的内容: ").strip()
            new_content = input("请输入修改后的内容: ").strip()

            if target_name and desc_type and old_content and new_content:
                r = requests.post(
                    f"{BASE_URL}/update_description",
                    json={
                        "target_name": target_name,
                        "desc_type": desc_type,
                        "old_content": old_content,
                        "new_content": new_content
                    }
                )
                print_response(r)
            else:
                print("所有参数都不能为空")

        elif choice == "10":
            print("\n>>> 多轮问答（AI生成最多3个问题，逐个询问）")
            print("流程：AI生成问题列表 → 逐个问用户 → 用户直接回答")
            print("-"*40)

            # 开始新一轮问答
            r = requests.post(f"{BASE_URL}/memory_qa", json={})
            result = r.json()
            print_response(r)

            # 如果有问题，循环让用户回答
            while result.get('has_questions') and result.get('remaining_count', 0) > 0:
                questions = result.get('questions', [])
                current_index = result.get('current_index', 0)
                current_question = result.get('current_question') or (questions[current_index] if current_index < len(questions) else None)

                if current_question:
                    print("\n" + "="*40)
                    print(f"问题 [{current_index + 1}/{len(questions)}]:")
                    print(f"AI问：{current_question.get('question', '')}")
                    print(f"（关于 {current_question.get('target', '')} 的 {current_question.get('desc_type', '')}）")
                    print("="*40)

                    answer = input("请直接回答（输入 '跳过' 跳过此问题，或输入内容回答）: ").strip()

                    if answer.lower() == '跳过':
                        print("跳过此问题，继续下一个...")
                        # 发送空回答表示跳过，但不结束流程
                        r = requests.post(
                            f"{BASE_URL}/memory_qa",
                            json={"answer": "SKIP"}  # 特殊标记表示跳过
                        )
                        result = r.json()
                        print_response(r)
                        continue

                    if not answer:
                        print("回答不能为空，请重新输入或输入'跳过'")
                        continue

                    # 发送回答
                    r = requests.post(
                        f"{BASE_URL}/memory_qa",
                        json={"answer": answer}
                    )
                    result = r.json()
                    print_response(r)
                else:
                    break

            print("\n问答结束")

        elif choice == "11":
            print("\n>>> 获取不确定内容建议")
            r = requests.get(f"{BASE_URL}/uncertain_suggestions")
            print_response(r)

        elif choice == "12":
            print("\n>>> 语音触发 VLM 分析")
            text = input("请输入语音文本: ").strip()
            if text:
                r = requests.post(
                    f"{BASE_URL}/vlm_analysis",
                    json={"transcript_text": text}
                )
                print_response(r)
            else:
                print("语音文本不能为空")

        elif choice == "13":
            print("\n>>> 生成AI设计建议")
            component_name = input("请输入部件名称: ").strip()
            if component_name:
                # 需要 parsed_vlm 参数，这里用模拟数据
                print("注意：此接口需要 parsed_vlm 参数（来自 /vlm_analysis 返回值）")
                print("这里使用模拟数据进行测试...")
                r = requests.post(
                    f"{BASE_URL}/ai_feedback",
                    json={
                        "component_name": component_name,
                        "parsed_vlm": {
                            "User intent": "Appearance design",
                            "Behavior description": "用户正在查看履带的外形",
                            "User Speaking": "希望更圆润一些"
                        }
                    }
                )
                print_response(r)
            else:
                print("部件名称不能为空")

        elif choice == "14":
            print("\n>>> 处理用户反馈")
            feedback = input("请输入反馈内容（如：可以更新颖一些）: ").strip()
            if feedback:
                r = requests.post(
                    f"{BASE_URL}/user_feedback",
                    json={"feedback": feedback}
                )
                print_response(r)
            else:
                print("反馈内容不能为空")

        elif choice == "15":
            print("\n>>> mico=1 问答处理")
            print("说明：模拟 mico=1→0 切换时的问答流程")
            # 提供默认问题
            question = input("请输入问题（默认：你觉得履带应该设计成什么风格？）: ").strip()
            if not question:
                question = "你觉得履带应该设计成什么风格？"
            print(f"问题: {question}")
            # 调用问答接口（传入问题）
            r = requests.post(
                f"{BASE_URL}/qa_switch",
                json={"question": question}
            )
            print_response(r)
            result = r.json()
            if result.get("success") and result.get("answer"):
                print("\n提示：AI回答已记录，可以调用选项16生成图片")

        elif choice == "16":
            print("\n>>> 从最新AI回答生成图片")
            print("说明：此功能自动使用mico=1模式下最新的AI回答，生成后自动清空记录")
            print("注意：不需要输入文件名，系统自动命名")
            r = requests.post(
                f"{BASE_URL}/generate_from_answer",
                json={}  # 不需要参数，自动命名
            )
            print_response(r)

        elif choice == "20":
            print("\n" * 50)

        else:
            print("无效的编号，请重新输入")

        input("\n按 Enter 继续...")


if __name__ == "__main__":
    interactive_test()