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

        print("\n【VLM 分析】")
        print("  9. POST /vlm_analysis    - 语音触发 VLM 分析")

        print("\n【其他】")
        print("  10. 清空屏幕")
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

        elif choice == "10":
            print("\n" * 50)

        else:
            print("无效的编号，请重新输入")

        input("\n按 Enter 继续...")


if __name__ == "__main__":
    interactive_test()