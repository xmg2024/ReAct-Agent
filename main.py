import json
from dotenv import load_dotenv
import os
from agent import CustomerServiceAgent
from op_llm_client import OllamaClient
from tools.query_product_data import query_by_product_name
from tools.calc import calculate
from tools.read_promotions import read_store_promotions
from openai import OpenAI

load_dotenv()
import re


def load_config():
    with open('config.json', 'r') as f:
        return json.load(f)


def get_client(config):
    if config['openai'].get('use_model', True):
        api_key = os.environ.get("API_KEY")
        if not api_key:
            raise ValueError("API_KEY not found in environment variables. Please check your .env file.")
        
        # 支持自定义 base_url（用于代理或第三方 API 服务）
        base_url = os.environ.get("OPENAI_BASE_URL", None)
        
        # 构建客户端参数
        client_params = {"api_key": api_key}
        if base_url:
            client_params["base_url"] = base_url
        
        return OpenAI(**client_params)
    else:
        return OllamaClient()


def get_max_iterations(config):
    # 选择使用的模型来确定最大迭代次数
    if config['ollama']['use_model']:
        return config['ollama']['max_iterations']
    elif config['openai']['use_model']:
        return config['openai']['max_iterations']
    else:
        return 10  # 如果没有启用任何模型，可以设置一个默认的迭代次数


def main():
    config = load_config()
    try:
        # 检查环境变量是否加载
        api_key = os.environ.get("API_KEY")
        if not api_key:
            print("警告: API_KEY 未在环境变量中找到")
            print("请确保 .env 文件存在且包含 API_KEY=your_key_here")
        
        # 获取服务端实例（OpenAI API 或者 Ollama Resuful API）
        client = get_client(config)

        # 实例化Agent
        agent = CustomerServiceAgent(client, config)
        print("AI 客户端初始化成功！")
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        print(f"Error initializing the AI client:")
        print(f"  Error Type: {error_type}")
        print(f"  Error Message: {error_msg}")
        
        if "API_KEY" in error_msg or "api_key" in error_msg.lower():
            print("\n请检查:")
            print("  1. .env 文件是否存在且格式正确")
            print("  2. API_KEY 是否正确设置（不要包含引号）")
            print("  3. 确保 .env 文件在项目根目录下")
        elif "Connection" in error_type or "connection" in error_msg.lower():
            print("\n连接问题，请检查:")
            print("  1. 网络连接是否正常")
            print("  2. 是否需要配置代理")
            print("  3. 可以在 .env 中添加 OPENAI_BASE_URL 使用代理服务")
        
        print("\nPlease check your configuration and ensure the AI service is running.")
        return

    tools = {
        "query_by_product_name": query_by_product_name,
        "read_store_promotions": read_store_promotions,
        "calculate": calculate,
    }

    # 主循环用于多次用户输入
    while True:
        query = input("输入您的问题或输入 '退出' 来结束: ")
        if query.lower() == '退出':
            break

        iteration = 0
        max_iterations = get_max_iterations(config)
        while iteration < max_iterations:  # 内部循环用于处理每一条 query
            try:
                result = agent(query)
                action_re = re.compile('^Action: (\w+): (.*)$')
                actions = [action_re.match(a) for a in result.split('\n') if action_re.match(a)]
                if actions:
                    action_parts = result.split("Action:", 1)[1].strip().split(": ", 1)
                    tool_name = action_parts[0]
                    tool_args = action_parts[1] if len(action_parts) > 1 else ""
                    if tool_name in tools:
                        try:
                            observation = tools[tool_name](tool_args)
                            query = f"Observation: {observation}"
                        except Exception as e:
                            query = f"Observation: Error occurred while executing the tool: {str(e)}"
                    else:
                        query = f"Observation: Tool '{tool_name}' not found"
                elif "Answer:" in result:
                    print(f"客服回复：{result.split('Answer:', 1)[1].strip()}")
                    break  # 收到答案后结束内部循环
                else:
                    query = "Observation: No valid action or answer found. Please provide a clear action or answer."

            except Exception as e:
                error_type = type(e).__name__
                error_msg = str(e)
                print(f"An error occurred while processing the query:")
                print(f"  Error Type: {error_type}")
                print(f"  Error Message: {error_msg}")
                
                # 提供更具体的错误诊断
                if "Connection" in error_type or "connection" in error_msg.lower():
                    print("\n连接错误诊断:")
                    print("  1. 检查网络连接是否正常")
                    print("  2. 如果在中国大陆，可能需要配置代理或使用第三方 API 服务")
                    print("  3. 可以在 .env 文件中设置 OPENAI_BASE_URL 来使用代理或第三方服务")
                    print("     例如: OPENAI_BASE_URL=https://api.openai-proxy.com/v1")
                elif "API" in error_type or "api" in error_msg.lower() or "key" in error_msg.lower():
                    print("\nAPI Key 错误诊断:")
                    print("  1. 检查 .env 文件中的 API_KEY 是否正确")
                    print("  2. 确保 API_KEY 没有多余的引号")
                    print("  3. 检查 API key 是否有效且未过期")
                
                print("\nPlease check your configuration and ensure the AI service is running.")
                break

            iteration += 1

        if iteration == max_iterations:
            print("Reached maximum number of iterations without a final answer.")


if __name__ == "__main__":
    main()
