from modelscope.hub.api import HubApi

YOUR_ACCESS_TOKEN = 'f7d94884-9f01-4319-9d80-fbab97d51d26'

api = HubApi()
api.login(YOUR_ACCESS_TOKEN)
api.push_model(
    model_id="wangyin717/MPP-Qwen1.5", 
    model_dir="/data1/vllm/MPP-LLaVA-Qwen1.5-github/MPP-Qwen1.5/lavis/output/weight" # 本地模型目录，要求目录中必须包含configuration.json
)
