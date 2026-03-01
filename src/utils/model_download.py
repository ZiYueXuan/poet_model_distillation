from huggingface_hub import snapshot_download

def download_model(model_id, save_path):
    print(f"开始下载模型 {model_id}...")
    snapshot_download(
        repo_id=model_id,
        local_dir=save_path,
        endpoint="https://hf-mirror.com"
    )
    print(f"模型 {model_id} 下载完成，保存路径为 {save_path}！")


if __name__ == '__main__':
    model_id = "deepseek-ai/DeepSeek-V2-Lite"
    # model_id = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
    save_path = "../../resource/models/deepseek-v2-lite"
    download_model(model_id, save_path)