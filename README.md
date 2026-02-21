# 基于知识蒸馏与检索增强生成的轻量级古诗词生成模型



整体思路是在DeepSeek R1-8B的基座上进行预训练+知识蒸馏+格律辅助学习（SFT）

其中基准模型选择[DeepSeek-R1-0528-Qwen3-8B](https://hf-mirror.com/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B/tree/main)，教师模型选择[DeepSeek-R1-Distill-Qwen-32B](https://hf-mirror.com/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B)。

## 模型下载

模型下载代码：

```python
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
    # model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    model_id = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
    save_path = "../../resource/models/deepseek-r1-8b"
    download_model(model_id, save_path)
```

## 基准模型的预训练

目标：让模型的生成更加偏向于诗歌的整体表达。

>   古诗：原始语料 = 7：3	学习率 = 2e-5	epoch = 2

数据结构化与清洗：将CSV文件数据处理成自然语言，清洗其中的无效信息

```txt
题目：春晓
朝代：唐
作者：孟浩然
正文：春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。
<eos>
```

Tokenizer与Packing：使用HuggingFace进行自动Packing

预训练：



## 使用DeepSeek R1-32B模型作为教师模型进行知识蒸馏



## 格律辅助学习（SFT）
