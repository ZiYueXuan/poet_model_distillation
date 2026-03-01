#!/bin/bash

# 获取当前脚本所在的目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "脚本所在目录: $SCRIPT_DIR"

# 获取项目根目录（向上两级）
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
echo "项目根目录: $PROJECT_ROOT"

# 设置 PYTHONPATH 为项目根目录
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
echo "PYTHONPATH 已设置为: $PROJECT_ROOT"

# 检查模型路径是否存在（根据你的代码中的路径）
MODEL_PATH="${PROJECT_ROOT}/resource/models/deepseek-r1-14b"
echo "模型路径: $MODEL_PATH"

if [ ! -d "$MODEL_PATH" ]; then
    echo "警告: 模型路径不存在: $MODEL_PATH"
    echo "请检查模型文件是否正确下载"
fi

# 运行 Python 脚本
echo "开始运行 teacher_multi_generation.py..."
pdm run python teacher_multi_generation.py