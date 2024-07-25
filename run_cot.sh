#!/bin/bash

# 设置Horovod超时
export HOROVOD_TIMEOUT=3600 # 设置为1小时（单位：秒）

# 设置日志目录和文件路径
LOG_DIR="./log"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/training_$TIMESTAMP.log"
FIXED_LOG_NAME="$LOG_DIR/latest_training_cot.log"

# 创建日志目录（如果不存在）
mkdir -p "$LOG_DIR"

# 确保日志目录创建成功
if [ ! -d "$LOG_DIR" ]; then
    echo "错误：无法创建日志目录 $LOG_DIR"
    exit 1
fi

# 使用nohup运行训练脚本，将输出重定向到日志文件
bash -c '
    /mmu_nlp_hdd/suzhou03/models/aimo-progress-prize/myenv_python3.10_numina/bin/accelerate launch \
    --config_file=training/configs/deepspeed_zero3.yaml \
    training/sft.py \
    training/configs/stage-1-cot.yaml
' > "$LOG_FILE" 2>&1 &

# 获取后台进程ID
BG_PID=$!

# 创建或更新软链接到最新的日志文件
ln -s "$LOG_FILE" "$FIXED_LOG_NAME"

echo "训练脚本已在后台启动，进程ID: $BG_PID"
echo "日志被写入到文件: $LOG_FILE"
echo "最新日志的软链接: $FIXED_LOG_NAME"
echo "你可以使用 'tail -f $FIXED_LOG_NAME' 来实时查看最新日志"
echo "使用 'kill $BG_PID' 来停止训练进程"