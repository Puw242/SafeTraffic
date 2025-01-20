#!/bin/bash
# Module and Environment Setup
module load anaconda3/2023.09-0
module load cuda/12.1.0
source activate CLLM

# Default Values
model_size="8B"
model_type="Llama-3.1"
master_port=$((RANDOM % 50 + 50000))
include="localhost:0,1"
predict="number_of_injuried_people"
predict_short="inj"
data_source="IL"
save_steps=50
eval_steps=50
load_best_model_at_end=true
output_dir=""

# Parse Command Line Arguments
while getopts "s:t:p:i:m:o:h" opt; do
  case $opt in
    s) model_size="$OPTARG" ;; # Model size
    t) model_type="$OPTARG" ;; # Model type
    p) predict="$OPTARG" ;; # Predict task
    i) include="$OPTARG" ;; # Include
    m) master_port="$OPTARG" ;; # Master port
    o) output_dir="$OPTARG" ;; # Output model directory
    h) echo "Usage: $0 [-s model_size] [-t model_type] [-p predict_task] [-i include] [-m master_port] [-o output_dir]"; exit 0 ;;
    *) echo "Invalid option: -$OPTARG"; exit 1 ;;
  esac
done

# Derive Short Name for Predict
case "$predict" in
    "number_of_injuried_people")
        predict_short="inj"
        ;;
    "severity")
        predict_short="sev"
        ;;
    "accident_type")
        predict_short="type"
        ;;
esac

# Create Output Directory
mkdir -p ${output_dir}
cp ../../scripts/${predict_short}.slurm ${output_dir}

# Run Deepspeed
cd ../train/sft/
deepspeed --master_port $master_port --include $include finetune_clm_lora.py \
    --model_name_or_path meta-llama/${model_type}-${model_size} \
    --task_type ${predict} \
    --train_files ../../data/${data_source}/train/${predict_short}.csv \
    --validation_files ../../data/${data_source}/val/${predict_short}.csv \
    --test_files ../../data/${data_source}/test/${predict_short}.csv \
    --data_source ${data_source} \
    --metric_for_best_model eval_f1 \
    --save_total_limit 2 \
    --load_best_model_at_end ${load_best_model_at_end} \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --do_predict \
    --use_fast_tokenizer false \
    --output_dir ${output_dir} \
    --evaluation_strategy steps \
    --max_eval_samples 9999 \
    --learning_rate 3e-4 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 2 \
    --warmup_steps 50 \
    --load_in_bits 4 \
    --lora_r 8 \
    --lora_alpha 16 \
    --target_modules q_proj,k_proj,v_proj,o_proj,down_proj,gate_proj,up_proj \
    --logging_dir ${output_dir}/logs \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --preprocessing_num_workers 10 \
    --save_steps ${save_steps} \
    --eval_steps ${eval_steps} \
    --seed 42 \
    --disable_tqdm false \
    --ddp_find_unused_parameters false \
    --block_size 20480 \
    --report_to tensorboard \
    --overwrite_output_dir \
    --deepspeed ds_config_zero2.json \
    --ignore_data_skip true \
    --bf16 \
    --gradient_checkpointing \
    --bf16_full_eval \
    --ddp_timeout 18000000 \
    | tee -a ${output_dir}/train.log

# Print Output Directory
echo ${output_dir}
