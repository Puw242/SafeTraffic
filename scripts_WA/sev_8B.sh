#!/bin/bash
# Load the necessary CUDA module (if your cluster uses module environment)

module load anaconda3/2023.09-0
module load cuda/12.1.0

source activate TrafficSafe

model_size=8B
export HF_HOME=/scratch4/haofrankyang/yang/cache/huggingface
master_port=$((RANDOM % 50 + 50000))
include=localhost:0
predict=severity
predict_short=sev
data_source=WA
dataset=text
cur_date=1113
checkpoint_path=/scratch4/haofrankyang/yang/logs/1008/train_WA_text/8B_sev_4/checkpoint-354

cd train/sft/

output_model=/scratch4/haofrankyang/yang/logs/${cur_date}/test_${data_source}_${dataset}/${model_size}_${predict_short}_4

if [ ! -d ${output_model} ];then
    mkdir -p ${output_model}
fi
export NCCL_P2P_DISABLE=1
cp ../../scripts/${predict_short}.slurm ${output_model}

deepspeed --master_port $master_port --include $include finetune_clm_lora.py \
    --model_name_or_path meta-llama/Llama-3.1-${model_size} \
    --task_type ${predict} \
    --resume_from_checkpoint ${checkpoint_path} \
    --train_files ../../../data/${data_source}/${dataset}/train/${predict_short}.csv \
    --validation_files  ../../../data/${data_source}/${dataset}/val/${predict_short}.csv \
    --test_files ../../../data/${data_source}/${dataset}/test/${predict_short}.csv \
    --data_source ${data_source} \
    --metric_for_best_model eval_f1 \
    --per_device_eval_batch_size 4 \
    --do_predict \
    --use_fast_tokenizer false \
    --output_dir ${output_model} \
    --evaluation_strategy  steps \
    --max_eval_samples 9999 \
    --load_in_bits 4 \
    --lora_r 16 \
    --lora_alpha 32 \
    --target_modules q_proj,k_proj,v_proj,o_proj,down_proj,gate_proj,up_proj \
    --logging_dir ${output_model}/logs \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --preprocessing_num_workers 10 \
    --seed 42 \
    --disable_tqdm false \
    --ddp_find_unused_parameters false \
    --block_size 20480 \
    --report_to tensorboard \
    --overwrite_output_dir \
    --ignore_data_skip true \
    --bf16 \
    --gradient_checkpointing \
    --bf16_full_eval \
    --ddp_timeout 18000000 \
    | tee -a ${output_model}/train.log

echo ${output_model}