#!/bin/bash
# Load the necessary CUDA module (if your cluster uses module environment)

module load anaconda3/2023.09-0
module load cuda/12.1.0

source activate CLLM

model_size=8B
model_type=Llama-3.1
export HF_HOME=/scratch4/haofrankyang/yang/cache/huggingface
master_port=$((RANDOM % 50 + 50000))
include=localhost:0,1
predict=number_of_injuried_people
predict_short=inj
data_source=IL
dataset=text
cur_date=1112
save_steps=50
eval_steps=50
load_best_model_at_end=true

cd train/sft/

output_model=/scratch4/haofrankyang/yang/logs/${cur_date}/train_${data_source}_${dataset}/${model_type}_${model_size}_${predict_short}

if [ ! -d ${output_model} ];then
    mkdir -p ${output_model}
fi
export NCCL_P2P_DISABLE=1
cp ../../scripts/${predict_short}.slurm ${output_model}

deepspeed --master_port $master_port --include $include finetune_clm_lora.py \
    --model_name_or_path meta-llama/${model_type}-${model_size} \
    --task_type ${predict} \
    --train_files ../../../data/${data_source}/${dataset}/train/${predict_short}.csv \
    --validation_files  ../../../data/${data_source}/${dataset}/val/${predict_short}.csv \
    --test_files ../../../data/${data_source}/${dataset}/test/${predict_short}.csv \
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
    --output_dir ${output_model} \
    --evaluation_strategy  steps \
    --max_eval_samples 9999 \
    --learning_rate 3e-4 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 2 \
    --warmup_steps 50 \
    --load_in_bits 4 \
    --lora_r 8 \
    --lora_alpha 16 \
    --target_modules q_proj,k_proj,v_proj,o_proj,down_proj,gate_proj,up_proj \
    --logging_dir ${output_model}/logs \
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
    | tee -a ${output_model}/train.log

echo ${output_model}