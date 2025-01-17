#!/bin/bash
# #SBATCH --job-name=cuda_version_check
# #SBATCH --output=cuda_version.txt
# #SBATCH --partition=a100  # Specify the partition
# #SBATCH --gres=gpu:1  # Request 1 GPU
# #SBATCH --time=00:05:00  # Set a short job runtime
# #SBATCH --ntasks=1  # Number of tasks

# # Load the necessary CUDA module (if your cluster uses module environment)
# module load anaconda3/2023.09-0
# module load cuda/12.1.0

# conda create -n CLLM python=3.10
# source activate CLLM

# pip install -r requirements.txt

cd train/sft/
if [ $model_size ];then
	echo "model_size = $model_size"
else
	model_size=7b
    echo "No model size is given, set model_size = $model_size"
fi
output_model=../../logs/${model_size}_inj

if [ ! -d ${output_model} ];then
    mkdir -p ${output_model}
fi
export NCCL_P2P_DISABLE=1
cp ./train_scripts/inj.sh ${output_model}

if [ $master_port ];then
	echo "master_port = $master_port"
else
	master_port=50001
    echo "model_size = $model_size"
fi

if [ $include ];then
	echo "include = $include"
else
	include=localhost:0
    echo "include = $include"
fi

deepspeed --master_port $master_port --include $include finetune_clm_lora.py \
    --model_name_or_path meta-llama/Llama-2-${model_size}-hf \
    --task_type number_of_injuried_people \
    --train_files ../../data/hsis_wa_0to20k_TaskInj_Monthrest_nonUni.csv \
    --validation_files  ../../data/hsis_wa_0to20k_TaskInj_MonthJanJunDec_Uni.csv \
    --metric_for_best_model eval_accuracy \
    --save_total_limit 2 \
    --load_best_model_at_end true \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --use_fast_tokenizer false \
    --output_dir ${output_model} \
    --evaluation_strategy  steps \
    --max_eval_samples 1000 \
    --learning_rate 3e-4 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 2 \
    --warmup_steps 50 \
    --load_in_bits 4 \
    --lora_r 16 \
    --lora_alpha 32 \
    --target_modules q_proj,k_proj,v_proj,o_proj,down_proj,gate_proj,up_proj \
    --logging_dir ${output_model}/logs \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --preprocessing_num_workers 10 \
    --save_steps 50 \
    --eval_steps 50 \
    --seed 42 \
    --disable_tqdm false \
    --ddp_find_unused_parameters false \
    --block_size 2048 \
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