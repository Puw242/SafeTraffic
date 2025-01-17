#! /bin/bash

cd train/sft/
if [ $model_size ];then
	echo "model_size = $model_size"
else
	model_size=7b
    echo "No model size is given, set model_size = $model_size"
fi

if [ $checkpoint_path ];then
	echo "checkpoint_path = $checkpoint_path"
else
	checkpoint_path=../../logs/${model_size}_inj/
    echo "No checkpoint path is given, set checkpoint_path = $checkpoint_path"
fi

output_model=../../logs/final/${model_size}_type_test
if [ ! -d ${output_model} ];then
    mkdir -p ${output_model}
fi
export NCCL_P2P_DISABLE=1

CUDA_VISIBLE_DEVICES=0 python finetune_clm_lora.py \
    --model_name_or_path meta-llama/Llama-2-${model_size}-hf \
    --task_type accident_type \
    --train_files ../../data/hsis_wa_0to20k_TaskType_Monthrest_nonUni.csv \
    --validation_files  ../../data/hsis_wa_0to20k_TaskType_MonthJanJunDec_Uni.csv \
    --resume_from_checkpoint ${checkpoint_path} \
    --metric_for_best_model eval_accuracy \
    --per_device_eval_batch_size 4 \
    --do_eval \
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
    --block_size 2048 \
    --report_to tensorboard \
    --overwrite_output_dir \
    --ignore_data_skip true \
    --bf16 \
    --gradient_checkpointing \
    --bf16_full_eval \
    --ddp_timeout 18000000 \
    | tee -a ${output_model}/train.log

echo ${output_model}