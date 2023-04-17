# MODELS_PATH=/hy-tmp
MODELS_PATH=/opt/local/llm_models/LLaMA/huggingface.co
BASE_MODEL_PATH=${MODELS_PATH}/llama-13b-hf
VICUNA_MODEL_PATH=${MODELS_PATH}/vicuna-13b-v1.1
#VICUNA_MODEL_PATH=${MODELS_PATH}/vicuna-13b-v0

apply_delta:
	python3 -m fastchat.model.apply_delta \
	--base-model-path ${BASE_MODEL_PATH} \
	--delta-path ${MODELS_PATH}/vicuna-13b-delta-v1.1 \
	--target-model-path ${VICUNA_MODEL_PATH} && \
	cp special_tokens_map.json ${VICUNA_MODEL_PATH}/

local_chat:
	python3 -m fastchat.serve.cli \
		--style rich \
		--model-path ${VICUNA_MODEL_PATH} \
		--load-8bit \
		--max-new-tokens 1024 \
		--temperature 0.9 \
		--num-gpus 1

finetune_cot:
	python finetune_cot.py \
		--model_type llama \
		--model_name_or_path LLaMA/vicuna-13b-v1.1 \
		--data alpaca-belle-cot \
		--lora_target_modules q_proj v_proj \
		--per_gpu_train_batch_size 16 \
		--gradient_accumulation_steps 8 \
		--learning_rate 3e-4 \
		--epochs 3 && \
	./upload.sh

	# python -m torch.distributed.launch --nproc_per_node 2  \
    # --nnodes=1 --node_rank=-1 --master_port=13579 \

ddp_finetune_cot:
	torchrun --nproc_per_node=6 --master_port=13579 \
	finetune_cot.py \
    --model_type llama \
	--model_name_or_path LLaMA/vicuna-13b-v1.1 \
    --data alpaca-belle-cot \
	--lora_target_modules q_proj v_proj \
    --per_gpu_train_batch_size 8 \
	--gradient_accumulation_steps 16 \
	--learning_rate 3e-4 \
	--epochs 1 && \
	./upload.sh

local_cot_chat:
	python server.py \
		--model_type llama \
		--size 13b \
		--lora_dir saved_models/llama-13b-hf_alpaca-belle-cot
