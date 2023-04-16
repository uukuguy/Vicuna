MODELS_PATH=/hy-tmp
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
		--conv-template vicuna_v1.1 \
		--max-new-tokens 1024 \
		--temperature 0.9 \
		--num-gpus 1
