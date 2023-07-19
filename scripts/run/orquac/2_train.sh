DATE=$(date +%Y_%m_%d)/$(date +%H_%M_%S)

CUDA_VISIBLE_DEVICES=1 make run-rc-orquac-hisContra MODEL_NAME=densephrases-orquac/${DATE}
