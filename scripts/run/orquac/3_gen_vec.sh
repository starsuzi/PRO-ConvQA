#5621
# Please change MODEL_NAME to your trained phrase retriever.
CUDA_VISIBLE_DEVICES=0 make gen-vecs-parallel MODEL_NAME=densephrases-orquac/2023_06_29/21_08_07 START=0 END=5621