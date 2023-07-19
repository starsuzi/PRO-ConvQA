DATE=$(date +%Y_%m_%d)/$(date +%H_%M_%S)
# Please change to your DUMP_DIR and LOAD_DIR as your path to phrase dump and path to the query-finetuned model
CUDA_VISIBLE_DEVICES=0 make eval-query-orquac MODEL_NAME=densephrases-orquac/query_finetune/eval/${DATE} DUMP_DIR=outputs/densephrases-orquac/2023_06_29/21_08_07_wiki-20181220/dump/ LOAD_DIR=densephrases-orquac/query_finetune/2023_07_03/20_07_07