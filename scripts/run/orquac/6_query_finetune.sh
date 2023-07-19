# Query-side fine-tune (model will be saved as MODEL_NAME)
DATE=$(date +%Y_%m_%d)/$(date +%H_%M_%S)
# Please change to your DUMP_DIR and LOAD_DIR as your path to phrase dump and path to the trained model
make train-query-orquac MODEL_NAME=densephrases-orquac/query_finetune/${DATE} DUMP_DIR=outputs/densephrases-orquac/2023_06_29/21_08_07_wiki-20181220/dump/ LOAD_DIR=outputs/densephrases-orquac/2023_06_29/21_08_07