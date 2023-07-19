############################## Single-passage Training + Normalization ###################################

model-name:
ifeq ($(MODEL_NAME),)
	echo "Please set MODEL_NAME before training (e.g., MODEL_NAME=test)"; exit 2;
endif

load-dir:
ifeq ($(LOAD_DIR),)
	echo "Please set LOAD_DIR before training (e.g., LOAD_DIR=test)"; exit 2;
endif

# Dataset paths for single-passage training (QG, train, dev, semi-od)
	
# TODO 
#orquac-rc-data:
orquac-rc-hisContra-data:
	$(eval TRAIN_DATA=orquac/preprocessed/retriever/train_answer_6_hisContra.json)
	$(eval DEV_DATA=orquac/preprocessed/retriever/dev_answer_6_hisContra.json)
# 	$(eval OPTIONS=--truecase)


# Choose hyperparameter
pbn-param:
	$(eval PBN_OPTIONS=--pbn_size 2)


# TODO LAMBDA_HIS_NEG=1 LAMBDA_KL=2.0 LAMBDA_NEG=4.0
orquac-hisContra-param:
	$(eval BS=24)
	$(eval LR=3e-5)
	$(eval MAX_SEQ_LEN=384)
	$(eval LAMBDA_KL=2.0)
	$(eval LAMBDA_NEG=4.0)
	$(eval LAMBDA_HIS_NEG=1.0)
	$(eval TEACHER_NAME=spanbert-base-cased-orquac)


# Choose index size
small-index:
	$(eval NUM_CLUSTERS=256)
	$(eval INDEX_TYPE=OPQ96)
medium1-index:
	$(eval NUM_CLUSTERS=16384)
	$(eval INDEX_TYPE=OPQ96)
medium2-index:
	$(eval NUM_CLUSTERS=131072)
	$(eval INDEX_TYPE=OPQ96)
large-index:
	$(eval NUM_CLUSTERS=1048576)
	$(eval INDEX_TYPE=OPQ96)
large-index-sq:
	$(eval NUM_CLUSTERS=1048576)
	$(eval INDEX_TYPE=SQ4)


# TODO
# Followings are template commands. See 'run-rc-nq' for a detailed use.
# 1) Training phrase and question encoders on reading comprehension.
train-rc-orquac-hisContra: model-name orquac-rc-hisContra-data orquac-hisContra-param
	mkdir -p $(SAVE_DIR)/$(MODEL_NAME)
	python train_rc_hisContra.py \
		--pretrained_name_or_path SpanBERT/spanbert-base-cased \
		--cache_dir $(CACHE_DIR) \
		--train_file $(DATA_DIR)/convqa/$(TRAIN_DATA) \
		--validation_file $(DATA_DIR)/convqa/$(DEV_DATA) \
		--do_train \
		--do_eval \
		--per_device_train_batch_size $(BS) \
		--learning_rate $(LR) \
		--fp16 \
		--num_train_epochs 3 \
		--max_seq_length $(MAX_SEQ_LEN) \
		--doc_stride 128 \
		--lambda_kl $(LAMBDA_KL) \
		--lambda_neg $(LAMBDA_NEG) \
		--lambda_his_neg $(LAMBDA_HIS_NEG) \
		--lambda_flt 1.0 \
		--filter_threshold -2.0 \
		--append_title \
		--evaluate_during_training \
		--output_dir $(SAVE_DIR)/$(MODEL_NAME) \
		--teacher_dir $(SAVE_DIR)/$(TEACHER_NAME) \
		--overwrite_output_dir \
		--overwrite_cache \
		--logging_steps=10 \
		$(OPTIONS)



# 2) Trained phrase encoders can be used to generate phrase vectors
gen-vecs: model-name nq-rc-data
	python generate_phrase_vecs.py \
		--pretrained_name_or_path SpanBERT/spanbert-base-cased \
		--cache_dir $(CACHE_DIR) \
		--test_file $(DATA_DIR)/single-qa/$(DEV_DATA) \
		--do_dump \
		--max_seq_length 512 \
		--doc_stride 462 \
		--fp16 \
		--filter_threshold -2.0 \
		--append_title \
		--load_dir $(SAVE_DIR)/$(MODEL_NAME) \
		--output_dir $(SAVE_DIR)/$(MODEL_NAME) \
		$(OPTIONS)

# 3) Build an IVFOPQ index for generated phrase vectors
index-vecs: dump-dir large-index
	python build_phrase_index.py \
		--dump_dir $(DUMP_DIR) \
		--stage all \
		--replace \
		--num_clusters $(NUM_CLUSTERS) \
		--fine_quant $(INDEX_TYPE) \
		--cuda

# 4) Compress metadata
compress-meta:
	python scripts/preprocess/compress_metadata.py \
		--input_dump_dir $(DUMP_DIR)/phrase \
		--output_dir $(DUMP_DIR)


# --pretrained_name_or_path SpanBERT/spanbert-base-cased \
# 5) Evaluate the phrase index for phrase retrieval
eval-index: dump-dir model-name large-index orquac-open-data
	python eval_phrase_retrieval.py \
		--run_mode eval \
		--cuda \
		--cache_dir $(CACHE_DIR) \
		--dump_dir $(DUMP_DIR) \
		--index_name start/$(NUM_CLUSTERS)_flat_$(INDEX_TYPE) \
		--load_dir $(SAVE_DIR)/$(MODEL_NAME) \
		--test_path $(DATA_DIR)/$(TEST_DATA) \
		--save_pred \
		--aggregate \
		$(OPTIONS)




# Wrapper for index => compress => eval
index-compress-eval: model-name nq-rc-data medium1-index
	make index-vecs \
		DUMP_DIR=$(DUMP_DIR) \
		NUM_CLUSTERS=$(NUM_CLUSTERS) INDEX_TYPE=$(INDEX_TYPE)
	make compress-meta \
		DUMP_DIR=$(DUMP_DIR)
	make eval-index \
		DUMP_DIR=$(DUMP_DIR) \
		NUM_CLUSTERS=$(NUM_CLUSTERS) INDEX_TYPE=$(INDEX_TYPE) \
		MODEL_LANE=$(MODEL_NAME) TEST_DATA=$(SOD_DATA) \
		OPTIONS=$(OPTIONS)


# TODO
# Single-passage training + additional negatives for NQ
# Available datasets: NQ (nq-rc-data), SQuAD (sqd-rc-data), NQ+SQuAD (nqsqd-rc-data)
# Should change hyperparams (e.g., nq-param) accordingly
run-rc-orquac-hisContra: model-name orquac-rc-hisContra-data orquac-hisContra-param pbn-param
	make train-rc-orquac-hisContra \
		TRAIN_DATA=$(TRAIN_DATA) DEV_DATA=$(DEV_DATA) \
		TEACHER_NAME=$(TEACHER_NAME) MODEL_NAME=$(MODEL_NAME)/tmp \
		BS=$(BS) LR=$(LR) MAX_SEQ_LEN=$(MAX_SEQ_LEN) \
		LAMBDA_KL=$(LAMBDA_KL) LAMBDA_NEG=$(LAMBDA_NEG) LAMBDA_HIS_NEG=$(LAMBDA_HIS_NEG) \
		OPTIONS='$(OPTIONS)'
	CUDA_VISIBLE_DEVICES=1 make train-rc-orquac-hisContra \
		TRAIN_DATA=$(TRAIN_DATA) DEV_DATA=$(DEV_DATA) \
		TEACHER_NAME=$(TEACHER_NAME) MODEL_NAME=$(MODEL_NAME) \
		BS=$(BS) LR=$(LR) MAX_SEQ_LEN=$(MAX_SEQ_LEN) \
		LAMBDA_KL=$(LAMBDA_KL) LAMBDA_NEG=$(LAMBDA_NEG) LAMBDA_HIS_NEG=$(LAMBDA_HIS_NEG) \
		OPTIONS='$(PBN_OPTIONS) $(OPTIONS) --load_dir $(SAVE_DIR)/$(MODEL_NAME)/tmp'



# Testing filter thresholds
filter-test: model-name nq-rc-data nq-param
	python train_rc.py \
		--pretrained_name_or_path SpanBERT/spanbert-base-cased \
		--cache_dir $(CACHE_DIR) \ 
		--validation_file $(DATA_DIR)/single-qa/$(DEV_DATA) \
		--do_eval \
		--do_filter_test \
		--max_seq_length $(MAX_SEQ_LEN) \
		--doc_stride 128 \
		--append_title \
		--filter_threshold_list " -4,-3,-2,-1,-0.5,0,1" \
		--load_dir $(SAVE_DIR)/$(MODEL_NAME) \
		--output_dir $(SAVE_DIR)/$(MODEL_NAME) \
		--overwrite_cache \
		--draft \
		$(OPTIONS)


# Training cross encoder
train-cross-orquac: model-name orquac-rc-hisContra-data
	mkdir -p $(SAVE_DIR)/$(MODEL_NAME)
	python train_cross_encoder.py \
		--model_name_or_path SpanBERT/spanbert-base-cased \
		--train_file $(DATA_DIR)/convqa/$(TRAIN_DATA) \
		--validation_file $(DATA_DIR)/convqa/$(DEV_DATA) \
		--do_train \
		--do_eval \
		--per_device_train_batch_size 24 \
		--learning_rate 3e-5 \
		--num_train_epochs 2 \
		--max_seq_length 384 \
		--doc_stride 128 \
		--save_steps 1000 \
		--output_dir $(SAVE_DIR)/$(MODEL_NAME) \
		--logging_steps=10 \
		--convert_squad_to_hf \
		--overwrite_cache


############################## Large-scale Dump & Indexing ###############################

dump-dir:
ifeq ($(DUMP_DIR),)
	echo "Please set DUMP_DIR before dumping/indexing (e.g., DUMP_DIR=test)"; exit 2;
endif

# Wikipedia dumps (specified as 'data_name') in diffent sizes and their recommended number of clusters for IVF
# - wiki-dev: 1/100 Wikpedia scale (sampled), num_clusters=16384 (medium1-index)
# - wiki-dev-noise: 1/10 Wikipedia scale (sampled), num_clusters=131072 (medium2-index)
# - wiki-20181220: full Wikipedia scale, num_clusters=1048576 (large-index)

# Dump phrase vectors in parallel. Dump will be saved in $(SAVE_DIR)/$(MODEL_NAME)_(data_name)/dump.
gen-vecs-parallel: model-name
	mkdir -p $(SAVE_DIR)/logs/$(MODEL_NAME)
	nohup python scripts/parallel/dump_phrases.py \
		--pretrained_name_or_path SpanBERT/spanbert-base-cased \
		--cache_dir $(CACHE_DIR) \
		--test_dir $(DATA_DIR)/wikidump \
		--dump_name wiki-20181220 \
		--load_dir $(SAVE_DIR)/$(MODEL_NAME) \
		--output_dir $(SAVE_DIR)/$(MODEL_NAME) \
		--filter_threshold 1.0 \
		--append_title \
		--start $(START) \
		--end $(END) \
		--num_procs 6 \
	> $(SAVE_DIR)/logs/$(MODEL_NAME)_$(START)-$(END).log &



# Parallel add for large-scale on-disk IVFSQ (start, end = file idx)
index-add: dump-dir large-index-sq
	export MKL_SERVICE_FORCE_INTEL=1
	python scripts/parallel/add_to_index.py \
		--dump_dir $(DUMP_DIR) \
		--num_clusters $(NUM_CLUSTERS) \
		--cuda \
		--start $(START) \
		--end $(END)

# Merge for large-scale on-disk IVFSQ
index-merge: dump-dir large-index-sq
	python build_phrase_index.py \
		--dump_dir $(DUMP_DIR) \
		--stage merge \
		--replace \
		--num_clusters $(NUM_CLUSTERS) \
		--fine_quant $(INDEX_TYPE)

############################## Open-domain Search & Query-side Fine-tuning ###################################

# TODO
orquac-open-data:
	$(eval TRAIN_DATA=convqa/orquac/preprocessed/qa/train_answer_6.json)
	$(eval TEST_DATA=convqa/orquac/preprocessed/qa/test_answer_6.json)
	$(eval OPTIONS=--truecase)


# Query-side fine-tuning
train-query-orquac: dump-dir model-name orquac-open-data large-index load-dir
	mkdir -p $(SAVE_DIR)/$(MODEL_NAME)
	python train_query.py \
		--run_mode train_query \
		--cache_dir $(CACHE_DIR) \
		--train_path $(DATA_DIR)/$(TRAIN_DATA) \
		--dev_path $(DATA_DIR)/$(DEV_DATA) \
		--test_path $(DATA_DIR)/$(TEST_DATA) \
		--per_device_train_batch_size 12 \
		--eval_batch_size 12 \
		--learning_rate 3e-5 \
		--num_train_epochs 3 \
		--dump_dir $(DUMP_DIR) \
		--index_name start/$(NUM_CLUSTERS)_flat_$(INDEX_TYPE) \
		--load_dir $(LOAD_DIR) \
		--output_dir $(SAVE_DIR)/$(MODEL_NAME) \
		--top_k 100 \
		--cuda \
		--save_pred \
		$(OPTIONS)


# Evaluate Query-side fine-tuning 
eval-query-orquac: dump-dir model-name orquac-open-data large-index load-dir
	mkdir -p $(SAVE_DIR)/$(MODEL_NAME)
	python train_query.py \
		--run_mode eval_query \
		--cache_dir $(CACHE_DIR) \
		--train_path $(DATA_DIR)/$(TRAIN_DATA) \
		--dev_path $(DATA_DIR)/$(DEV_DATA) \
		--test_path $(DATA_DIR)/$(TEST_DATA) \
		--per_device_train_batch_size 12 \
		--eval_batch_size 12 \
		--learning_rate 3e-5 \
		--num_train_epochs 1 \
		--dump_dir $(DUMP_DIR) \
		--index_name start/$(NUM_CLUSTERS)_flat_$(INDEX_TYPE) \
		--load_dir $(LOAD_DIR) \
		--output_dir $(SAVE_DIR)/$(MODEL_NAME) \
		--top_k 50 \
		--cuda \
		--save_pred \
		$(OPTIONS)



############################## Passage-level evaluation ###################################

# agg_strat=opt2 means passage retrieval
eval-index-psg: dump-dir model-name large-index nq-open-data
	python eval_phrase_retrieval.py \
		--run_mode eval \
		--pretrained_name_or_path SpanBERT/spanbert-base-cased \
		--cuda \
		--dump_dir $(DUMP_DIR) \
		--index_name start/$(NUM_CLUSTERS)_flat_$(INDEX_TYPE) \
		--load_dir $(SAVE_DIR)/$(MODEL_NAME) \
		--test_path $(DATA_DIR)/$(TEST_DATA) \
		--save_pred \
		--aggregate \
		--agg_strat opt2 \
		--top_k 200 \
		--eval_psg \
		--psg_top_k 100 \
		$(OPTIONS)

# transform prediction for the recall evaluation (when you already have prediction files)
recall-eval: model-name
	python scripts/postprocess/recall_transform.py \
		--model_dir $(SAVE_DIR)/$(MODEL_NAME) \
		--pred_file test_answer_6_4406_top10.pred \
		--psg_top_k 10
	python scripts/postprocess/recall.py \
		--k_values 1,3,5,10 \
		--results_file $(SAVE_DIR)/$(MODEL_NAME)/pred/test_answer_6_4406_top10_psg-top10.json \
		--ans_fn string


############################## Data Pre/Post-processing ###################################

preprocess-openqa:
	python scripts/preprocess/create_openqa.py \
		$(FS)/fid-data/download/NQ-open.train.jsonl \
		$(DATA_DIR)/open-qa/nq-new \
		--input_type jsonl

# Convert SQuAD format into HF-style json format
preprocess-rc: multi-rc-data
	python scripts/preprocess/convert_squad_to_hf.py \
		$(DATA_DIR)/single-qa/$(TRAIN_DATA)

# Warning: many scripts below are not documented well.
# Each script may rely on external resources (e.g., original NQ datasets).
data-config:
	$(eval NQORIG_DIR=$(DATA_DIR)/natural-questions)
	$(eval NQOPEN_DIR=$(DATA_DIR)/nq-open)
	$(eval NQREADER_DIR=$(DATA_DIR)/single-qa/nq)
	$(eval SQUAD_DIR=$(DATA_DIR)/single-qa/squad)
	$(eval SQUADREADER_DOC_DIR=$(DATA_DIR)/squad-reader-docs)
	$(eval NQREADER_DOC_DIR=$(DATA_DIR)/nq-reader-docs)
	$(eval WIKI_DIR=$(DATA_DIR)/wikidump)

nq-reader-to-wiki:
	python scripts/preprocess/create_nq_reader_wiki.py \
		$(DATA_DIR)/single-qa/nq/train.json,$(DATA_DIR)/single-qa/nq/dev.json \
		$(DATA_DIR)/single-qa/nq \
		$(DATA_DIR)/wikidump/20181220_concat/

download-wiki: data-config
	python scripts/preprocess/download_wikidump.py \
		--output_dir $(WIKI_DIR)

nq-reader-train: data-config
	python scripts/preprocess/create_nq_reader.py \
		--nq_orig_path_pattern "$(NQORIG_DIR)/train/nq-train-*.jsonl.gz" \
		--nq_open_path $(NQOPEN_DIR)/train.json \
		--output_path $(NQREADER_DIR)/train_79168.json

nq-reader-dev: data-config
	python scripts/preprocess/create_nq_reader.py \
		--nq_orig_path_pattern "$(NQORIG_DIR)/train/nq-train-*.jsonl.gz" \
		--nq_open_path $(NQOPEN_DIR)/dev.json \
		--output_path $(NQREADER_DIR)/dev_8757.json

nq-reader-dev-sample: data-config
	python scripts/preprocess/create_nq_reader.py \
		--nq_orig_path_pattern "$(NQORIG_DIR)/sample/nq-train-sample.jsonl.gz" \
		--nq_open_path $(NQOPEN_DIR)/dev.json \
		--output_path $(NQREADER_DIR)/dev_sample.json

nq-reader-train-docs: data-config
	python scripts/preprocess/create_nq_reader_doc.py \
		--nq_orig_path_pattern "$(NQORIG_DIR)/train/nq-train-*.jsonl.gz" \
		--nq_open_path $(NQOPEN_DIR)/train.json \
		--output_dir $(NQREADER_DOC_DIR)/train

nq-reader-dev-docs: data-config
	python scripts/preprocess/create_nq_reader_doc.py \
		--nq_orig_path_pattern "$(NQORIG_DIR)/train/nq-train-*.jsonl.gz" \
		--nq_open_path $(NQOPEN_DIR)/dev.json \
		--output_dir $(NQREADER_DOC_DIR)/dev

nq-reader-dev-docs-sample: data-config
	python scripts/preprocess/create_nq_reader_doc.py \
		--nq_orig_path_pattern "$(NQORIG_DIR)/sample/nq-train-sample.jsonl.gz" \
		--nq_open_path $(NQOPEN_DIR)/dev.json \
		--output_dir $(NQREADER_DOC_DIR)-sample

nq-_reader-train-docs-wiki: data-config
	python scripts/preprocess/create_nq_reader_doc_wiki.py \
		--wiki_dir $(WIKI_DIR)/20181220_concat \
		--nq_reader_docs_dir $(NQREADER_DOC_DIR)/train \
		--output_dir $(NQREADER_DOC_DIR)/train_wiki

nq-reader-dev-docs-wiki: data-config
	python scripts/preprocess/create_nq_reader_doc_wiki.py \
		--wiki_dir $(WIKI_DIR)/20181220_concat \
		--nq_reader_docs_dir $(NQREADER_DOC_DIR)/dev \
		--output_dir $(NQREADER_DOC_DIR)/dev_wiki

squad-reader-train-docs-wiki: data-config
	python scripts/preprocess/create_nq_reader_doc_wiki.py \
		--wiki_dir $(WIKI_DIR)/20181220_concat \
		--nq_reader_docs_dir $(SQUAD_DIR)/train-v1.1.json \
		--output_dir $(SQUADREADER_DOC_DIR)/train_wiki \
		--is_squad

squad-reader-dev-docs-wiki: data-config
	python scripts/preprocess/create_nq_reader_doc_wiki.py \
		--wiki_dir $(WIKI_DIR)/20181220_concat \
		--nq_reader_docs_dir $(SQUAD_DIR)/dev-v1.1.json \
		--output_dir $(SQUADREADER_DOC_DIR)/dev_wiki \
		--is_squad

build-db: data-config
	python scripts/preprocess/build_db.py \
		--data_path $(WIKI_DIR)/extracted \
		--save_path $(WIKI_DIR)/docs_20181220_nolist.db \
		--preprocess scripts/preprocess/prep_wikipedia.py \
		--overwrite

build-wikisquad: data-config
	python scripts/preprocess/build_wikisquad.py \
		--db_path $(WIKI_DIR)/docs_20181220_nolist.db \
		--out_dir $(WIKI_DIR)/20181220_nolist

concat-wikisquad: data-config
	python scripts/preprocess/concat_wikisquad.py \
		--input_dir $(WIKI_DIR)/20181220_nolist \
		--output_dir $(WIKI_DIR)/20181220_nolist_concat

first-para-wikisquad: data-config
	python scripts/preprocess/first_para_wikisquad.py \
		--input_dir $(WIKI_DIR)/20181220_nolist \
		--output_dir $(WIKI_DIR)/20181220_nolist_first

compare-db: data-config
	python scripts/preprocess/compare_db.py \
		--db1 $(DATA_DIR)/denspi/docs.db \
		--db2 $(WIKI_DIR)/docs_20181220.db
		
sample-nq-reader-doc-wiki-train: data-config
	python scripts/preprocess/sample_nq_reader_doc_wiki.py \
		--sampling_ratio 0.15 \
		--wiki_dir $(WIKI_DIR)/20181220_concat \
		--docs_wiki_dir $(NQREADER_DOC_DIR)/train_wiki \
		--output_dir $(NQREADER_DOC_DIR)/train_wiki_noise

sample-nq-reader-doc-wiki-dev: data-config
	python scripts/preprocess/sample_nq_reader_doc_wiki.py \
		--sampling_ratio 0.1 \
		--wiki_dir $(WIKI_DIR)/20181220_concat \
		--docs_wiki_dir $(NQREADER_DOC_DIR)/dev_wiki \
		--output_dir $(NQREADER_DOC_DIR)/dev_wiki_noise
