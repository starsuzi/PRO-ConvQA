# Phrase Retrieval for Open-Domain Conversational Question Answering with Conversational Dependency Modeling via Contrastive Learning

Official Code Repository for the paper "Phrase Retrieval for Open-Domain Conversational Question Answering with Conversational Dependency Modeling via Contrastive Learning" (Findings of ACL 2023): https://aclanthology.org/2023.findings-acl.374.pdf

## Abstract

<div align="center">
  <img alt="PRO-ConvQA Overview" src="./images/pro-convqa.png" width="400px">
</div>

Open-Domain Conversational Question Answering (ODConvQA) aims at answering questions through a multi-turn conversation based on a retriever-reader pipeline, which retrieves passages and then predicts answers with them. However, such a pipeline approach not only makes the reader vulnerable to the errors propagated from the retriever, but also demands additional effort to develop both the retriever and the reader, which further makes it slower since they are not runnable in parallel. In this work, we propose a method to directly predict answers with a phrase retrieval scheme for a sequence of words, reducing the conventional two distinct subtasks into a single one. Also, for the first time, we study its capability for ODConvQA tasks. However, simply adopting it is largely problematic, due to the dependencies between previous and current turns in a conversation. To address this problem, we further introduce a novel contrastive learning strategy, making sure to reflect previous turns when retrieving the phrase for the current context, by maximizing representational similarities of consecutive turns in a conversation while minimizing irrelevant conversational contexts. We validate our model on two ODConvQA datasets, whose experimental results show that it substantially outperforms the relevant baselines with the retriever-reader.

## Installation
We refer to the following repository: https://github.com/princeton-nlp/DensePhrases.

```bash
$ conda create -n proconvqa python=3.8
$ conda activate proconvqa
$ pip3 install torch==1.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
$ pip install -r requirements.txt
```
## Dataset
We first set the default directories as follows:
```bash
source config.sh
```
We download wikidump as follows:
```bash
source download.sh
```
We download OR-QuAC dataset from https://ciir.cs.umass.edu/downloads/ORConvQA/.
Then, we preprocess OR-QuAC as follows:
```bash
python scripts/preprocess/data/orquac/retriever/convert_orconv_retriever_train.py
python scripts/preprocess/data/orquac/retriever/convert_orconv_retriever_dev.py
python scripts/preprocess/data/orquac/retriever/convert_orconv_retriever_test.py

python scripts/preprocess/data/orquac/qa/convert_qa_train.py
python scripts/preprocess/data/orquac/qa/convert_qa_test.py
```

## Run
We start training a cross encoder.
```bash
bash scripts/run/orquac/1_train_cross.sh
```
Then, we train a phrase retriever.
```bash
bash scripts/run/orquac/2_train.sh
```

We generate vectors of the Wikipedia.
Here, we generate vectors in parallel.
```bash
bash scripts/run/orquac/3_gen_vec.sh
```

Now, we create a phrase index.
```bash
bash scripts/run/orquac/4_index.sh
bash scripts/run/orquac/5_compress.sh
```

Lastly, we further finetune and evaluate a query-side encoder.
```bash
bash scripts/run/orquac/6_query_finetune.sh
bash scripts/run/orquac/7_query_finetune_eval.sh
```

## Citation
If you found the provided code with our paper useful, we kindly request that you cite our work.
```BibTex
@inproceedings{jeong-etal-2023-phrase,
    title = "Phrase Retrieval for Open Domain Conversational Question Answering with Conversational Dependency Modeling via Contrastive Learning",
    author = "Jeong, Soyeong  and
      Baek, Jinheon  and
      Hwang, Sung Ju  and
      Park, Jong",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-acl.374",
    pages = "6019--6031",
}
```
