# Semi-supervised learning via EBM

Codes for reproducing experiments  in 
* Yunfu Song, Zhijian Ou, Zitao Liu, and Songfan Yang. Upgrading CRFs to JRFs and its benefits to sequence modeling and labeling. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2020.
*  Yunfu Song, Huahuan Zheng, Zhijian Ou. An empirical comparison of joint-training and pre-training for domain-agnostic semi-supervised learning via energy-based models. IEEE International Workshop on Machine Learning for Signal Processing (MLSP), 2021.

# Prerequistes

- Python 3.6
- TensorFlow 1.8

# For ICASSP-2020 paper

* Processing data

  Place source data of PTB datasets to data/raw/pos, CoNLL-2003 English NER datasets to data/raw/ner, CoNLL-2000 chunking datasets to data/raw/chunking, Google-one-billion-word datasets to data/raw/1-billion-word-language-modeling-benchmark-r13output, Glove word embedding to data/raw/glove.

  The nbest lists for language model rescoring experiments are already provided at data/raw/WSJ92-test-data.

  ~~~
  sh scripts/process_data.sh
  ~~~

  Then all datasets are processing, and placed at data/processed.

* Language model rescoring experiments
  ~~~
  sh scripts/run_rescoring_experiments.sh [seed]
  ~~~

  "seed" is the random seed of experiments, e.g. 1.
  
  To reproduce the results of Table 1 in paper for both TRF and JRF.
  
* Sequence labeling experiments.
  
    ~~~
    sh scripts/run_sequence_labeling_experiments.sh [task] [part] [seed]
    ~~~
  
    "task" in one of ['pos', 'ner', 'chunk']; "part" represents the amounts of labeled data, 1 means all labeled data available and 10 means 10% of labeled data available.
  
    To reproduce the results of Table 2 in paper for CRF, self-training and JRF.

# For MLSP-2021 paper

#### Semi-Supervised Learning (SSL) for Image Classification

We conduct semi-supervised classification experiment over CIFAR-10.
- Joint-training EBM: refer to the **theano_SSL** folder in repository https://github.com/thu-spmi/Inclusive-NRF.
- Pre-training+fine-tuning EBM: refer to the **CIFAR_pretrain_finetune** folder.

#### SSL for Natural Language Labeling

In this experiment, we evaluate different SSL methods for natural language labeling, through three tasks (POS tagging, chunking and NER).

- For Joint-training EBM based SSL and Pre-training EBM based SSL, run
```
sh scripts/run_trf_finetune.sh [task] [lab] [unl] [seed]
```
Parameters:
	- "task" could be pos, ner, chunk; 
	- "lab" represents label proportion w.r.t. the full set of labels. 1 denotes 100%, 10 denotes 10%, 50 denotes 2%.
	- "unl" represents the ratio between the amount of unlabeled and labeled data, which can take 50, 250, 500.

- For running biLSTMLM pretrain+finetune
```
sh scripts/run_bilm_finetune.sh [task] [lab] [unl] [seed]
```
Parameters "task", "lab", "unl", "seed": the same meanings as above.