# Copyright 2020 Tsinghua University, Author: Yunfu Song
# Apache 2.0.
# This script contrains TRF unsupervised training experiments.

import tensorflow as tf
import numpy as np
import json
import os

from base import *
import trf_uns, nce_net_uns
import argparse

paser = argparse.ArgumentParser()
paser.add_argument('--alpha', default=1, type=float)
paser.add_argument('--sf', default='', type=str)
paser.add_argument('--net', default='Net', type=str)
paser.add_argument('--model', default='', type=str)
paser.add_argument('--nbest', default=0, type=int)
paser.add_argument('--bs', default=32, type=int)
paser.add_argument('--df', default=1, type=float)
paser.add_argument('--nf', default=1, type=float)
paser.add_argument('--do', default=0.5, type=float)
paser.add_argument('--nt', default=1, type=int)
paser.add_argument('--unl', default=50, type=int)
paser.add_argument('--lab', default=1, type=int)
paser.add_argument('--data', default='all', type=str)
paser.add_argument('--lr', default=1e-3, type=float)
paser.add_argument('--lrd', default=0.03, type=float)
paser.add_argument('--opt', default='adam', type=str)
paser.add_argument('--word_emb', action='store_false', default=True)
paser.add_argument('--std', default=1, type=int)
# paser.add_argument('--data', default='one-ten', type=str)
paser.add_argument('--task', default='pos', type=str)
paser.add_argument('--seed', default=1, type=int)
paser.add_argument('--hl', default=1, type=int)
args = paser.parse_args()
print(args)

tf.set_random_seed(args.seed)
np.random.seed(args.seed)
import random
random.seed(args.seed)

def main():
    with open('data/processed/%s/data.info'%args.task) as f:
        data_info = json.load(f)
    nbest=None
    if args.nbest:
        nbest=[
            "data/raw/WSJ92-test-data/1000best.sent",
            "data/raw/WSJ92-test-data/transcript.txt",
            "data/raw/WSJ92-test-data/1000best.acscore",
            "data/raw/WSJ92-test-data/1000best.lmscore"
        ]
    task2all = {'pos': 56554, 'ner': 14041, 'chunk': 7436}


    train_num = task2all[args.task]//args.lab

    if args.nbest:
        assert args.task=='pos'
        data = seq.Data(vocab_files=data_info['vocab'],
                        train_list=data_info['train%d'%train_num],
                        valid_list=data_info['valid'],
                        test_list=data_info['test']
                        )
    else:
      data = seq.Data(vocab_files=data_info['vocab'],
                        train_list=data_info['train%d.part%d'%(train_num,args.unl)],
                        valid_list=data_info['valid'],
                        test_list=data_info['test'],
                        max_len=60
                        )

    config = trf_uns.Config(data)

    # config.mix_config.c2w_type = 'cnn'
    # config.mix_config.chr_embedding_size = 30
    # config.mix_config.c2w_cnn_size = 30

    config.mix_config.c2w_type = 'cnn'
    config.mix_config.chr_embedding_size = 50
    config.mix_config.c2w_cnn_size = 100
    config.mix_config.c2w_cnn_width = [2, 3, 4]

    config.mix_config.rnn_hidden_size = 512
    config.mix_config.rnn_hidden_layers = args.hl
    config.mix_config.embedding_size = 300
    config.mix_config.rnn_proj_size = 512
    config.mix_config.opt_method = args.opt
    config.sampler_config.optimize_method = args.opt
    config.sampler_config.learning_rate = args.lr
    config.mix_config.dropout = args.do
    config.max_epoch = 5
    config.crf_batch_size = 64
    config.trf_batch_size = args.bs
    config.data_factor = args.df
    config.noise_factor = args.nf
    config.lr = args.lr

    config.eval_every = 0.1
    config.warm_up_steps=100
    config.lr_decay = 0.005

    # config.lr = lr.LearningRateEpochDelay2(1e-3, delay=0.02)

    if args.word_emb:
        config.mix_config.embedding_init_npy=data_info['word_emb_d300']

    logdir = 'models/%s_trf_pretrain_lab%dunl%d_%s/' % (args.task,args.lab,args.unl, args.sf)
    logdir = wb.mklogdir(logdir, is_recreate=True)
    config.print()
    wb.mkdir(os.path.join(logdir, 'crf_models'))

    m = trf_uns.TRF(config, data, logdir, device='/gpu:1',net_name=args.net)


    sv = tf.train.Supervisor(logdir=os.path.join(logdir, 'logs'))
    sv.summary_writer.add_graph(tf.get_default_graph())  # write the graph to logs
    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    with sv.managed_session(config=session_config) as session:
        with session.as_default():
            m.initialize()
            if args.nbest:
                ops = trf_uns.DefaultOps(m, nbest)
            else:
                ops=None
            m.train(0.1,ops)



if __name__ == '__main__':

    main()