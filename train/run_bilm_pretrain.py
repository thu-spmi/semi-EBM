# Copyright 2020 Tsinghua University, Author: Yunfu Song
# Apache 2.0.
# This script contrains biLSTMLM unsupervised training experiments.

import tensorflow as tf
import numpy as np
import json
import os

from base import *
import bilm_uns

import argparse

paser = argparse.ArgumentParser()
paser.add_argument('--alpha', default=1, type=float)
paser.add_argument('--sf', default='', type=str)
paser.add_argument('--net', default='Net', type=str)
paser.add_argument('--model', default='', type=str)
paser.add_argument('--nbest', default=0, type=int)
paser.add_argument('--bs', default = 64, type=int)
paser.add_argument('--df', default=1, type=float)
paser.add_argument('--do', default=0.5, type=float)
paser.add_argument('--nt', default=1, type=int)
paser.add_argument('--unl', default=50, type=int)
paser.add_argument('--lab', default=50, type=int)
paser.add_argument('--data', default='all', type=str)
paser.add_argument('--lr', default=0.1, type=float)
paser.add_argument('--lrd', default=0.03, type=float)
paser.add_argument('--opt', default='momentum', type=str)
paser.add_argument('--word_emb', action='store_false', default=True)
paser.add_argument('--std', default=1, type=int)
# paser.add_argument('--data', default='one-ten', type=str)
paser.add_argument('--task', default='pos', type=str)
paser.add_argument('--seed', default=1, type=int)
paser.add_argument('--hl', default=1, type=int)
args = paser.parse_args()
print(args)

def main():
    with open('data/processed/%s/data.info'%args.task) as f:
        data_info = json.load(f)

    task2all = {'pos': 56554, 'ner': 14041, 'chunk': 7436}

    train_num = task2all[args.task] // args.lab

    data = seq.Data(vocab_files=data_info['vocab'],
                    train_list=data_info['train%d.part%d' % (train_num, args.unl)],
                    valid_list=data_info['valid'],
                    test_list=data_info['test'],
                    max_len=60
                    )


    config = bilm_uns.Config(data)
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
    config.mix_config.dropout = args.do
    config.max_epoch = 5
    config.crf_batch_size = 64
    config.trf_batch_size = args.bs
    config.data_factor = args.df

    config.lr = args.lr
    config.eval_every = 0.1
    config.warm_up_steps = 100
    config.lr_decay = 0.005

    if args.word_emb:
        config.mix_config.embedding_init_npy=data_info['word_emb_d300']

    logdir = 'models/%s_bilm_pretrain_lab%dunl%d_%s/' % (args.task, args.lab, args.unl, args.sf)
    logdir = wb.mklogdir(logdir, is_recreate=True)
    config.print()
    wb.mkdir(os.path.join(logdir, 'crf_models'))

    m = bilm_uns.TRF(config, data, logdir, device='/gpu:1')

    sv = tf.train.Supervisor(logdir=os.path.join(logdir, 'logs'))
    sv.summary_writer.add_graph(tf.get_default_graph())  # write the graph to logs
    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    with sv.managed_session(config=session_config) as session:
        with session.as_default():
            # print(m.get_crf_logps(data_full.datas[0][0: 10]))
            # print(m.get_trf_logps(data_full.datas[0][0: 10]))
            # ops = trf.DefaultOps(m, data.datas[-2], data.datas[-1], data_info['nbest'])
            # ops = trf_uns.DefaultOps(m, data.datas[-2], data.datas[-1], None)
            # ops.perform(0, 0)
            m.train(0.1)


if __name__ == '__main__':
    main()