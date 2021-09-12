# Copyright 2020 Tsinghua University, Author: Yunfu Song
# Apache 2.0.
# This script contrains TRF semi-supervised (JRF) training experiments.

import tensorflow as tf
import numpy as np
import json
import os

from base import *
import trf_semi

import argparse

paser = argparse.ArgumentParser()
paser.add_argument('--alpha', default=0.1, type=float)
paser.add_argument('--cw', default=1, type=float)
paser.add_argument('--tw', default=0, type=float)
paser.add_argument('--data', default='one-ten', type=str)
paser.add_argument('--task', default='pos', type=str)
paser.add_argument('--sf', default='', type=str)
paser.add_argument('--model', default='', type=str)
paser.add_argument('--mode', default='train', type=str)
paser.add_argument('--bs', default=32, type=int)
paser.add_argument('--lr', default = 0.1, type=float)
paser.add_argument('--do', default=0.5, type=float)
paser.add_argument('--unl', default=50, type=int)
paser.add_argument('--lab', default=1, type=int)
paser.add_argument('--lrd', default=0.03, type=float)
paser.add_argument('--nbest', default=0, type=int)
paser.add_argument('--opt', default='momentum', type=str)
paser.add_argument('--std', default=1, type=int)
paser.add_argument('--word_emb', action='store_false', default=True)
paser.add_argument('--steps', default=0, type=int)
paser.add_argument('--seed', default=1, type=int)
paser.add_argument('--sgd', default=0, type=int)
paser.add_argument('--maxnorm', default=10, type=float)
paser.add_argument('--maxstep', default=0, type=int)
paser.add_argument('--hl', default=1, type=int)
# 'train1000/trf_noise1.0_blstm_cnn_we100_ce30_c2wcnn_dropout0.5_adampretrain_baseline/crf_models/TRF-uns_ep11.ckpt'
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
        nbest = [
            "data/raw/WSJ92-test-data/1000best.sent",
            "data/raw/WSJ92-test-data/transcript.txt",
            "data/raw/WSJ92-test-data/1000best.acscore",
            "data/raw/WSJ92-test-data/1000best.lmscore"
        ]
    task2all = {'pos': 56554, 'ner': 14041, 'chunk': 7436}
    train_num = task2all[args.task] // args.lab

    if args.nbest:
        data = seq.Data(vocab_files=data_info['vocab'],
                        train_list=data_info['train%d' % train_num],
                        valid_list=data_info['valid'],
                        test_list=data_info['test'],
                        )
    else:

        data = seq.Data(vocab_files=data_info['vocab'],
                        train_list=data_info['train%d.part%d'%(train_num,args.unl)],
                        valid_list=data_info['valid'],
                        test_list=data_info['test'],
                        max_len=60
                        )

    data_full = seq.Data(vocab_files=data_info['vocab'],
                         train_list=data_info['train%d'%train_num],
                         valid_list=data_info['valid'],
                         test_list=data_info['test']
                         )

    config = trf_semi.Config(data)
    config.nbest=args.nbest
    config.mix_config.std = args.std
    config.mix_config.c2w_type = 'cnn'
    config.mix_config.chr_embedding_size = 50
    config.mix_config.c2w_cnn_size = 100
    config.mix_config.c2w_cnn_width = [2, 3, 4]

    config.mix_config.rnn_hidden_size = 512
    config.mix_config.rnn_hidden_layers = args.hl
    config.mix_config.embedding_size = 300
    config.mix_config.rnn_proj_size = 512

    config.mix_config.opt_method = args.opt
    config.sampler_config.optimize_method =args.opt
    config.sampler_config.learning_rate = args.lr
    if args.sgd:
        config.sampler_config.optimize_method = 'SGD'
    config.mix_config.dropout = args.do
    config.mix_config.trf_dropout = args.do
    config.mix_config.crf_weight = args.cw
    config.mix_config.trf_weight = args.tw
    config.crf_batch_size = 64
    config.trf_batch_size = args.bs
    config.data_factor = 1
    config.mix_config.inter_alpha = args.alpha
    if args.word_emb:
        config.mix_config.embedding_init_npy = data_info['word_emb_d300']

    if args.maxstep>0:
        config.max_steps=args.maxstep
    elif args.lab==1:
        config.max_steps= 40000
    else:
        config.max_steps= 20000

    if args.nbest:
        config.eval_every = 1000
    else:
        config.eval_every = 500
    config.warm_up_steps=100
    config.lr_decay = 0.005
    config.lr=args.lr

    logdir = 'models/%s_trf_semi_lab%dunl%d_%s/' % (args.task, args.lab, args.unl, args.sf)
    logdir = wb.mklogdir(logdir,is_recreate=True)
    config.print()

    m = trf_semi.TRF(config, data, data_full, logdir, device='/gpu:0')

    if args.model:
        variables = tf.contrib.slim.get_variables_to_restore()
        print('----------------------')
        print(len(variables))
        print('----------------------')

        if 'bilm' in args.model:
            variables_to_restore = [v for v in variables if 'edge_mat' not in v.name and 'final_linear' not in v.name and 'auxiliary_lstm' not in v.name and 'logz' not in v.name and v.trainable == True]
        elif 'pretrain' in args.model:
            variables_to_restore = [v for v in variables if 'edge_mat' not in v.name and 'final_linear' not in v.name and 'auxiliary_lstm' not in v.name and v.trainable == True]
        else:
            variables_to_restore = [v for v in variables if 'auxiliary_lstm' not in v.name and 'logz' not in v.name and v.trainable == True]

        model_path = tf.train.latest_checkpoint('models/' + args.model + '/check_models')
        print('----------------------')
        print(len(variables_to_restore))
        for v in variables_to_restore:
            print(v.name)
        print('----------------------')

        saver = tf.train.Saver(variables_to_restore)

    sv = tf.train.Supervisor(logdir=os.path.join(logdir, 'logs'))
    sv.summary_writer.add_graph(tf.get_default_graph())  # write the graph to logs
    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    # session_config.gpu_options.allow_growth = True
    with sv.managed_session(config=session_config) as session:
        with session.as_default():
            m.initialize()
            if args.model:
                saver.restore(session, model_path)

            ops = trf_semi.DefaultOps(args, m, data.datas[-2], data.datas[-1], id2tag_name=data_info['id2tag'],nbest_files=nbest)
            m.train(0.1, ops,args.steps)
            m.best_saver.restore(session, tf.train.latest_checkpoint(m.best_save_name))
            ops = trf_semi.DefaultOps(args, m, data.datas[-2], data.datas[-1], id2tag_name=data_info['id2tag'], test=True,nbest_files=nbest)
            print('------------begin test-----------')
            ops.perform(0, 0)


if __name__ == '__main__':

    main()