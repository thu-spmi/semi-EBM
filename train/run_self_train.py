# Copyright 2020 Tsinghua University, Author: Yunfu Song
# Apache 2.0.
# This script contrains self-training experiments.

import tensorflow as tf
import numpy as np
import json
import os

from base import *
import crf

import argparse

paser = argparse.ArgumentParser()
paser.add_argument('--alpha', default=1, type=float)
paser.add_argument('--cw', default=1, type=float)
paser.add_argument('--tw', default=0, type=float)
paser.add_argument('--data', default='one-ten', type=str)
paser.add_argument('--task', default='pos', type=str)
paser.add_argument('--sf', default='', type=str)
paser.add_argument('--model', default='', type=str)
paser.add_argument('--mode', default='train', type=str)
paser.add_argument('--nbest', default=0, type=int)
paser.add_argument('--bs', default=64, type=int)
paser.add_argument('--nt', default=1, type=int)
paser.add_argument('--data_type', default=1, type=int)
paser.add_argument('--lr', default=0.1, type=float)
paser.add_argument('--lrd', default=0.03, type=float)
paser.add_argument('--opt', default='momentum', type=str)
paser.add_argument('--word_emb', action='store_false', default=True)
paser.add_argument('--std', default=1, type=int)
paser.add_argument('--seed', default=1, type=int)
paser.add_argument('--lab', default=1, type=int)
paser.add_argument('--unl', default=50, type=int)
args = paser.parse_args()
print(args)

tf.set_random_seed(args.seed)
np.random.seed(args.seed)
import random
random.seed(args.seed)

def main():
    with open('data/processed/%s/data.info'%args.task) as f:
        data_info = json.load(f)

    task2all = {'pos': 56554, 'ner': 14041, 'chunk': 7436}
    train_num = task2all[args.task] // args.lab
    data = seq.Data(vocab_files=data_info['vocab'],
                    train_list=data_info['train%d.part' % train_num],
                    valid_list=data_info['valid'],
                    test_list=data_info['test'],
                    max_len=60
                    )

    data_full = seq.Data(vocab_files=data_info['vocab'],
                         train_list=data_info['train%d'%train_num],
                         valid_list=data_info['valid'],
                         test_list=data_info['test']
                         )

    config = crf.Config(data_full)
    config.mix_config.std = 1

    config.mix_config.c2w_type = 'cnn'
    config.mix_config.chr_embedding_size = 50
    config.mix_config.c2w_cnn_size = 100
    config.mix_config.c2w_cnn_width = [2, 3, 4]

    config.mix_config.rnn_hidden_size = 512
    config.mix_config.embedding_size = 300
    config.mix_config.rnn_proj_size = 512

    config.mix_config.opt_method = args.opt
    config.mix_config.dropout = 0.5

    config.train_batch_size = 64
    config.mix_config.embedding_init_npy = data_info['word_emb_d300']

    config.eval_every = 500
    config.warm_up_steps = 100
    config.lr_decay = 0.005
    config.lr = args.lr

    config.max_steps = 20000


    logdir = 'models/%s_selftrain_lab%d_unl%d_%s/'%(args.task,args.lab,args.unl,args.sf)
    logdir = wb.mklogdir(logdir,is_recreate=True)
    config.print()

    m = crf.self_CRF(config, data, data_full, logdir, device='/gpu:0',uns_weight=args.alpha)

    if args.model:
        # variables = tf.contrib.slim.get_variables_to_restore()
        # print('----------------------')
        # print(len(variables))
        # print('----------------------')
        #
        # # variables_to_restore = [v for v in variables if 'edge_mat' not in v.name and 'final_linear' not in v.name]
        # variables_to_restore = [v for v in variables if 'edge_mat' not in v.name and 'final_linear' not in v.name and v.trainable==True]
        # print('----------------------')
        # print(len(variables_to_restore))
        # print('----------------------')

        saver = tf.train.Saver()
        model_path = tf.train.latest_checkpoint('models/'+args.model+'/check_models/')


    sv = tf.train.Supervisor(logdir=os.path.join(logdir, 'logs'))
    sv.summary_writer.add_graph(tf.get_default_graph())  # write the graph to logs
    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    with sv.managed_session(config=session_config) as session:
        with session.as_default():
            m.initialize()

            if args.model:
                saver.restore(session, model_path)

            ops = crf.DefaultOps(args, m, data_full.datas[-2], data_full.datas[-1], id2tag_name=data_info['id2tag'])

            m.train(0.1, ops)
            m.best_saver.restore(session, tf.train.latest_checkpoint(m.best_save_name))
            ops = crf.DefaultOps(args, m, data_full.datas[-2], data_full.datas[-1], id2tag_name=data_info['id2tag'], test=True)
            print('------------begin test-----------')
            ops.perform(0, 0)


if __name__ == '__main__':

    main()