# Copyright 2020 Tsinghua University, Author: Yunfu Song
# Apache 2.0.
# This script contrains supervised training experiments, also could be used with fine-tuning.


import tensorflow as tf
import numpy as np
import json
import os

from base import *
import crf, nce_net


def main():
    with open('data/processed/%s/data.info'%args.task) as f:
        data_info = json.load(f)

    task2all = {'pos': 56554, 'ner': 14041, 'chunk': 7436}
    # data = seq.Data(vocab_files=data_info['vocab'],
    #                 train_list=data_info['train%d.part' % train_num],
    #                 valid_list=data_info['valid'],
    #                 test_list=data_info['test'],
    #                 max_len=60
    #                 )
    train_num = task2all[args.task] // args.lab

    data_full = seq.Data(vocab_files=data_info['vocab'],
                         train_list=data_info['train%d'%train_num],
                         valid_list=data_info['valid'],
                         test_list=data_info['test']
                         )

    config = crf.Config(data_full)
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
    config.mix_config.dropout = 0.5
    config.mix_config.max_grad_norm = args.maxnorm

    config.train_batch_size = 64

    config.lr_mix = lr.LearningRateEpochDelay2(args.lr, delay=args.lrd)
    # config.max_epoch = 100

    if args.word_emb:
        config.mix_config.embedding_init_npy = data_info['word_emb_d300']

    config.eval_every = 500
    config.warm_up_steps = 100
    config.lr_decay = 0.005
    config.lr = args.lr
    if args.maxstep>0:
        config.max_steps=args.maxstep
    elif args.lab==1:
        config.max_steps= 20000
    else:
        config.max_steps= 10000

    logdir = 'models/%s_crf_lab%dunl%d_%s/'%(args.task,args.lab,args.unl,args.sf)
    logdir = wb.mklogdir(logdir, is_recreate=True)
    config.print()

    m = crf.CRF(config, data_full, logdir, device='/gpu:1')
    if args.model:
        variables = tf.contrib.slim.get_variables_to_restore()
        print('----------------------')
        print(len(variables))
        print('----------------------')

        variables_to_restore = [v for v in variables if 'edge_mat' not in v.name and 'final_linear' not in v.name and v.trainable == True]
        print('----------------------')
        print(len(variables_to_restore))
        print('----------------------')

        saver = tf.train.Saver(variables_to_restore)

        model_path = tf.train.latest_checkpoint('models/' + args.model + '/check_models')


    sv = tf.train.Supervisor(logdir=os.path.join(logdir, 'logs'))
    sv.summary_writer.add_graph(tf.get_default_graph())  # write the graph to logs
    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    id2tag_name=data_info['id2tag'] if args.task!='pos' else None
    with sv.managed_session(config=session_config) as session:
        with session.as_default():
            m.initialize()

            if args.model:
                saver.restore(session, model_path)
            # print(m.get_crf_logps(data_full.datas[0][0: 10]))
            # print(m.get_trf_logps(data_full.datas[0][0: 10]))
            # ops = trf.DefaultOps(m, data.datas[-2], data.datas[-1], data_info['nbest'])
            ops = crf.DefaultOps(args,m, data_full.datas[-2], data_full.datas[-1],id2tag_name)
            # ops.perform(0, 0)
            if not args.test:m.train(0.1, ops)
            best_name=tf.train.latest_checkpoint(m.best_save_name)
            m.best_saver.restore(session,best_name)
            ops = crf.DefaultOps(args, m, data_full.datas[-2], data_full.datas[-1],id2tag_name,test=True)
            ops.perform(0,0)


if __name__ == '__main__':
    import argparse
    paser=argparse.ArgumentParser()
    paser.add_argument('--alpha',default=1,type=float)
    paser.add_argument('--data', default='train56554', type=str)
    paser.add_argument('--sf', default='', type=str)
    paser.add_argument('--model', default='', type=str)
    paser.add_argument('--lab', default=1, type=int)
    paser.add_argument('--unl', default=50, type=int)
    paser.add_argument('--task', default='pos', type=str)
    paser.add_argument('--seed', default=1, type=int)
    paser.add_argument('--maxstep', default=0, type=int)
    paser.add_argument('--lr', default=0.1, type=float)
    paser.add_argument('--lrd', default=0.03, type=float)
    paser.add_argument('--maxnorm', default=10, type=float)
    paser.add_argument('--word_emb', action='store_false', default=True)
    paser.add_argument('--test', action='store_true', default=False)
    paser.add_argument('--std', default=1, type=int)
    paser.add_argument('--opt', default='momentum', type=str)
    paser.add_argument('--hl', default=1, type=int)
    # 'train1000/trf_noise1.0_blstm_cnn_we100_ce30_c2wcnn_dropout0.5_adampretrain_baseline/crf_models/TRF-uns_ep11.ckpt'
    args=paser.parse_args()
    print(args)
    tf.set_random_seed(args.seed)
    np.random.seed(args.seed)
    import random
    random.seed(args.seed)
    main()