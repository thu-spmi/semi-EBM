# Copyright 2020 Tsinghua University, Author: Yunfu Song
# Apache 2.0.
# This script contrains trainer of biLSTMLM unsupervised training experiments.

import numpy as np
import tensorflow as tf
import os
import time
import json

from base import *
import bilstm_uns
from trf.isample import sampler


class Config(wb.Config):
    def __init__(self, data):
        Config.value_encoding_map[lr.LearningRate] = str

        self.min_len = data.get_min_len()
        self.max_len = data.get_max_len()
        self.pi_true = data.get_pi_true()
        self.pi_0 = self.pi_true
        self.word_vocab_size = data.get_vocab_size()
        self.tag_vocab_size = data.get_tag_size()
        self.beg_tokens = data.get_beg_tokens()  # [word_beg_token, tag_beg_token]
        self.end_tokens = data.get_end_tokens()  # [word_end_token, tag_end_token]

        # used in sampler
        self.beg_token = self.beg_tokens[0]
        self.end_token = self.end_tokens[0]

        # potential features
        self.mix_config = bilstm_uns.Config(data)

        # training
        self.crf_batch_size = 100
        self.trf_batch_size = 100
        self.noise_factor = 1
        self.data_factor = 1  # the generated data rate
        self.sampler_config = sampler.LSTM.Config(self.word_vocab_size, 200, 1)
        self.sampler_config.learning_rate = 0.1

        # learning rate
        self.lr = lr.LearningRateEpochDelay(1.0)
        self.max_epoch = 50

        # dbg
        self.write_dbg = False

    def __str__(self):
        return 'trf_' + str(self.mix_config)


def seq_list_package(seq_list, pad_value=0):
    lengths = [len(s) for s in seq_list]
    inputs = np.ones([len(seq_list), np.max(lengths)]) * pad_value
    labels = np.ones_like(inputs) * pad_value

    for i, s in enumerate(seq_list):
        n = len(s)
        inputs[i][0:n] = s.x[0]
        labels[i][0:n] = s.x[1]

    return inputs, labels, lengths


def seq_list_unfold(inputs, labels, lengths):
    seq_list = []
    for x, y, n in zip(inputs, labels, lengths):
        seq_list.append(seq.Seq([x[0:n], y[0:n]]))
    return seq_list


class TRF(object):
    def __init__(self, config, data, logdir,
                 device='/gpu:0', name='crf'):

        self.config = config
        self.data_obse = data
        self.logdir = logdir
        self.name = name

        # net
        self.config.mix_config.noise_factor = self.config.noise_factor
        self.net = bilstm_uns.UnsNet(self.config.mix_config, device=device,
                                   word_to_chars=data.vocabs[0].word_to_chars)

        # sampler
        # self.sampler = sampler.LSTM(config, device)
        # self.sampler.len_prob = config.pi_true
        # self.sampler.update_len_prob = False

        # learning rate
        self.cur_lr = 1.0

        # training info
        self.training_info = {'trained_step': 0,
                              'trained_epoch': 0,
                              'trained_time': 0}

        # debuger
        self.write_files = wb.FileBank(os.path.join(logdir, name + '.dbg'))
        # time recorder
        self.time_recoder = wb.clock()
        # default save name
        self.check_save_name = os.path.join(self.logdir, 'check_models')
        self.best_save_name = os.path.join(self.logdir, 'best_models')
        os.makedirs(self.check_save_name, exist_ok=True)
        os.makedirs(self.best_save_name, exist_ok=True)

        self.check_saver = tf.train.Saver(max_to_keep=100)
        self.best_saver = tf.train.Saver(max_to_keep=100)


    @property
    def session(self):
        return tf.get_default_session()

    def save(self, fname=None):
        if fname is None:
            fname = self.default_save_name+'/default'
        else:
            fname=self.default_save_name+'/'+fname
        print('[TRF-uns] save to', fname)
        with open(fname + '.config', 'wt') as f:
            json.dump(self.training_info, f, indent=4)
            f.write('\n')
            self.config.save(f)
        self.net.save(self.session, fname)

    def restore(self, fname=None):
        if fname is None:
            fname = self.default_save_name
        print('[TRF-uns] restore from', fname)
        with open(fname + '.config', 'rt') as f:
            self.training_info = wb.json_load(f)
            print(json.dumps(self.training_info, indent=2))
        self.net.restore(self.session, fname)


    def get_trf_logps(self, seq_list, batch_size=100):
        logps = []

        for i in range(0, len(seq_list), batch_size):
            if isinstance(seq_list[0], seq.Seq):
                inputs, labels, lengths = seq_list_package(seq_list[i: i + batch_size])
            else:
                inputs, lengths = reader.produce_data_to_array(seq_list[i: i+batch_size])
            logps.append( self.net.run_trf_logp(self.session, inputs, lengths))

        return logps

    def rescore(self, x_list):
        return - self.get_trf_logps(x_list)


    def trf_eval(self, seq_list):
        logps = self.get_trf_logps(seq_list)
        nll = -np.mean(logps)
        # words = np.sum([len(x) - 1 for x in seq_list])
        # ppl = np.exp(-np.sum(logps) / words)
        return nll, 0

    def initialize(self):
        print('[TRF-uns] mix_param_num = {:,}'.format(self.net.run_parameter_num(self.session)))

    def update(self, data_list):

        data_x_list = seq.get_x(data_list)
        src_data_num = len(data_list)

        # generate noise samples
        with self.time_recoder.recode('sampling'):

            nd = len(data_list)
            k1 = int(nd * self.config.data_factor)
        #     k2 = int((nd + k1) * self.config.noise_factor)
        #
        #     samples_all = self.sampler.generate(k1 + k2)
        #
        #     seq_x_list = data_x_list + samples_all
        #     seq_list = [seq.Seq(x) for x in seq_x_list]
        #     noise_logps = self.sampler.get_log_probs(seq_x_list)
            data_num = nd + k1
        #     noise_num = k2
        #     seq_lens = [len(x) for x in seq_x_list]

        with self.time_recoder.recode('update_net'):
            trf_inputs, trf_lengths = reader.produce_data_to_array(data_x_list)

            # self.net.crf_net.run_crf_update(self.session, crf_inputs, crf_labels, crf_lengths, learning_rate=self.cur_lr)
            self.net.run_update(self.session,
                                trf_inputs, trf_lengths, data_num,
                                learning_rate=self.cur_lr)

        # update simulater
        # with self.time_recoder.recode('update_sampler'):
        #     self.sampler.update(data_x_list, np.ones(src_data_num) / src_data_num)
        #     sampler_ll = self.sampler.eval_nll(data_x_list)
        #
        # # update dbg info
        # if self.config.write_dbg:
        #     f = self.write_files.get('noise')
        #     f.write('step={}\n'.format(self.training_info['trained_step']))
        #     f.write('[d/s] [model_logp] [noise_logp] [ seq ]\n')
        #     for i, s in enumerate(seq_x_list):
        #         f.write('{:>5} {:<12.5f} {:<12.5f} '.format(
        #             'd' if i < len(data_x_list) else 's',
        #             noise_logps[i], noise_logps[i]))
        #         f.write('[' + ' '.join(str(w) for w in s) + ']\n')
        #     f.flush()
        #
        # print_infos = OrderedDict()
        # print_infos['aux_nll'] = sampler_ll

        return 0
    def get_lr(self,step):
        warm_up_multiplier = np.minimum(step,self.config.warm_up_steps) / self.config.warm_up_steps
        decay_multiplier = 1.0 / (1 + self.config.lr_decay *np.sqrt(step))
        lr = self.config.lr * warm_up_multiplier * decay_multiplier
        return lr
    def train(self, print_per_epoch=0.1, operation=None):

        # initialize
        self.initialize()

        # train_list = self.data_obse.datas[0]

        print('[bilm] [Train]...')
        time_beginning = time.time()
        trf_train_nll = []

        self.data_obse.train_batch_size = self.config.trf_batch_size
        self.data_obse.is_shuffle = True
        epoch_step_num = self.data_obse.get_epoch_step_num()
        print('[bilm] epoch_step_num={}'.format(epoch_step_num))
        print('[bilm] trf_train_list={}'.format(len(self.data_obse.datas[0])))
        max_num = epoch_step_num * self.config.max_epoch
        eval_every = int(epoch_step_num * self.config.eval_every)
        print_next_epoch = 0
        for step in range(max_num):

            data_seqs = self.data_obse.__next__()

            ###########################
            # extra operations
            ###########################
            if (step + 1) % eval_every == 0:
                self.check_saver.save(tf.get_default_session(), os.path.join(self.check_save_name, 'step%d' % (step + 1)))
                if operation is not None:
                    operation.run(step, 0)



            # update epoches


            # update training information
            self.training_info['trained_step'] += 1
            self.training_info['trained_time'] = (time.time() - time_beginning) / 60

            # update paramters
            with self.time_recoder.recode('update'):
                # learining rate
                self.cur_lr = self.get_lr(step + 1)
                # update
                nce_loss = self.update(data_seqs)
                # nce_loss=0

            # evaulate the nll
            with self.time_recoder.recode('eval_train_nll'):
                trf_train_nll.append(self.trf_eval(data_seqs)[0])

            if (step + 1) % eval_every == 0:

                time_since_beg = (time.time() - time_beginning) / 60


                info = OrderedDict()
                info['step'] = step
                info['time'] = time_since_beg
                info['lr_mix'] = '{:.2e}'.format(self.cur_lr)
                info['loss'] = nce_loss
                info['trf_nll'] = np.mean(trf_train_nll[-self.data_obse.get_epoch_step_num():])

                log.print_line(info)
                print('[end]')

                # write time
                f = self.write_files.get('time')
                f.write('step={} time={:.2f} '.format(step, time_since_beg))
                f.write(
                    ' '.join(['{}={:.2f}'.format(x[0], x[1]) for x in self.time_recoder.time_recoder.items()]) + '\n')
                f.flush()

class DefaultOps(wb.Operation):
    def __init__(self, m, valid_seq_list, test_seq_list, nbest_files=None):
        super().__init__()

        self.m = m
        self.seq_list_tuple = (valid_seq_list, test_seq_list)

        self.perform_next_epoch = 1.0
        self.perform_per_epoch = 1.0
        self.write_models = wb.mkdir(os.path.join(self.m.logdir, 'crf_models'))

        # nbest
        if nbest_files is not None:
            self.nbest_cmp = reader.NBest(*nbest_files)
        else:
            self.nbest_cmp = None

    def perform(self, step, epoch):
        print('[Ops] performing')

        res_prec = []
        res_time = []
        for name, seq_list in zip(['valid', 'test'], self.seq_list_tuple):
            # tagging
            time_beg = time.time()
            tag_res_list = self.m.get_tag(seq.get_x(seq_list))
            tag_time = time.time() - time_beg

            # write
            log.write_seq_to_file(tag_res_list, os.path.join(self.m.logdir, 'result_%s.tag' % name))
            gold_tag = seq.get_h(seq_list)
            log.write_seq_to_file(gold_tag, os.path.join(self.m.logdir, 'result_%s.gold.tag' % name))

            # compute wer
            P, R, F = seq.tag_error(gold_tag, tag_res_list)

            res_prec.append(P)
            res_time.append(tag_time / 60)

        print('epoch={:.2f} valid={:.2f} test={:.2f} valid_time={:.2f} test_time={:.2f}'.format(
            epoch, res_prec[0], res_prec[1], res_time[0], res_time[1]
        ))

        res = wb.FRes(os.path.join(self.m.logdir, 'results_tag_err.log'))
        res_name = 'epoch%.2f' % epoch
        res.Add(res_name, ['valid', 'test'], res_prec)

        # nbest
        if self.nbest_cmp is not None:
            # resocring
            time_beg = time.time()
            self.nbest_cmp.lmscore = self.m.rescore(self.nbest_cmp.get_nbest_list(self.m.data_obse))
            rescore_time = time.time() - time_beg

            # compute wer
            time_beg = time.time()
            wer = self.nbest_cmp.wer()
            wer_time = time.time() - time_beg

            wb.WriteScore(self.write_models + '/epoch%.2f' % epoch + '.lmscore', self.nbest_cmp.lmscore)
            print('epoch={:.2f} test_wer={:.2f} lmscale={} '
                  'rescore_time={:.2f}, wer_time={:.2f}'.format(
                epoch, wer, self.nbest_cmp.lmscale,
                rescore_time / 60, wer_time / 60))

            res = wb.FRes(os.path.join(self.m.logdir, 'results_nbest_wer.log'))
            res_name = 'epoch%.2f' % epoch
            res.Add(res_name, ['lm-scale'], [self.nbest_cmp.lmscale])
            res.Add(res_name, ['wer'], [wer])
