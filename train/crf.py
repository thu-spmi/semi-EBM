# Copyright 2020 Tsinghua University, Author: Yunfu Song
# Apache 2.0.
# This script contrains trainer of CRF supervised training experiments.

import numpy as np
import tensorflow as tf
import os
import time
import json

from base import *
import mix_net


class Config(wb.Config):
    def __init__(self, data):
        Config.value_encoding_map[lr.LearningRate] = str

        self.word_vocab_size = data.get_vocab_size()
        self.tag_vocab_size = data.get_tag_size()
        self.beg_tokens = data.get_beg_tokens()  # [word_beg_token, tag_beg_token]
        self.end_tokens = data.get_end_tokens()  # [word_end_token, tag_end_token]

        # potential features
        self.mix_config = mix_net.Config(self.word_vocab_size, data.get_char_size(), self.tag_vocab_size,
                                         self.beg_tokens[1],
                                         self.end_tokens[1])  # discrete features

        # training
        self.train_batch_size = 100

        # learning rate
        self.lr_mix = lr.LearningRateEpochDelay(1.0)
        self.opt_mix = 'sgd'
        self.max_epoch = 100

        # dbg
        self.write_dbg = False

    def __str__(self):
        return 'crf_' + str(self.mix_config)


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


class CRF(object):
    def __init__(self, config, data, logdir,
                 device='/gpu:0', name='crf'):

        self.config = config
        self.data = data
        self.logdir = logdir
        self.name = name

        self.phi_mix = mix_net.Net(self.config.mix_config, is_training=True, device=device,
                                   word_to_chars=data.vocabs[0].word_to_chars)

        # learning rate
        self.cur_lr_mix = 1.0

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
        self.check_saver=tf.train.Saver(max_to_keep=1)
        self.best_saver=tf.train.Saver(max_to_keep=1)

    @property
    def session(self):
        return tf.get_default_session()

    def save(self, fname=None):
        if fname is None:
            fname = self.default_save_name
        print('[CRF] save to', fname)
        with open(fname + '.config', 'wt') as f:
            json.dump(self.training_info, f, indent=4)
            f.write('\n')
            self.config.save(f)
        self.phi_mix.save(self.session, fname)

    def restore(self, fname=None):
        if fname is None:
            fname = self.default_save_name
        print('[CRF] restore from', fname)
        with open(fname + '.config', 'rt') as f:
            self.training_info = wb.json_load(f)
            print(json.dumps(self.training_info, indent=2))
        self.phi_mix.restore(self.session, fname)

    def phi(self, inputs, labels, lengths):
        return self.phi_mix.run_phi(self.session, inputs, labels, lengths)

    def logz(self, inputs, lengths):
        return self.phi_mix.run_logz(self.session, inputs, lengths)

    def logps(self, inputs, labels, lengths):
        return self.phi_mix.run_logp(self.session, inputs, labels, lengths)

    def get_log_probs(self, seq_list, is_norm=True, batch_size=100):
        logps = np.zeros(len(seq_list))

        for i in range(0, len(seq_list), batch_size):
            if is_norm:
                logps[i: i + batch_size] = self.logps(*seq_list_package(seq_list[i: i + batch_size]))
            else:
                logps[i: i + batch_size] = self.phi(*seq_list_package(seq_list[i: i + batch_size]))

        return logps

    def get_tag(self, x_list, batch_size=100):
        t_list = []
        for i in range(0, len(x_list), batch_size):
            t_list += self.phi_mix.run_opt_labels(self.session,
                                                  *reader.produce_data_to_array(x_list[i: i+batch_size]))
        return t_list

    def eval(self, seq_list):
        logps = self.get_log_probs(seq_list)
        nll = -np.mean(logps)
        words = np.sum([len(x)-1 for x in seq_list])
        ppl = np.exp(-np.sum(logps) / words)

        return nll, ppl

    def initialize(self):
        print('[CRF] mix_param_num = {:,}'.format(self.phi_mix.run_parameter_num(self.session)))

    def update(self, data_list):
        # compute the scalars
        inputs, labels, lengths = seq_list_package(data_list)
        self.phi_mix.run_update(self.session, inputs, labels, lengths, learning_rate=self.cur_lr_mix)
        return None
    def get_lr(self,step):
        warm_up_multiplier = np.minimum(step,self.config.warm_up_steps) / self.config.warm_up_steps
        decay_multiplier = 1.0 / (1 + self.config.lr_decay *np.sqrt(step))
        lr = self.config.lr * warm_up_multiplier * decay_multiplier
        return lr
    def train(self, print_per_epoch=0.1, operation=None):

        # initialize

        # self.initialize()

        train_list = self.data.datas[0]
        valid_list = self.data.datas[1]

        print('[CRF] [Train]...')
        print('train_list=', len(train_list))
        print('valid_list=', len(valid_list))
        time_beginning = time.time()
        model_train_nll = []

        self.data.train_batch_size = self.config.train_batch_size
        self.data.is_shuffle = True
        last_epoch = 0
        print_next_epoch = 0
        best_res = 0
        for step, data_seqs in enumerate(self.data):

            # update training information
            self.training_info['trained_step'] += 1
            self.training_info['trained_epoch'] = self.data.get_cur_epoch()
            self.training_info['trained_time'] = (time.time() - time_beginning) / 60
            # update paramters
            with self.time_recoder.recode('update'):
                # learining rate
                self.cur_lr_mix = self.get_lr(step+1)
                # update
                self.update(data_seqs)

            # evaulate the NLL
            with self.time_recoder.recode('eval_train_nll'):
                model_train_nll.append(self.eval(data_seqs)[0])

            if (step+1)%self.config.eval_every ==0:

                time_since_beg = (time.time() - time_beginning) / 60

                with self.time_recoder.recode('eval'):
                    model_valid_nll = self.eval(valid_list)[0]

                info = OrderedDict()
                info['step'] = step+1
                info['time'] = time_since_beg
                info['lr_mix'] = '{:.2e}'.format(self.cur_lr_mix)
                info['train'] = np.mean(model_train_nll[-100:])
                info['valid'] = model_valid_nll
                log.print_line(info)

                print('[end]')

                # write time
                f = self.write_files.get('time')
                f.write('step={} time={:.2f} '.format(step+1, time_since_beg))
                f.write(' '.join(['{}={:.2f}'.format(x[0], x[1]) for x in self.time_recoder.items()]) + '\n')
                f.flush()


                self.check_saver.save(tf.get_default_session(), os.path.join(self.check_save_name, 'step%d' % (step+1)))
                res = operation.perform(step+1, 0)
                if res > best_res:
                    best_res = res
                    print('save new best:', os.path.join(self.best_save_name, 'step%d' % (step+1)))
                    self.best_saver.save(tf.get_default_session(), os.path.join(self.best_save_name, 'step%d' % (step+1)))

            if step+1 >= self.config.max_steps:
                print('[CRF] train stop!')
                # self.save(self.logdir+'/crf_models/crf_sup_ep%d'%epoch)
                # if operation is not None:
                #     operation.perform(step, epoch)
                break




class self_CRF(object):
    def __init__(self, config, data, data_full, logdir,
                 device='/gpu:0', name='crf',uns_weight=0.1):

        self.config = config
        self.data = data
        self.data_full = data_full
        self.logdir = logdir
        self.name = name

        self.phi_mix = mix_net.self_Net(self.config.mix_config, is_training=True, device=device,
                                   word_to_chars=data.vocabs[0].word_to_chars,uns_weight=uns_weight)

        # learning rate
        self.cur_lr_mix = 1.0

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
        self.check_saver=tf.train.Saver(max_to_keep=1)
        self.best_saver=tf.train.Saver(max_to_keep=1)

    @property
    def session(self):
        return tf.get_default_session()

    def save(self, fname=None):
        if fname is None:
            fname = self.default_save_name
        print('[CRF] save to', fname)
        with open(fname + '.config', 'wt') as f:
            json.dump(self.training_info, f, indent=4)
            f.write('\n')
            self.config.save(f)
        self.phi_mix.save(self.session, fname)

    def restore(self, fname=None):
        if fname is None:
            fname = self.default_save_name
        print('[CRF] restore from', fname)
        with open(fname + '.config', 'rt') as f:
            self.training_info = wb.json_load(f)
            print(json.dumps(self.training_info, indent=2))
        self.phi_mix.restore(self.session, fname)

    def phi(self, inputs, labels, lengths):
        return self.phi_mix.run_phi(self.session, inputs, labels, lengths)

    def logz(self, inputs, lengths):
        return self.phi_mix.run_logz(self.session, inputs, lengths)

    def logps(self, inputs, labels, lengths):
        return self.phi_mix.run_logp(self.session, inputs, labels, lengths)

    def get_log_probs(self, seq_list, is_norm=True, batch_size=100):
        logps = np.zeros(len(seq_list))

        for i in range(0, len(seq_list), batch_size):
            if is_norm:
                logps[i: i + batch_size] = self.logps(*seq_list_package(seq_list[i: i + batch_size]))
            else:
                logps[i: i + batch_size] = self.phi(*seq_list_package(seq_list[i: i + batch_size]))

        return logps

    def get_tag(self, x_list, batch_size=100):
        t_list = []
        for i in range(0, len(x_list), batch_size):
            t_list += self.phi_mix.run_opt_labels(self.session,
                                                  *reader.produce_data_to_array(x_list[i: i+batch_size]))
        return t_list

    def eval(self, seq_list):
        logps = self.get_log_probs(seq_list)
        nll = -np.mean(logps)
        words = np.sum([len(x)-1 for x in seq_list])
        ppl = np.exp(-np.sum(logps) / words)

        return nll, ppl

    def initialize(self):
        print('[CRF] mix_param_num = {:,}'.format(self.phi_mix.run_parameter_num(self.session)))

    def update(self, data_list, data_full_list):
        # compute the scalars
        inputs, labels, lengths = seq_list_package(data_full_list)

        data_x_list = seq.get_x(data_list)
        trf_inputs, trf_lengths = reader.produce_data_to_array(data_x_list)

        pred = self.phi_mix.run_opt_labels(self.session, trf_inputs, trf_lengths)
        pred_label, _ = reader.produce_data_to_array(pred)

        self.phi_mix.run_update(self.session, inputs, labels, lengths,trf_inputs,pred_label,trf_lengths, learning_rate=self.cur_lr_mix)
        return None
    def get_lr(self,step):
        warm_up_multiplier = np.minimum(step,self.config.warm_up_steps) / self.config.warm_up_steps
        decay_multiplier = 1.0 / (1 + self.config.lr_decay *np.sqrt(step))
        lr = self.config.lr * warm_up_multiplier * decay_multiplier
        return lr
    def train(self, print_per_epoch=0.1, operation=None):

        # initialize

        # self.initialize()

        train_list = self.data.datas[0]
        valid_list = self.data.datas[1]

        print('[CRF] [Train]...')
        print('train_list=', len(train_list))
        print('valid_list=', len(valid_list))
        time_beginning = time.time()
        model_train_nll = []

        self.data.train_batch_size = self.config.train_batch_size
        self.data.is_shuffle = True
        self.data_full.train_batch_size = self.config.train_batch_size
        self.data_full.is_shuffle = True
        best_res = 0
        for step, data_full_seqs in enumerate(self.data_full):
            epoch = self.data.get_cur_epoch()
            data_seqs = self.data.__next__()
            # update training information
            self.training_info['trained_step'] += 1
            self.training_info['trained_time'] = (time.time() - time_beginning) / 60

            # update paramters
            with self.time_recoder.recode('update'):
                # learining rate
                self.cur_lr_mix = self.get_lr(step+1)
                # update
                self.update(data_seqs,data_full_seqs)

            # evaulate the NLL
            with self.time_recoder.recode('eval_train_nll'):
                model_train_nll.append(self.eval(data_full_seqs)[0])

            if (step + 1) % self.config.eval_every == 0:

                time_since_beg = (time.time() - time_beginning) / 60

                with self.time_recoder.recode('eval'):
                    model_valid_nll = self.eval(valid_list)[0]

                info = OrderedDict()
                info['step'] = step+1
                info['time'] = time_since_beg
                info['lr_mix'] = '{:.2e}'.format(self.cur_lr_mix)
                info['train'] = np.mean(model_train_nll[-100:])
                info['valid'] = model_valid_nll
                log.print_line(info)

                print('[end]')

                # write time
                f = self.write_files.get('time')
                f.write('step={} time={:.2f} '.format(step+1, time_since_beg))
                f.write(' '.join(['{}={:.2f}'.format(x[0], x[1]) for x in self.time_recoder.items()]) + '\n')
                f.flush()

                self.check_saver.save(tf.get_default_session(), os.path.join(self.check_save_name, 'step%d' % (step + 1)))
                res = operation.perform(step+1, 0)
                if res > best_res:
                    best_res = res
                    print('save new best:', os.path.join(self.best_save_name, 'step%d' % (step + 1)))
                    self.best_saver.save(tf.get_default_session(), os.path.join(self.best_save_name, 'step%d' % (step + 1)))

            if step + 1 >= self.config.max_steps:
                print('[CRF] train stop!')
                break




class DefaultOps(wb.Operation):
    def __init__(self,args, m, valid_seq_list, test_seq_list,id2tag_name=None,test=False):
        super().__init__()

        self.m = m
        if not test:
            self.name='valid'
            self.seq_list=valid_seq_list
        else:
            self.name='test'
            self.seq_list=test_seq_list
        import scorer
        if args.task == 'pos':
            self.scorer = scorer.AccuracyScorer()
        else:
            import pickle
            id2tag = pickle.load(open(id2tag_name, 'rb'))
            print(id2tag)
            self.scorer = scorer.EntityLevelF1Scorer(id2tag)
        self.perform_next_epoch = 1.0
        self.perform_per_epoch = 1.0
        # self.write_models = wb.mkdir(os.path.join(self.m.logdir, 'crf_models'))
        self.task=args.task
    def perform(self, step, epoch):
        print('[Ops] performing')

        tag_res_list = self.m.get_tag(seq.get_x(self.seq_list))

        # write
        log.write_seq_to_file(tag_res_list, os.path.join(self.m.logdir, 'result_%s.tag' % self.name))
        gold_tag = seq.get_h(self.seq_list)
        # print([len(tag) for tag in gold_tag[:100]])
        log.write_seq_to_file(gold_tag, os.path.join(self.m.logdir, 'result_%s.gold.tag' % self.name))
        self.scorer.update(gold_tag, tag_res_list)
        res = self.scorer.get_results()
        if self.task == 'pos':
            acc = res[0]
            print('epoch={:.2f} {}={:.2f}'.format(
                epoch, acc[0], acc[1],
            ))
            return acc[1]
        else:
            P, R, F = res
            print('epoch={:.2f} {}={:.2f} {}={:.2f} {}={:.2f}'.format(
                epoch, P[0], P[1], R[0], R[1], F[0], F[1]
            ))
            return F[1]






















