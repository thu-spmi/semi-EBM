import json
import os
from base import *
# please run ./full/data_process.py first


def get_first_lines(input_file, output_file, num):
    with open(input_file, 'rt') as fin, open(output_file, 'wt') as fout:
        n = 0
        for line in fin:
            fout.write(line)
            n += 1
            if n >= num:
                break


def read_txt(file):
    with open(file, 'rt') as f:
        return [line for line in f]


def write_txt(file, str_list):
    with open(file, 'wt') as f:
        for s in str_list:
            f.write(s)


def add_path(old_paths):
    return [s for s in old_paths]

data_emb = 'data/raw/glove/glove.6B.300d.txt'
def emb_trans(v_wod):
    # prepare the word embedding
    print('load word embedding')
    wod_num = v_wod.get_size()
    emb_dict = dict()
    emb_size = 0
    with open(data_emb) as f:
        for line in f:
            a = line.split()
            wod = a[0]
            emb = [float(i) for i in a[1:]]
            emb_size = max(emb_size, len(emb))

            emb_dict[wod] = emb

    print('load emb, size=%d' % emb_size)
    emb_arr = np.zeros([wod_num, emb_size])
    miss_num = 0
    for i in range(len(emb_arr)):
        wod = v_wod.words[i]
        if wod in emb_dict:
            emb_arr[i] = np.array(emb_dict[wod])
        elif wod.lower() in emb_dict:
            emb_arr[i] = np.array(emb_dict[wod.lower()])
        else:
            print('load emb, no emb of i=%d word=%s' % (i, wod))
            miss_num += 1

    print('load emb: missing num=%d (%.2f%%)' % (miss_num, 100.0 * miss_num / wod_num))

    return emb_arr

if __name__ == '__main__':
    with open('data/raw/ner/data.info') as f:
        info = json.load(f)

    with open('data/processed/google1B/data.info') as f:
        info_ext = json.load(f)

    info['train'] = [add_path(info['train'][0])]
    info['valid'] = [add_path(info['valid'][0])]
    info['test'] = [add_path(info['test'][0])]
    info['vocab'] = add_path(info['vocab'])
    info['google_1b'] = [(s, None) for s in info_ext['train_all']]

    id2tag = {}
    vocab = open(info['vocab'][-1], 'r')
    for line in vocab.readlines():
        line = line.strip().split()
        try:
            id = int(line[0])
            id2tag[id] = line[1]
        except:
            pass
    print(id2tag)

    v_wod = seq.vocab.Vocab().read(info['vocab'][0])
    emb_arr = emb_trans(v_wod)
    info['word_emb_d%d' % emb_arr.shape[1]] = 'data/processed/ner/word_emb_d%d.npy' % emb_arr.shape[1]
    np.save(info['word_emb_d%d' % emb_arr.shape[1]], emb_arr)

    # rate = 0.02
    train_wod_list = read_txt(info['train'][0][0])
    train_tag_list = read_txt(info['train'][0][1])

    print(len(train_wod_list))
    import numpy as np

    np.random.seed(0)
    inds = np.random.permutation(len(train_wod_list))
    train_wod_list = [train_wod_list[i] for i in inds]
    train_tag_list = [train_tag_list[i] for i in inds]

    for n in [len(train_wod_list)//50, len(train_wod_list)//10, len(train_wod_list)]:
        file_name = wb.mkdir('data/processed/ner/') + 'train.%d' % n
        write_txt(file_name + '.wod', train_wod_list[0: n])
        write_txt(file_name + '.tag', train_tag_list[0: n])

        part_n50 = n * 50
        part_n250 = n * 250
        part_n500 = n * 500

        print('n=%d, part_n=%d %d %d Load 1billion data!' % (n, part_n50, part_n250, part_n500))

        train_wod_list_ex = read_txt(info_ext['train_all'][0])
        fi = 1
        while len(train_wod_list_ex) < part_n50:
            train_wod_list_ex += read_txt(info_ext['train_all'][fi])
            fi += 1
        assert len(train_wod_list_ex) >= part_n50

        write_txt(file_name + '.part50.wod', train_wod_list_ex[0: part_n50])
        info['train%d.part50' % n] = [(file_name + '.part50.wod', None)]

        while len(train_wod_list_ex) < part_n250:
            train_wod_list_ex += read_txt(info_ext['train_all'][fi])
            fi += 1
        assert len(train_wod_list_ex) >= part_n250

        write_txt(file_name + '.part250.wod', train_wod_list_ex[0: part_n250])
        info['train%d.part250' % n] = [(file_name + '.part250.wod', None)]

        while len(train_wod_list_ex) < part_n500:
            train_wod_list_ex += read_txt(info_ext['train_all'][fi])
            fi += 1
        assert len(train_wod_list_ex) >= part_n500

        write_txt(file_name + '.part500.wod', train_wod_list_ex[0: part_n500])
        info['train%d.part500' % n] = [(file_name + '.part500.wod', None)]

        info['train%d' % n] = [(file_name + '.wod', file_name + '.tag')]

    # n = len(train_wod_list) // 10
    # for seed in [10, 100, 1000, 10000]:
    #     np.random.seed(seed)
    #     inds = np.random.permutation(len(train_wod_list))
    #     write_txt('data/processed/ner/train.%d.seed.%d.wod' % (n, seed), [train_wod_list[i] for i in inds[:n]])
    #     write_txt('data/processed/ner/train.%d.seed.%d.tag' % (n, seed), [train_tag_list[i] for i in inds[:n]])
    #     info['train%d_seed%d' % (n, seed)] = [('data/processed/ner/train.%d.seed.%d.wod' % (n, seed), 'data/processed/ner/train.%d.seed.%d.tag' % (n, seed))]
    import pickle
    pickle.dump(id2tag,open('data/processed/ner/id2tag.pkl','wb'))
    info['id2tag']='data/processed/ner/id2tag.pkl'

    with open('data/processed/ner/data.info', 'wt') as f:
        json.dump(info, f, indent=4)