import os
import json
import re
import zipfile

from base import *



output_path = 'data/raw/chunk/'

wb.mkdir(output_path)

DIGIT_RE = re.compile('\d')
normalize_digits=True


def process_file(src_file, fword, ftag, word_vocab=None, tag_vocab=None):
    """
    read from the source file and write to fword and ftag
    Args:
        src_file: str, file name
        fword: fp, word file
        ftag:  fp, tag file
        word_vocab: word Vocab
        tag_vocab:  tag Vocab
    """
    with open(src_file, 'rt') as f:
        word_list = []
        tag_list = []
        for line in f:
            line = line.strip().split()
            if not line:
                if word_list:
                    span_labels=get_span_labels(tag_list)
                    tag_list=get_tags(span_labels,len(tag_list))
                    fword.write(' '.join(word_list) + '\n')
                    ftag.write(' '.join(tag_list) + '\n')
                    word_list = []
                    tag_list = []
                continue
            if line[0] == '-DOCSTART-':
                continue

            a = line
            word = DIGIT_RE.sub('0', a[0]) if normalize_digits else a[0]
            tag = a[-1]
            word_list.append(word)
            tag_list.append(tag)




def process_data(file_list, output_name, word_vocab=None, tag_vocab=None):
    with open(output_name + '.wod', 'wt') as fwod, open(output_name + '.tag', 'wt') as ftag:
        for src_file in file_list:
            process_file(src_file, fwod, ftag, word_vocab, tag_vocab)

def get_span_labels(sentence_tags, inv_label_mapping=None):
  """Go from token-level labels to list of entities (start, end, class)."""

  if inv_label_mapping:
    sentence_tags = [inv_label_mapping[i] for i in sentence_tags]
  span_labels = []
  last = 'O'
  start = -1
  for i, tag in enumerate(sentence_tags):
    pos, _ = (None, 'O') if tag == 'O' else tag.split('-')
    if (pos == 'S' or pos == 'B' or tag == 'O') and last != 'O':
      span_labels.append((start, i - 1, last.split('-')[-1]))
    if pos == 'B' or pos == 'S' or last == 'O':
      start = i
    last = tag
  if sentence_tags[-1] != 'O':
    span_labels.append((start, len(sentence_tags) - 1,
                        sentence_tags[-1].split('-')[-1]))
  return span_labels


def get_tags(span_labels, length, encoding='BIOES'):
  """Converts a list of entities to token-label labels based on the provided
  encoding (e.g., BIOES).
  """

  tags = ['O' for _ in range(length)]
  for s, e, t in span_labels:
    for i in range(s, e + 1):
      tags[i] = 'I-' + t
    if 'E' in encoding:
      tags[e] = 'E-' + t
    if 'B' in encoding:
      tags[s] = 'B-' + t
    if 'S' in encoding and s == e:
      tags[s] = 'S-' + t
  return tags


if __name__ == '__main__':
    train_files=['data/raw/chunk/train_subset.txt']
    valid_files=['data/raw/chunk/dev.txt']
    test_files = ['data/raw/chunk/test.txt']

    # summary all the files
    whole_data = os.path.join(output_path, 'src_data')
    process_data(train_files + valid_files + test_files, whole_data)

    # generate vocabularys
    v_wod = vocab.Vocab().generate_vocab([whole_data + '.wod'], max_size=None,
                                         add_beg_token='<s>',
                                         add_end_token='</s>',
                                         add_unk_token='<unk>')
    v_wod.create_chars()
    v_wod.write(os.path.join(output_path, 'vocab.wod'))

    v_tag = vocab.Vocab().generate_vocab([whole_data + '.tag'],
                                         add_beg_token='<s>',
                                         add_end_token='</s>',
                                         add_unk_token=None)
    v_tag.write(os.path.join(output_path, 'vocab.tag'))

    # output the final files
    output_train = os.path.join(output_path, 'train')
    output_valid = os.path.join(output_path, 'valid')
    output_test = os.path.join(output_path, 'test')
    process_data(train_files, output_train, v_wod, v_tag)
    process_data(valid_files, output_valid, v_wod, v_tag)
    process_data(test_files, output_test, v_wod, v_tag)

    # write information
    info = dict()
    info['train'] = [(output_train + '.wod', output_train + '.tag')]
    info['valid'] = [(output_valid + '.wod', output_valid + '.tag')]
    info['test'] = [(output_test + '.wod', output_test + '.tag')]
    info['vocab'] = (os.path.join(output_path, 'vocab.wod'), os.path.join(output_path, 'vocab.tag'))
    info['nbest'] = reader.wsj0_nbest()
    with open(output_path+'data.info', 'wt') as f:
        json.dump(info, f, indent=4)

    # data = reader.Data().load_raw_data([info['train'][0][0], info['valid'][0][0], info['test'][0][0]])
    # print(data.get_vocab_size())
    #
    # txtinfo = wb.TxtInfo(info['train'][0][0])
    # print(txtinfo)

