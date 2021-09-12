# Copyright 2020 Tsinghua University, Author: Yunfu Song
# Apache 2.0.
# This script contrains functions to calculate F1 score.

import abc

def get_span_labels(sentence_tags, inv_label_mapping=None):
  """Go from token-level labels to list of entities (start, end, class)."""

  if inv_label_mapping:
    sentence_tags = [inv_label_mapping[i] for i in sentence_tags]
  span_labels = []
  last = 'O'
  start = -1
  for i, tag in enumerate(sentence_tags):
    pos, _ = (None, 'O') if tag in ['O','<s>','</s>'] else tag.split('-')
    if (pos == 'S' or pos == 'B' or tag == 'O') and last != 'O':
      span_labels.append((start, i - 1, last.split('-')[-1]))
    if pos == 'B' or pos == 'S' or last == 'O':
      start = i
    last = tag
  if sentence_tags[-1] != 'O':
    span_labels.append((start, len(sentence_tags) - 1,
                        sentence_tags[-1].split('-')[-1]))
  return span_labels


def get_tags(span_labels, length, encoding):
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

class Scorer(object):
  __metaclass__ = abc.ABCMeta

  def __init__(self):
    self._updated = False
    self._cached_results = {}

  @abc.abstractmethod
  def update(self, examples, predictions):
    self._updated = True


  @abc.abstractmethod
  def _get_results(self):
    return []

  def get_results(self, prefix=""):
    results = self._get_results() if self._updated else self._cached_results
    self._cached_results = results
    self._updated = False
    return [(prefix + k, v) for k, v in results]

  def results_str(self):
    return " - ".join(["{:}: {:.2f}".format(k, v)
                       for k, v in self.get_results()])


class WordLevelScorer(Scorer):
  __metaclass__ = abc.ABCMeta

  def __init__(self):
    super(WordLevelScorer, self).__init__()
    self._total_words = 0
    self._examples = []
    self._preds = []

  def update(self, examples, predictions):
    super(WordLevelScorer, self).update(examples, predictions)
    n_words = 0
    for example, preds in zip(examples, predictions):
      self._examples.append(example[1:len(example) - 1])
      self._preds.append(list(preds)[1:len(example) - 1])
      n_words += len(example) - 2
    self._total_words += n_words



class F1Scorer(WordLevelScorer):
  __metaclass__ = abc.ABCMeta

  def __init__(self):
    super(F1Scorer, self).__init__()
    self._n_correct, self._n_predicted, self._n_gold = 0, 0, 0

  def _get_results(self):
    if self._n_correct == 0:
      p, r, f1 = 0, 0, 0
    else:
      p = 100.0 * self._n_correct / self._n_predicted
      r = 100.0 * self._n_correct / self._n_gold
      f1 = 2 * p * r / (p + r)
    return [
        ("precision", p),
        ("recall", r),
        ("f1", f1)
    ]


class EntityLevelF1Scorer(F1Scorer):
  def __init__(self, label_mapping=None):
    super(EntityLevelF1Scorer, self).__init__()

    self._inv_label_mapping=label_mapping
  def _get_results(self):
    self._n_correct, self._n_predicted, self._n_gold = 0, 0, 0
    for example, preds in zip(self._examples, self._preds):
      sent_spans = set(get_span_labels(
          example, self._inv_label_mapping))
      span_preds = set(get_span_labels(
          preds, self._inv_label_mapping))
      self._n_correct += len(sent_spans & span_preds)
      self._n_gold += len(sent_spans)
      self._n_predicted += len(span_preds)
    return super(EntityLevelF1Scorer, self)._get_results()


class AccuracyScorer(WordLevelScorer):
  def __init__(self, auto_fail_label=None):
    super(AccuracyScorer, self).__init__()
    self._auto_fail_label = auto_fail_label

  def _get_results(self):
    correct, count = 0, 0
    for example, preds in zip(self._examples, self._preds):
      for y_true, y_pred in zip(example, preds):
        count += 1
        correct += (1 if y_pred == y_true and y_true != self._auto_fail_label
                    else 0)
    return [
        ("accuracy", 100.0 * correct / count)
    ]
