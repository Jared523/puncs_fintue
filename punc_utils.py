# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Named entity recognition fine-tuning: utilities to work with CoNLL-2003 task. """

from __future__ import absolute_import, division, print_function

import logging
import os
import torch
from torch.utils.data import TensorDataset
import json
import random
from io import open
from tqdm import tqdm
logger = logging.getLogger(__name__)


PUN_DICT  = ['O', 'COMMA', 'PERIOD', 'COLON']

class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, words, labels):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.words = words
        self.labels = labels


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, valid_mask, mask_labels):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.valid_mask = valid_mask
        self.mask_labels = mask_labels

def read_examples_from_file(data_dir, mode):
    file_path = os.path.join(data_dir, "{}.json".format(mode))
    modes_num = {'train':50000, 'dev':10000, 'test':10000 }
    guid_index = 1
    examples = []
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)
        # random.shuffle(data)
        data = data[:modes_num[mode]]
        data_iterator = tqdm(data, desc="Loading: {} Data".format(mode))
        for example in data_iterator:
            words = example['tokens']
            labels = example['tags']
            examples.append(InputExample(guid="{}-{}".format(mode, guid_index),
                                         words=words,
                                         labels=labels))
            guid_index += 1
        # for line in f:
        #     line=line.replace("\u200e","")
        #     line=line.replace("\u200f","")

        #     if line.startswith("-DOCSTART-") or line == "" or line == "\n":
        #         if words:
        #             examples.append(InputExample(guid="{}-{}".format(mode, guid_index),
        #                                          words=words,
        #                                          labels=labels))
        #             guid_index += 1
        #             words = []
        #             labels = []
        #     else:
        #         splits = line.split(" ")
        #         if splits[0] =="":
        #             continue
        #         words.append(splits[0])
        #         if len(splits) > 1:
        #             labels.append(splits[-1].replace("\n", ""))
        #         else:
        #             # Examples could have no label for mode = "test"
        #             labels.append("O")
        # if words:
        #     examples.append(InputExample(guid="%s-%d".format(mode, guid_index),
        #                                  words=words,
        #                                  labels=labels))
    return examples


def convert_examples_to_features(examples,
                                 label_list,
                                 max_seq_length,
                                 ltp_tokenizer,
                                 tokenizer,
                                 special_tokens_count = 2,
                                #  cls_token_at_end=False,
                                 cls_token="[CLS]",
                                #  cls_token_segment_id=1,
                                 sep_token="[SEP]",
                                 padding_label_ids = -100
                                #  sep_token_extra=False,
                                #  pad_on_left=False,
                                #  pad_token=0,
                                #  pad_token_segment_id=0,
                                #  pad_token_label_id=-1,
                                #  sequence_a_segment_id=0,
                                #  mask_padding_with_zero=True
                                 ):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 1000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        tokens = example.words
        label_ids = [ label_map[label] for label in example.labels]

        ltp_tokens = ltp_tokenizer.pipeline(''.join(tokens), tasks=["cws"])

        # add [MASK]
        item = []
        for i in ltp_tokens.cws:
            item.append(i)
            item.append('[MASK]')
        tokens = tokenizer.tokenize(''.join(item))

        # for word, label in zip(example.words, example.labels):
        #     word_tokens = tokenizer.tokenize(word)
        #     tokens.extend(word_tokens)
        #     # Use the real label id for the first token of the word, and padding ids for the remaining tokens
        #     label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        # special_tokens_count = 3 if sep_token_extra else 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[:(max_seq_length - special_tokens_count)]
            # label_ids = label_ids[:(max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.

        tokens = [cls_token] + tokens + [sep_token]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        mask_positions = [idx for idx,t in enumerate(tokens) if t == '[MASK]' ]
        mask_num = len(mask_positions)
        mask_labels = []
        start = 0
        for i in ltp_tokens.cws:
            mask_labels.append(label_ids[len(i)+start-1])
            start += len(i)
            if len(mask_labels) >= mask_num:
                break
        assert len(mask_positions) == len(mask_labels)

        valid_mask = [0] * max_seq_length
        for i in mask_positions:
            valid_mask[i] = 1

        # padding
        padding_length = max_seq_length - len(input_ids)
        input_ids += [0] * padding_length
        input_mask += [0] * padding_length
        segment_ids = [0] * len(input_ids)

        # padding labels
        mask_labels += [padding_label_ids] * (max_seq_length - len(mask_labels))

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(valid_mask) == max_seq_length
        assert len(mask_labels) == max_seq_length  

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              valid_mask=valid_mask,
                              mask_labels=mask_labels))
    return features


def load_and_cache_examples(args, ltp_tokenizer, tokenizer, labels, mode):
    # if args.local_rank not in [-1, 0] and not evaluate:
    #     torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    if not os.path.exists(args.data_cache_dir):
        os.makedirs(args.data_cache_dir)
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_cache_dir, "cached_{}_{}_{}".format(mode,
        list(filter(None, args.model_name_or_path.split("/"))).pop(),
        str(args.max_seq_length)))
    
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        examples = read_examples_from_file(args.data_dir, mode)
        features = convert_examples_to_features(examples, labels, args.max_seq_length, ltp_tokenizer, tokenizer)
        # if args.local_rank in [-1, 0]:
        #     logger.info("Saving features into cached file %s", cached_features_file)
        #     torch.save(features, cached_features_file)
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)
    # if args.local_rank == 0 and not evaluate:
    #     torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_valid_mask = torch.tensor([f.valid_mask for f in features], dtype=torch.long)
    all_mask_labels = torch.tensor([f.mask_labels for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_valid_mask, all_mask_labels)
    return dataset

# def get_labels(path):
#     if path:
#         with open(path, "r") as f:
#             labels = f.read().splitlines()
#         if "O" not in labels:
#             labels = ["O"] + labels
#         return labels
#     else:
#         return ["O", "B-MISC", "I-MISC",  "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]