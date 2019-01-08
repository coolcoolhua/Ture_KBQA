import torch.utils.data as Data
import numpy as np
from prepare import Log
import logging


class EntityDetectionDataset(Data.Dataset):

    def __init__(self, data_path, word2idx, oov_idx, padding_idx, padding_label, SEQUENCE_LEN):
        """
        Load datasets for entity detection
        :param data_path: e.g. data_train/valid/test.txt that are generated from annotated_fb_data_*.txt
        :param word2idx: dictionary for word to index
        :param oov_idx: index used for words which are Out of Vocabulary
        :param padding_idx: index used for padding in sequence
        :param padding_label: label value used for label sequence
        :param SEQUENCE_LEN: Max length of sequence
        """
        self.line_count = 0
        self.data = []
        self.label = []
        self.seq_len = []
        oov_count = 0
        word_count = 0
        oov_set = set()
        word_set = set()
        logging.debug("Start to read '{}'".format(data_path))
        for line in open(data_path, 'r'):
            self.line_count += 1
            line = line.strip()
            sentence_len = 0
            splits = line.split('\t')
            if len(splits) != 2:
                raise Exception("Failed to convert '{}' at line {}".format(line, self.line_count))
            question_split = splits[0].split(' ')
            label_split = splits[1].split(' ')
            if len(question_split) != len(label_split):
                raise Exception("Failed to convert '{}' at line {}".format(line, self.line_count))
            data_in = np.zeros(SEQUENCE_LEN, dtype=np.long)
            label_in = np.zeros(SEQUENCE_LEN, dtype=np.long)
            for idx in range(SEQUENCE_LEN):
                word_count += 1
                if idx >= len(question_split):
                    # add padding
                    data_in[idx] = padding_idx
                    label_in[idx] = padding_label
                else:
                    # add word
                    sentence_len += 1
                    word = question_split[idx].strip()
                    word_set.add(word)
                    if word2idx.__contains__(word):
                        data_in[idx] = word2idx[word]
                    else:
                        oov_set.add(word)
                        oov_count += 1
                        data_in[idx] = oov_idx
                    # add label
                    _label_idx = int(label_split[idx].strip())
                    if _label_idx != 1 and _label_idx != 0:
                        raise Exception("Failed to convert label in '{}' at line {}".format(line, self.line_count))
                    label_in[idx] = _label_idx
            self.data.append(data_in)
            self.label.append(label_in)
            self.seq_len.append(sentence_len)

            if self.line_count % 5000 == 0:
                logging.debug("Read {} lines".format(self.line_count))
        # logging.debug("OOV: {}".format(oov_set))
        logging.info("Read {} lines with OOV {:.3f}% ({}/{}), OOV(distinct) {:.3f}% ({}/{}) from '{}'"
              .format(self.line_count,
                      oov_count*100.0/word_count, oov_count, word_count,
                      len(oov_set) * 100.0 / len(word_set), len(oov_set), len(word_set),
                      data_path))

    def __getitem__(self, index):
        return self.data[index], self.label[index], self.seq_len[index]

    def __len__(self):
        """
        Return total number of data sample
        :return: number of samples
        """
        return self.line_count