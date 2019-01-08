import torch
import torch.nn as nn
from torch.autograd import Variable
from prepare.Utils import clean_sentence
import numpy as np


def wrap(seq):
    """
    Wrap data in Variable
    :param seq: data
    :return: Variable
    """
    return Variable(torch.LongTensor(seq))


class BiGRU(nn.Module):

    def __init__(self, pretrained_weight, padding_idx, output_dim, gru_hidden_size, batch_size,
                 loss_func, gru_num_layers=2, gru_dropout=0.1):
        """
        Initialize the network: Embedding -> BiGRU -> Linear -> SoftMax
        :param pretrained_weight: word2vec vectors
        :param padding_idx: index for padding element
        :param output_dim: dimension to output
        :param gru_hidden_size: number of hidden unit in BiGRU
        :param batch_size: batch size
        :param loss_func: loss function
        :param gru_num_layers: number of layers in BiGRU
        :param gru_dropout: dropout ratio for BiGRU
        """
        super(BiGRU, self).__init__()

        self.gru_num_layers = gru_num_layers
        self.batch_size = batch_size
        self.loss_func = loss_func

        self.embedding_layer = nn.Embedding.from_pretrained(torch.FloatTensor(pretrained_weight))
        self.embedding_layer.padding_idx = padding_idx

        self.gru_hidden_size = gru_hidden_size
        self.bigru = nn.GRU(input_size=pretrained_weight.shape[1],  # embedding dim
                            hidden_size=self.gru_hidden_size,
                            num_layers=self.gru_num_layers,
                            dropout=gru_dropout,
                            bidirectional=True,
                            batch_first=True)

        self.bi_num = 2  # bidirectional
        self.output_dim = output_dim
        self.linear_layer = nn.Sequential(
            nn.Linear(gru_hidden_size * self.bi_num, self.output_dim)
        )
        self.soft_max = nn.Softmax(dim=-1)

    def init_hidden(self, batch_size):
        """
        Initial new hidden states
        :param batch_size: batch size
        :return: new state (random)
        """

        # the weights are of the form (nb_layers, batch_size, nb_gru_units)
        # h_0: (num_layers * num_directions, batch, hidden_size)
        h_0 = torch.randn(self.gru_num_layers * self.bi_num, batch_size, self.gru_hidden_size)
        h_0 = Variable(h_0)
        return h_0

    def forward(self, sentences, seq_length):
        """
        Forward in network
        :param sentences: sequences (batch) to processed
        :param seq_length: lengths of each sequence
        :return: output for model
        """
        assert list(sentences.shape[:-1]) == list(seq_length.shape)

        # sentences: (batch, seq_len)
        # embed: (batch, seq_len, embedding_dim)
        embed = self.embedding_layer(sentences)

        # bigru_out: (batch, seq_len, num_directions * hidden_size)
        # hidden: (num_layers*num_directions, batch, hidden_size)
        _, hidden = self.bigru(embed)

        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)

        # output: (batch, output_dim)
        output = self.linear_layer(hidden)
        return output

    def step(self, batch_x, batch_y, batch_len):
        """
        Forward and compute loss
        :param batch_x: batch data (sequences), (batch_size, seq_len)
        :param batch_y: batch label (class for each sentence), (batch_size)
        :param batch_len: lengths for sequences, (batch_size)
        :return: prediction and loss
        """

        batch_x = wrap(batch_x)
        batch_y = wrap(batch_y)

        # prediction: (batch, output_dim)
        prediction = self.forward(batch_x, batch_len)

        # input: (N, C)
        # target: (N)
        loss = self.loss_func(prediction, batch_y)

        return prediction, loss

    def predict(self, sentence, TOP_K=3):
        sentence_cleaned = clean_sentence(sentence)
        question_split = sentence_cleaned.split(' ')
        data_in = np.zeros((1, self.SEQUENCE_LEN), dtype=np.long)
        sentence_len = 0
        for idx in range(self.SEQUENCE_LEN):
            if idx >= len(question_split):
                # add padding
                data_in[0][idx] = self.padding_idx
            else:
                # add word
                sentence_len += 1
                word = question_split[idx].strip()
                if self.word2idx.__contains__(word):
                    data_in[0][idx] = self.word2idx[word]
                else:
                    data_in[0][idx] = self.oov_idx

        batch_x = wrap(data_in)
        batch_len = np.zeros(1)
        batch_len[0] = sentence_len
        batch_len = wrap(batch_len)

        prediction = self.soft_max(self.forward(batch_x, batch_len))
        sorted = prediction[0].sort(dim=0, descending=True)
        sorted_top_k_prob = list(sorted[0].data.numpy()[:TOP_K])
        sorted_top_k = list(sorted[1].numpy()[:TOP_K])

        return sorted_top_k, sorted_top_k_prob
