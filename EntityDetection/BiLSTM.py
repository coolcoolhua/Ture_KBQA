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


class BiLSTM(nn.Module):

    def __init__(self, pretrained_weight, padding_idx, output_dim, lstm_hidden_size, batch_size,
                 loss_func, lstm_num_layers=2, lstm_dropout=0.1):
        """
        Initialize the network: Embedding -> BiLSTM -> Linear -> SoftMax
        :param pretrained_weight: word2vec vectors
        :param padding_idx: index for padding element
        :param output_dim: dimension to output
        :param lstm_hidden_size: number of hidden unit in BiLSTM
        :param batch_size: batch size
        :param loss_func: loss function
        :param lstm_num_layers: number of layers in BiLSTM
        :param lstm_dropout: dropout ratio for BiLSTM
        """
        super(BiLSTM, self).__init__()

        self.lstm_num_layers = lstm_num_layers
        self.batch_size = batch_size
        self.loss_func = loss_func
        self.padding_idx = padding_idx

        self.embedding_layer = nn.Embedding.from_pretrained(torch.FloatTensor(pretrained_weight))
        self.embedding_layer.padding_idx = padding_idx

        self.lstm_hidden_size = lstm_hidden_size
        self.bilstm = nn.LSTM(input_size=pretrained_weight.shape[1],
                              hidden_size=self.lstm_hidden_size,
                              num_layers=self.lstm_num_layers,
                              dropout=lstm_dropout,
                              bidirectional=True,
                              batch_first=True)

        self.bi_num = 2  # bidirectional
        self.output_dim = output_dim
        self.linear_layer = nn.Sequential(
            nn.Linear(lstm_hidden_size * self.bi_num, self.output_dim)
        )
        self.soft_max = nn.Softmax(dim=-1)

    def init_hidden(self, batch_size):
        """
        Initial new hidden states
        :param batch_size: batch size
        :return: new state (random)
        """

        # the weights are of the form (nb_layers, batch_size, nb_lstm_units)
        # h_n: (num_layers * num_directions, batch, hidden_size)
        # c_n: (num_layers * num_directions, batch, hidden_size)
        h_n = torch.randn(self.lstm_num_layers*self.bi_num, batch_size, self.lstm_hidden_size)
        c_n = torch.randn(self.lstm_num_layers*self.bi_num, batch_size, self.lstm_hidden_size)

        h_n = Variable(h_n)
        c_n = Variable(c_n)

        return h_n, c_n

    def forward(self, sentences, seq_length):
        """
        Forward in network
        :param sentences: sequences (batch) to processed
        :param seq_length: lengths of each sequence
        :return: output for model
        """
        assert list(sentences.shape[:-1]) == list(seq_length.shape)

        batch_size = sentences.shape[0]
        self.hidden = self.init_hidden(batch_size)

        # sentences: (batch, seq_len)
        # embed: (batch, seq_len, embedding_dim)
        embed = self.embedding_layer(sentences)
        seq_length = seq_length.float()

        sorted_length, idx_sort = torch.sort(seq_length, descending=True)
        _, idx_unsort = torch.sort(idx_sort)
        sorted_length = sorted_length.int()
        embed = embed[idx_sort]

        # pack
        pack = nn.utils.rnn.pack_padded_sequence(embed, sorted_length, batch_first=True)

        # bilstm_out: (batch, seq_len, num_directions * hidden_size)
        bilstm_out, _ = self.bilstm(pack, self.hidden)

        # unpack
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(bilstm_out, batch_first=True)
        unpacked = unpacked[idx_unsort]

        # output: (batch, max(seq_len), output_dim)
        output = self.linear_layer(unpacked)
        return output

    def step(self, batch_x, batch_y, batch_len):
        """
        Forward and compute loss
        :param batch_x: batch data (sequences), (batch_size, seq_len)
        :param batch_y: batch label (class for each word in each sentence), (batch_size, seq_len)
        :param batch_len: lengths for sequences, (batch_size)
        :return: prediction and loss
        """
        batch_x = wrap(batch_x)
        batch_y = wrap(batch_y)
        max_len = batch_len.max().int().item()

        # prediction: (batch, max(seq_len), output_dim)
        prediction = self.forward(batch_x, batch_len)

        # === Method 1
        # prediction_transposed: (batch, output_dim, seq_len)
        # prediction_transposed = prediction.transpose(1, 2)
        # loss = loss_func(prediction_transposed, batch_y)

        # === Method 2
        # batch_y: (batch, seq_len, output_dim)
        # input: (N, C)
        # target: (N)
        loss = self.loss_func(prediction[:, :max_len, :].reshape(-1, self.output_dim), batch_y[:, :max_len].reshape(-1))
        return prediction, loss

    def predict(self, sentence, extract_feature=False):
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
                if word in self.word2idx:
                    data_in[0][idx] = self.word2idx[word]
                else:
                    data_in[0][idx] = self.oov_idx

        batch_x = wrap(data_in)
        batch_len = np.zeros(1)
        batch_len[0] = sentence_len
        batch_len = wrap(batch_len)

        prediction = self.soft_max(self.forward(batch_x, batch_len))
        prediction_max = prediction[0][:sentence_len].max(dim=1)
        entity_marks = prediction_max[1].numpy()
        entity_mark = list(entity_marks)
        entity_score = prediction_max[0].prod().item()
        entity_length = entity_marks.sum()

        assert len(entity_mark) == len(question_split)

        start_idx = -1
        end_idx = -1
        for idx, mark in enumerate(entity_mark):
            if mark == 1:
                if start_idx == -1:
                    start_idx = idx
                end_idx = idx

        entity = ""
        if start_idx != -1:
            for i in range(start_idx, end_idx + 1):
                entity += question_split[i] + " "
            entity = entity.strip()

        if extract_feature:
            return entity, entity_mark, (entity_length, entity_score)
        return entity, entity_mark
