import torch
import os
import torch.utils.data as Data
from EntityDetection.DataLoader import EntityDetectionDataset
import prepare.Utils as Utils
from EntityDetection.BiLSTM import BiLSTM
import numpy as np
from EntityDetection import Config
from prepare.Utils import clean_sentence
import logging
import time


def prepare_vectors(bin_file_path_base, bin_file_name, padding_idx):
    """
    Get word vectors from binary word2vec file
    :param bin_file_path_base: base directory of binary file
    :param bin_file_name: e.g. `GoogleNews-vectors-negative300.bin`
    :return: padding_idx, oov_idx, word2vec
    """
    word2vec = Utils.load_word2vec_bin(os.path.join(bin_file_path_base, bin_file_name))
    dim = word2vec.vectors.shape[1]
    oov_idx = word2vec.vectors.shape[0] - 1
    word2vec.vectors[padding_idx] = np.zeros((1, dim))
    word2vec.vectors[oov_idx] = np.random.rand(1, dim)
    return oov_idx, word2vec


def convert_list_to_word2idx(word_list, padding_idx, oov_idx):
    """
    Convert list to dictionary (map): word -> index
    :param word_list: list of words
    :param padding_idx: index used for padding in sequence
    :param oov_idx: index used for words which are Out of Vocabulary
    :return: dictionary of word to index
    """
    word2idx = {}
    for idx, word in enumerate(word_list):
        if idx != padding_idx and idx != oov_idx:
            word2idx[word] = idx
    return word2idx


def get_data_loader(data_base_dir, data_names, word2idx, oov_idx, padding_idx, padding_label, SEQUENCE_LEN, BATCH_SIZE):
    """
    Generate loaders for train data, valid data and test data
    :param data_base_dir: base directory of datasets
    :param data_names: train / valid / test file names, e.g. 'data_train.txt', 'data_test.txt', 'data_valid.txt'
    :param word2idx: dictionary of word to index
    :param oov_idx: index used for words which are Out of Vocabulary
    :param padding_idx: index used for padding in sequence
    :param padding_label: label value used for label sequence
    :param SEQUENCE_LEN: Max length of sequence
    :param BATCH_SIZE: batch size
    :return: loaders for train, valid, test
    """
    loaders = []
    for file_name in data_names:
        path = os.path.join(data_base_dir, file_name)
        dataset = EntityDetectionDataset(
            data_path=path,
            word2idx=word2idx,
            oov_idx=oov_idx,
            padding_idx=padding_idx,
            padding_label=padding_label,
            SEQUENCE_LEN=SEQUENCE_LEN
        )
        loader = Data.DataLoader(
            dataset=dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2
        )
        loaders.append(loader)
    return loaders


def build_network(pretrained_weight, padding_idx, padding_label, output_dim, BATCH_SIZE,
                  lstm_hidden_size, learning_rate):

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_func = torch.nn.CrossEntropyLoss(ignore_index=padding_label, reduction='sum')
    # loss_func.to(device)

    model = BiLSTM(
        pretrained_weight=pretrained_weight,
        padding_idx=padding_idx,
        output_dim=output_dim,
        batch_size=BATCH_SIZE,
        lstm_hidden_size=lstm_hidden_size,
        loss_func=loss_func
    )
    # model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    return model, optimizer


def calculate_accuracy(prediction, batch_y, batch_len):
    all_word = torch.sum(batch_len)
    all_sentence = batch_len.size(0)
    correct_words = 0
    correct_sentences = 0
    for idx, seq_len in enumerate(batch_len):
        cur_correct = (torch.max(prediction[idx][:seq_len], 1)[1] == batch_y[idx][:seq_len]).sum()
        if cur_correct == seq_len:
            correct_sentences += 1
        correct_words += cur_correct
    accuracy_word = float(correct_words) * 100.0 / all_word
    accuracy_sentence = float(correct_sentences) * 100.0 / all_sentence
    return (accuracy_word, correct_words, all_word), (accuracy_sentence, correct_sentences, all_sentence)


def load_model_path(save_dir, epoch_num, model):
    model_path = '{}_Epoch-{}.pt'.format(os.path.join(save_dir, 'Snapshot'), epoch_num)
    model.load_state_dict(torch.load(model_path))
    return model


def save_model(save_dir, epoch_num, model):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    save_prefix = os.path.join(save_dir, 'Snapshot')
    # Remove last model
    last_model = '{}_Epoch-{}.pt'.format(save_prefix, epoch_num - 1)
    if os.path.isfile(last_model):
        os.remove(last_model)

    save_path = '{}_Epoch-{}.pt'.format(save_prefix, epoch_num)
    torch.save(model.state_dict(), save_path)


def evaluate_model(model, loader):
    loss_sum, word_accuracy, sentence_accuracy = 0., 0., 0.
    for step, (batch_x, batch_y, batch_len) in enumerate(loader):
        prediction, loss = model.step(batch_x, batch_y, batch_len)
        loss_sum += loss
        accuracy_word, accuracy_sentence = calculate_accuracy(prediction, batch_y, batch_len)
        word_accuracy += accuracy_word[0]
        sentence_accuracy += accuracy_sentence[0]
        loss += loss
    return loss_sum / len(loader), word_accuracy / len(loader), sentence_accuracy / len(loader)


def train(train_loader, valid_loader, EPOCH_MAX, optimizer, model, model_save_dir):
    model_count = 0
    logging.info("Start training on {} samples, validating on {} samples...".format(
        len(train_loader.dataset), len(valid_loader.dataset)
    ))
    last_valid_sentence_accuracy = 0
    for epoch_count in range(EPOCH_MAX):
        epoch_count += 1

        start_time = time.time()
        train_loss, train_word_accuracy, train_sentence_accuracy = 0., 0., 0.
        for step, (batch_x, batch_y, batch_len) in enumerate(train_loader):
            # batch_len: (batch,)
            optimizer.zero_grad()

            prediction, loss = model.step(batch_x, batch_y, batch_len)
            accuracy_word, accuracy_sentence = calculate_accuracy(prediction, batch_y, batch_len)
            train_word_accuracy += accuracy_word[0]
            train_sentence_accuracy += accuracy_sentence[0]
            train_loss += float(loss)

            loss.backward()  # back propagation, compute gradients
            optimizer.step()  # apply gradients

        valid_loss, valid_word_accuracy, valid_sentence_accuracy = evaluate_model(model, valid_loader)
        train_len = len(train_loader)
        end_time = time.time()
        logging.info("[Epoch {}] ({:.3f} s) loss: {:.6f} - acc(word): {:.3f}%  acc(sentence): {:.3f}%  |  "
                     "valid_loss: {:.6f} - valid_acc(word): {:.3f}%  valid_acc(sentence): {:.3f}%".format(
            epoch_count,
            end_time - start_time,
            train_loss / train_len,
            train_word_accuracy / train_len,
            train_sentence_accuracy / train_len,
            valid_loss,
            valid_word_accuracy,
            valid_sentence_accuracy
        ))

        if valid_sentence_accuracy < last_valid_sentence_accuracy:
            return load_model_path(model_save_dir, epoch_count - 1, model)

        last_valid_sentence_accuracy = valid_sentence_accuracy
        save_model(model_save_dir, epoch_count, model)
        model_count += 1
    return model


def load_model(model_path):
    logging.info("======== 1. Load word2vec ========")
    oov_idx, word2vec = prepare_vectors(bin_file_path_base, bin_file_name, padding_idx)
    logging.info("==== word2vec loaded!")

    logging.info("======== 2. Load model ========")
    word2idx = convert_list_to_word2idx(word2vec.index2entity, padding_idx, oov_idx)

    # Load network
    model, _ = build_network(
        pretrained_weight=word2vec.vectors,
        padding_idx=padding_idx,
        padding_label=padding_label,
        output_dim=output_dim,
        BATCH_SIZE=BATCH_SIZE,
        lstm_hidden_size=lstm_hidden_size,
        learning_rate=learning_rate
    )
    model.load_state_dict(torch.load(model_path))
    logging.info("==== model loaded!")

    model.SEQUENCE_LEN = SEQUENCE_LEN
    model.padding_idx = padding_idx
    model.oov_idx = oov_idx
    model.word2idx = word2idx
    return model


def train_model():
    logging.info("======== 1. Load word2vec ========")
    oov_idx, word2vec = prepare_vectors(bin_file_path_base, bin_file_name, padding_idx)
    logging.info("==== word2vec loaded!")

    logging.info("======== 2. Load train/valid/test datasets ========")
    word2idx = convert_list_to_word2idx(word2vec.index2entity, padding_idx, oov_idx)
    train_loader, valid_loader, test_loader = get_data_loader(
        data_base_dir=data_base_dir,
        data_names=data_names,
        word2idx=word2idx,
        oov_idx=oov_idx,
        padding_idx=padding_idx,
        padding_label=padding_label,
        SEQUENCE_LEN=SEQUENCE_LEN,
        BATCH_SIZE=BATCH_SIZE
    )
    logging.info("==== datasets loaded!")

    # Build network
    model, optimizer = build_network(
        pretrained_weight=word2vec.vectors,
        padding_idx=padding_idx,
        padding_label=padding_label,
        output_dim=output_dim,
        BATCH_SIZE=BATCH_SIZE,
        lstm_hidden_size=lstm_hidden_size,
        learning_rate=learning_rate
    )

    logging.info("======== 3. Start training ========")
    model = train(
        train_loader=train_loader,
        valid_loader=valid_loader,
        EPOCH_MAX=EPOCH_MAX,
        optimizer=optimizer,
        model=model,
        model_save_dir=model_save_dir
    )
    model.SEQUENCE_LEN = SEQUENCE_LEN
    model.padding_idx = padding_idx
    model.oov_idx = oov_idx
    model.word2idx = word2idx
    logging.info("==== training finished!")

    logging.info("======== 4. Start testing ========")
    logging.info("Start testing on {} samples...".format(len(test_loader.dataset)))
    test_loss, test_word_accuracy, test_sentence_accuracy = evaluate_model(model, test_loader)
    logging.info("[Test] loss: {:.6f} - acc(word): {:.3f}%  acc(sentence): {:.3f}%".format(
        test_loss,
        test_word_accuracy,
        test_sentence_accuracy
    ))
    logging.info("==== testing finished!")
    return model


# =========== Configuration ===========
padding_idx = 0

bin_file_path_base = Utils.bin_file_path_base
bin_file_name = Utils.bin_file_name

model_save_dir = Config.model_save_dir
output_dim = 2
lstm_hidden_size = 60
learning_rate = 0.005
padding_label = -1
EPOCH_MAX = 20
BATCH_SIZE = 24
SEQUENCE_LEN = 36

data_base_dir = Config.data_base_dir
data_names = Config.data_names
# =========== Configuration ===========


if __name__ == '__main__':
    # train model
    # TODO enable save
    # model = train_model()

    # load model
    model = load_model("/Users/jiecxy/PycharmProjects/ture/EntityDetection/model/Snapshot_Epoch-11.pt")
