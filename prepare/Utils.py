from gensim.models import KeyedVectors
import os
import re
from prepare import CommonConfig, Log
import logging


def convert_mids_url_to_simplified_format(urls):
    urls = urls.strip()
    ls = []
    for url in urls.split(' '):
        ls.append(convert_single_mid_url_to_simplified_format(url))
    return ls


def convert_single_mid_url_to_simplified_format(url):
    url = url.strip()
    splits = url.split('/')
    if len(splits) != 3 or splits[1] != 'm':
        raise Exception("Failed to convert mid: " + url)
    return 'm.' + splits[2].strip()


def convert_single_relation_url_to_simplified_format(relation_url):
    """
    Convert relation url string to formatted url
    :param relation_url: e.g.  www.freebase.com/people/person/place_of_birth
    :return: e.g.  people.person.place_of_birth
    """
    relation_url = relation_url.strip()
    prefix = 'www.freebase.com/'
    if not relation_url.startswith(prefix):
        raise Exception("Invalid format of relation '{}', expected prefix '{}'".format(relation_url, prefix))
    return relation_url[len(prefix):].replace('/', '.').strip()


def clean_sentence(sentence):
    clean = sentence
    clean = re.sub(r'\'s ', ' ', clean)
    clean = re.sub(r'\'d ', ' ', clean)
    clean = re.sub(r'\'m ', ' ', clean)
    clean = re.sub(r'[`~!@#$%^&*()+=\[\]{\}:;\'\"<>?,.\\|！￥（）—、【】；：，。？《》’]', '', clean)
    clean = re.sub(r'\s+', ' ', clean)
    clean = clean.strip()
    return clean


word2vec_maps = {}
def load_word2vec_bin(bin_file_path):
    if bin_file_path in word2vec_maps:
        return word2vec_maps[bin_file_path]
    vec = KeyedVectors.load_word2vec_format(bin_file_path, binary=True)
    word2vec_maps[bin_file_path] = vec
    return vec


def convert_word2vec_from_bin_to_txt(file_path, save_path):
    model = KeyedVectors.load_word2vec_format(file_path, binary=True)
    KeyedVectors.save_word2vec_format(model, save_path)


def convert():
    output_file_name = 'GoogleNews-vectors-negative300.txt'
    bin_file_path = os.path.join(bin_file_path_base, bin_file_name)
    txt_file_path = os.path.join(bin_file_path_base, output_file_name)
    convert_word2vec_from_bin_to_txt(bin_file_path, txt_file_path)


def load_mid_name_map_triple(map_file):
    # read file `out` format:
    #     <mid>  <\t>  <object.name>  <\t>  <alias>

    map = {}
    line_count = 0
    for line in open(map_file):
        line_count += 1
        names = []
        splits = line.split('\t')
        if len(splits) != 3 or splits[1].strip() == '' == splits[2].strip():
            logging.warning("WARNING: invalid format in '{}' at line {}".format(line, line_count))
            continue
        if splits[1].strip() != '':
            names.append(splits[1].strip())
        if splits[2].strip() != '':
            names.append(splits[2].strip())
        map[splits[0].strip()] = names
    return map


relation_maps = {}
def load_relation_list_file(relation_list_path):
    #  Read relation list from file
    if relation_list_path in relation_maps:
        return relation_maps[relation_list_path]

    relations = []

    last_relation = ''
    line_count = 0
    for relation in open(relation_list_path, 'r'):
        line_count += 1
        if not relation > last_relation:
            raise Exception("Relations should be sorted (Ascending order), "
                            "invalid relation '{}' at line '{}' in file '{}'"
                            .format(relation, line_count, relation_list_path))
        relations.append(relation.strip())
    if len(relations) != CommonConfig.TOTAL_RELATION_TYPE:
        raise Exception("Invalid number of relation type, expected '{}' but got {}"
                        .format(CommonConfig.TOTAL_RELATION_TYPE, len(relations)))
    relation_maps[relation_list_path] = relations
    return relations


mid_maps = {}
def load_mid_name_map(map_file):
    """
    read file `names.trimmed.5M.txt` format:
        <fb:mid>  <\t>  <fb:type.object/alias.name>  <\t>  <name>
    :param map_file: file
    :return: Map(mid -> names)
    """
    if map_file in mid_maps:
        return mid_maps[map_file]
    map = {}
    line_count = 0
    logging.info("Starting to load mid name map file... '{}'".format(map_file))
    for line in open(map_file):
        line_count += 1
        splits = line.split('\t')
        if len(splits) != 3 or '' == splits[2].strip():
            logging.warning("WARNING: invalid format in '{}' at line {}".format(line, line_count))
            continue
        name = splits[2].strip()
        key = splits[0].strip()[3:]
        if key not in map:
            map[key] = []
        if name not in map[key]:
            map[key].append(name)
        if line_count % 1000000 == 0:
            logging.debug("Read {} lines ...".format(line_count))
    logging.info("Mid name map file ({} lines) loaded successfully!".format(line_count))
    mid_maps[map_file] = map
    return map


bin_file_path_base = '/Users/jiecxy/Desktop/dissertation/word2vec'
bin_file_name = 'GoogleNews-vectors-negative300.bin'