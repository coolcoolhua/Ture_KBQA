from EntityLinking import Config
from prepare.Utils import clean_sentence
import math
from prepare.Utils import load_mid_name_map
from prepare import Log
import logging


class EntityInfo(object):

    def __init__(self):
        self.__node_list = []
        self.__mid_set = set()

    def add(self, mid, n_gram, n, name_cleaned, name_split_num):
        if mid in self.__mid_set:
            return
        gram_count = name_cleaned.count(n_gram)
        if gram_count <= 0:
            raise Exception("Error: Invalid '{}' in '{}'".format(n_gram, name_cleaned))
        tf = float(gram_count) / (name_split_num - n + 1)
        self.__node_list.append((mid, tf))
        self.__mid_set.add(mid)

    def __len__(self):
        return len(self.__node_list)

    def calculate_idf(self, total_doc):
        doc_freq = len(self.__node_list)
        self.__idf = math.log((float(total_doc) + 1.0) / (float(doc_freq) + 1.0)) + 1.0
        if self.__idf < 0:
            raise Exception("Error: Invalid '{}' / '{}'".format(total_doc, doc_freq))

    def get_nodes(self):
        ls = []
        for node in self.__node_list:
            ls.append((node[0], node[1] * self.__idf))
        return ls


class EntityIndex(object):

    """
    EntityIndex:
        __n_gram_entities {
              gram_n
           ->
              EntityInfo:  __node_list [ (mid, __tf), ... ],  __idf
        }
    """

    def __init__(self, file_path):
        self.__1_gram_entities = {}
        self.__2_gram_entities = {}
        self.__3_gram_entities = {}
        self.__n_gram_entities = {}
        self.__doc_count = 0
        self.__warning_count = 0

        self.mid_names_map = load_mid_name_map(file_path)
        logging.info("Start to process maps ...")
        count = 0
        for mid, name_strs in self.mid_names_map.items():
            count += 1
            self.__add(mid, name_strs)
            if count % 500000 == 0:
                logging.debug("Processed {} maps ...".format(count))
        logging.info("Processed {} maps successfully!".format(count))

        # Calculate idf
        logging.info("Start to process TF-IDF ...")
        logging.debug("Start to process TF-IDF (1-gram: {}) ...".format(len(self.__1_gram_entities)))
        for entity in self.__1_gram_entities.values():
            entity.calculate_idf(self.__doc_count)
        logging.debug("Start to process TF-IDF (2-gram: {}) ...".format(len(self.__2_gram_entities)))
        for entity in self.__2_gram_entities.values():
            entity.calculate_idf(self.__doc_count)
        logging.debug("Start to process TF-IDF (3-gram: {}) ...".format(len(self.__3_gram_entities)))
        for entity in self.__3_gram_entities.values():
            entity.calculate_idf(self.__doc_count)
        logging.debug("Start to process TF-IDF (n-gram: {}) ...".format(len(self.__n_gram_entities)))
        for entity in self.__n_gram_entities.values():
            entity.calculate_idf(self.__doc_count)
        logging.info("File processed ({} names with {} warning names) successfully!"
                     .format(self.__doc_count, self.__warning_count))

    def __add(self, mid, name_strs):
        for name_str in name_strs:
            # Clean and Lower
            name_cleaned = clean_sentence(name_str).lower()
            name_splits = name_cleaned.split(" ")
            name_splits = [name.strip() for name in name_splits]
            name_split_num = len(name_splits)
            if name_split_num < 1 or name_splits[0].strip() == '':
                # logging.warning("Failed to convert the name '{}' in \"<{}> : <{}>\"".format(name_str, mid, name_strs))
                self.__warning_count += 1
                continue

            self.__doc_count += 1

            # n-gram
            gram_n = name_cleaned
            if gram_n not in self.__n_gram_entities:
                self.__n_gram_entities[gram_n] = EntityInfo()
            self.__n_gram_entities[gram_n].add(mid, name_cleaned, name_split_num, name_cleaned, name_split_num)

            # 1-gram
            if name_split_num > 1:
                for gram1 in name_splits:
                    if gram1 not in self.__1_gram_entities:
                        self.__1_gram_entities[gram1] = EntityInfo()
                    self.__1_gram_entities[gram1].add(mid, gram1, 1, name_cleaned, name_split_num)

            # 2-gram
            if name_split_num > 2:
                for idx in range(name_split_num - 1):
                    gram2 = name_splits[idx] + ' ' + name_splits[idx+1]
                    if gram2 not in self.__2_gram_entities:
                        self.__2_gram_entities[gram2] = EntityInfo()
                    self.__2_gram_entities[gram2].add(mid, gram2, 2, name_cleaned, name_split_num)

            # 3-gram
            if name_split_num > 3:
                for idx in range(name_split_num - 2):
                    gram3 = name_splits[idx] + ' ' + name_splits[idx+1] + ' ' + name_splits[idx+2]
                    if gram3 not in self.__3_gram_entities:
                        self.__3_gram_entities[gram3] = EntityInfo()
                    self.__3_gram_entities[gram3].add(mid, gram3, 3, name_cleaned, name_split_num)

    def get_candidate_nodes(self, entity_text, extract_features=False):
        # element: (mid, tf_idf)
        candidate_nodes = []
        candidate_mid_map = {}

        # Clean and Lower
        name_cleaned = clean_sentence(entity_text).lower()
        name_splits = name_cleaned.split(" ")
        name_splits = [name.strip() for name in name_splits]
        name_split_num = len(name_splits)

        # Try exact match
        exact_match = 0
        if name_cleaned in self.__n_gram_entities:
            for mid, tf_idf in self.__n_gram_entities[name_cleaned].get_nodes():
                if mid not in candidate_mid_map or tf_idf > candidate_mid_map[mid]:
                    candidate_mid_map[mid] = tf_idf
            exact_match = 1

        # Check if find entity
        if len(candidate_mid_map) > 0:
            for mid, score in candidate_mid_map.items():
                candidate_nodes.append((mid, score))
            candidate_nodes.sort(key=lambda x: x[1], reverse=True)
            if extract_features:
                return candidate_nodes, exact_match, 0
            return candidate_nodes

        # 3-gram
        if name_split_num >= 3:
            for idx in range(name_split_num - 2):
                gram3 = name_splits[idx] + ' ' + name_splits[idx + 1] + ' ' + name_splits[idx + 2]
                if gram3 in self.__3_gram_entities:
                    for mid, tf_idf in self.__3_gram_entities[gram3].get_nodes():
                        if mid not in candidate_mid_map or tf_idf > candidate_mid_map[mid]:
                            candidate_mid_map[mid] = tf_idf

        if len(candidate_mid_map) > 0 and 2 <= name_split_num:
            for mid, score in candidate_mid_map.items():
                candidate_nodes.append((mid, score))
            candidate_nodes.sort(key=lambda x: x[1], reverse=True)
            if extract_features:
                return candidate_nodes, exact_match, 3
            return candidate_nodes

        # 2-gram
        if name_split_num >= 2:
            for idx in range(name_split_num - 1):
                gram2 = name_splits[idx] + ' ' + name_splits[idx + 1]
                if gram2 in self.__2_gram_entities:
                    for mid, tf_idf in self.__2_gram_entities[gram2].get_nodes():
                        if mid not in candidate_mid_map or tf_idf > candidate_mid_map[mid]:
                            candidate_mid_map[mid] = tf_idf

        if len(candidate_mid_map) > 0 and 1 <= name_split_num:
            for mid, score in candidate_mid_map.items():
                candidate_nodes.append((mid, score))
            candidate_nodes.sort(key=lambda x: x[1], reverse=True)
            if extract_features:
                return candidate_nodes, exact_match, 2
            return candidate_nodes

        # 1-gram
        if name_split_num >= 1:
            for gram1 in name_splits:
                if gram1 in self.__1_gram_entities:
                    for mid, tf_idf in self.__1_gram_entities[gram1].get_nodes():
                        if mid not in candidate_mid_map or tf_idf > candidate_mid_map[mid]:
                            candidate_mid_map[mid] = tf_idf

        # sort by score
        for mid, score in candidate_mid_map.items():
            candidate_nodes.append((mid, score))
        candidate_nodes.sort(key=lambda x: x[1], reverse=True)

        if extract_features:
            if len(candidate_nodes) > 0:
                return candidate_nodes, exact_match, 1
            return candidate_nodes, exact_match, 0
        return candidate_nodes

    def get_mid_name_map(self):
        return self.mid_names_map


if __name__ == '__main__':
    map_file = Config.map_file
    entity_index = EntityIndex(map_file)