from AnswerSelection import Config
from prepare.Utils import convert_single_mid_url_to_simplified_format
from prepare.Utils import convert_mids_url_to_simplified_format
from prepare.Utils import convert_single_relation_url_to_simplified_format
from prepare import Log
import logging


class ReachIndex(object):
    # only consider about single hop away
    """
    ReachIndex:
        ___mid_reach {
              mid
           ->
              __node_list {
                  relation
                ->
                  set(mid)
              }
        }
    """

    def __init__(self, file_path):
        """
        Load fact data
        :param file_path: path of `freebase-FB5M.txt`
        """
        self.___mid_reach = {}

        logging.info("Start to process facts '{}' ...".format(file_path))
        line_count = 0
        warning_count = 0
        for line in open(file_path, 'r'):
            line_count += 1
            splits = line.strip().split('\t')
            try:
                if len(splits) != 3:
                    raise Exception("Failed to convert '{}' at line {}".format(line, line_count))
                mid = convert_single_mid_url_to_simplified_format(splits[0])
                relation = convert_single_relation_url_to_simplified_format(splits[1])
                mids_reached = convert_mids_url_to_simplified_format(splits[2])
                if mid not in self.___mid_reach:
                    self.___mid_reach[mid] = {}
                if relation not in self.___mid_reach[mid]:
                    self.___mid_reach[mid][relation] = set()
                for mid_reached in mids_reached:
                    self.___mid_reach[mid][relation].add(mid_reached)
            except Exception as e:
                warning_count += 1
                logging.warning("Failed to convert '{}': {}".format(line, str(e)))
            if line_count % 1000000 == 0:
                logging.debug("Processed {} facts ({} mids) ...".format(line_count, len(self.___mid_reach)))
        logging.info("Processed {} facts (with {} warning facts) with {} mids successfully!"
                     .format(line_count, warning_count, len(self.___mid_reach)))

    def __len__(self):
        return len(self.___mid_reach)

    def get_candidate_answers(self, candidate_nodes, relation, relation_prob, extract_features=False):
        """
        Given candidate nodes and predicted relation, return candidate answers (with tf-idf score)
        :param candidate_nodes: list of (mid, tf_idf)
        :param relation: relation predicted, e.g. `music.release_track.release`
        :param relation_prob: relation probability
        :return: candidate_answers: list of (mid, tf_idf)
        """
        candidate_answers = []
        candidate_mid_map = {}

        mid_idx = 0
        for mid, tf_idf in candidate_nodes:
            mid_idx += 1
            if mid in self.___mid_reach:
                # Only check single hop away
                reach_info_relation = self.___mid_reach[mid]
                if relation in reach_info_relation:
                    for mid_reached in reach_info_relation[relation]:
                        if mid_reached not in candidate_mid_map or tf_idf * relation_prob > candidate_mid_map[mid_reached][1]:
                            candidate_mid_map[mid_reached] = (mid_idx, tf_idf * relation_prob)

        for mid_reached, (mid_idx, score) in candidate_mid_map.items():
            if extract_features:
                candidate_answers.append((mid_reached, score, mid_idx))
            else:
                candidate_answers.append((mid_reached, score))
        candidate_answers.sort(key=lambda x: x[1], reverse=True)
        return candidate_answers


if __name__ == '__main__':
    freebase_subset_file = Config.freebase_subset_file
    reach_index = ReachIndex(freebase_subset_file)