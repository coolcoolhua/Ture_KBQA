from EntityDetection.TrainModel import load_model as ED_load_model
from RelationPrediction.TrainModel import load_model as RP_load_model
from EntityLinking.InvertedIndexOfEntity import EntityIndex
from AnswerSelection.ReachIndexOfEntity import ReachIndex
from prepare.Utils import load_relation_list_file, clean_sentence
import os
import EntityLinking.Config
import AnswerSelection.Config
import RelationPrediction.Config
from KBQA import Config
from prepare import Log
import logging


class KBQA(object):

    def __init__(self, entity_detection_model_path, relation_prediction_model_path, map_file, freebase_subset_file, relation_list_path):
        self.debug = False

        # load model of entity detection
        self.entity_detection_model = ED_load_model(entity_detection_model_path)

        # load model of relation prediction
        self.relation_prediction_model = RP_load_model(relation_prediction_model_path)

        # load entity index
        self.entity_index = EntityIndex(map_file)
        self.mid_map = self.entity_index.get_mid_name_map()

        # load relation list file
        self.relation_list = load_relation_list_file(relation_list_path)

        # load reach index
        self.reach_index = ReachIndex(freebase_subset_file)

    def check(self, question, entity_label, relation_indexes, TOP_K=None):
        """
        Check whether given question can be answered correctly or not
            A question is counted as correct if and only if the entity we select
        and the relation we predict (i.e, r) match the ground truth.

        Return correctness and cause of error (by entity detection or relation prediction)
        :param question: Raw question in natural language
        :param entity_label: Actual entity labels
        :param relation_indexes: Actual relations
        :param TOP_K: Consider top K of relations
        :return: (isCorrect, causedByEntityDetection)
        """
        _, entity_mark = self.entity_detection_model.predict(question)
        if entity_mark != entity_label:
            return False, True

        if TOP_K is not None:
            relation_ids, _ = self.relation_prediction_model.predict(question, TOP_K=TOP_K)
        else:
            relation_ids, _ = self.relation_prediction_model.predict(question)

        for relation_idx in relation_ids:
            if relation_idx in relation_indexes:
                return True, None
        return False

    def get_answer(self, question):
        """
        Given natural language question, return candidate answers
        :param question: in natural language
        :return: candidate answers
        """
        entity, _ = self.entity_detection_model.predict(question)
        if self.debug:
            print("[DEBUG] Entity detected in ED: '{}''".format(entity))

        relation_ids, relation_prob = self.relation_prediction_model.predict(question)
        relations = [self.relation_list[idx] for idx in relation_ids]

        if self.debug:
            print("[DEBUG] Relation predicted in ED: '{}' with ids '{}' and probabilities '{}'"
                  .format(relations, relation_ids, relation_prob))

        candidate_node_set = self.entity_index.get_candidate_nodes(entity)
        if self.debug:
            print("[DEBUG]    Entity '{}' with candidate node {}".format(entity, candidate_node_set))

        answers = {}
        for idx, relation in enumerate(relations):
            candidate_answers = self.reach_index.get_candidate_answers(candidate_node_set, relation,
                                                                       relation_prob[idx])
            if self.debug:
                print("[DEBUG]    Entity '{}' with candidate answers {}".format(entity, candidate_answers))

            for mid_reached, score in candidate_answers:
                if mid_reached not in answers or score > answers[mid_reached]:
                    answers[mid_reached] = score

        result = []
        for mid, score in answers.items():
            if mid in self.mid_map:
                names = self.mid_map[mid]
                if names is not None and len(names) > 0:
                    result.append((mid, self.mid_map[mid][0], score))
                else:
                    result.append((mid, "", score))
            else:
                result.append((mid, "", score))
        result.sort(key=lambda x: x[2], reverse=True)
        return result

    def enable_debug(self):
        self.debug = True

    def disable_debug(self):
        self.debug = False


def evaluation_system(kbqa_system, data_path, TOP_K=None):
    line_count = 0
    line_error = 0
    logging.info("Start to test KB-QA system on file '{}' ...".format(data_path))
    sample_correct = 0
    sample_all = 0
    sample_error = 0
    sample_error_by_ED = 0
    for line in open(data_path):
        line_count += 1
        line = line.strip()
        splits = line.split('\t')
        splits = [v.strip() for v in splits]
        try:
            if len(splits) != 3:
                raise Warning("Invalid data sample '{}' at line {}".format(line, line_count))
            question = splits[0].strip()
            labels = splits[1].strip().split(" ")
            labels = [int(mark) for mark in labels]
            relation_indexes = splits[2].strip().split(" ")
            relation_indexes = [int(idx) for idx in relation_indexes]

            isCorrect, causedByEntityDetection = kbqa_system.check(question, labels, relation_indexes, TOP_K=TOP_K)
            if isCorrect:
                sample_correct += 1
            else:
                sample_error += 1
                if causedByEntityDetection:
                    sample_error_by_ED += 1

            sample_all += 1
        except Warning as e:
            # logging.debug(str(e))
            line_error += 1

        if line_count % 2000 == 0:
            logging.debug("    Processed {} lines ({:.3f}%)..."
                          .format(line_count, sample_correct * 100.0 / sample_all))

    accuracy = sample_correct * 100.0 / sample_all
    logging.info("The accuracy of system on {} samples: {:.3f}% ({}/{})"
                 .format(line_count, accuracy, sample_correct, sample_all))
    logging.info("    Error caused by Entity Detection: {:.3f}% ({}/{})"
                 .format(sample_error_by_ED * 100.0 / sample_error, sample_error_by_ED, sample_error))
    logging.info("    Error caused by Relation Prediction: {:.3f}% ({}/{})"
                 .format(100 - sample_error_by_ED * 100.0 / sample_error, sample_error - sample_error_by_ED, sample_error))


if __name__ == '__main__':
    entity_detection_model_path = "/Users/jiecxy/PycharmProjects/ture/EntityDetection/model/Snapshot_Epoch-11.pt"
    relation_prediction_model_path = "/Users/jiecxy/PycharmProjects/ture/RelationPrediction/model/Snapshot_Epoch-3.pt"

    # Parameters for KB-QA system
    map_file = EntityLinking.Config.map_file
    freebase_subset_file = AnswerSelection.Config.freebase_subset_file
    relation_list_path = os.path.join(RelationPrediction.Config.out_file_base_dir, RelationPrediction.Config.relation_list_file_name)

    # Start KB-QA system
    kbqa = KBQA(entity_detection_model_path, relation_prediction_model_path, map_file, freebase_subset_file, relation_list_path)

    # Test KB-QA system
    data_path = os.path.join(Config.output_base_dir, Config.output_data_name)
    # evaluation_system(kbqa, data_path)