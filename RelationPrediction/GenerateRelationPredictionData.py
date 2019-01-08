import os
from prepare.Utils import clean_sentence
from RelationPrediction import Config
from prepare.Utils import load_relation_list_file, convert_single_relation_url_to_simplified_format
from prepare import CommonConfig
from prepare import Log
import logging


def format_data_label(question, relation_index):
    return clean_sentence(question), relation_index


def convert_dataset_to_relation_list(data_base_dir, data_names, output_base_dir, relation_list_file_name):
    """
    read file `annotated_fb_data_*` format:
        <subject_mid>  <\t>  <relation>  <\t>  <object_mid>  <\t>  <question>
    :param data_base_dir map_file: base directory for datasets
    :param data_names: train / valid / test datasets
    :param save_path: path to save relation list (unique)
    :return: relation list
    """
    assert isinstance(data_names, list) and len(data_names) == 3

    relations = []
    for data_name in data_names:
        line_count = 0
        file = os.path.join(data_base_dir, data_name)
        for line in open(file):
            line_count += 1
            splits = line.split('\t')
            if len(splits) != 4 or '' == splits[1].strip():
                raise Exception("Invalid format: '{}' at line {} in file {}".format(line, line_count, data_name))
            relation = convert_single_relation_url_to_simplified_format(splits[1])
            if not relations.__contains__(relation):
                relations.append(relation)

    if len(relations) != CommonConfig.TOTAL_RELATION_TYPE:
        raise Exception("Invalid number of relation type, expected '{}' but got {}".format(CommonConfig.TOTAL_RELATION_TYPE, len(relations)))

    relations.sort()

    if relation_list_file_name:
        relation_list_file = os.path.join(output_base_dir, relation_list_file_name)
        with open(relation_list_file, 'w') as f:
            for relation in relations:
                f.write(relation + '\n')
        logging.info("Relation list file saved at '{}' !".format(relation_list_file))
    return relations


def generate_datasets(relation_list_file_name, data_base_dir, data_names, output_base_dir, output_names):
    """
    Convert simple questions dataset to dataset for relation prediction
        output format:
            <question> <\t> <relation index>
    :param data_base_dir map_file: base directory for datasets
    :param data_names: train / valid / test datasets
    :param relation_list_file: file storing relation list (must contains Config.TOTAL_RELATION_TYPE relations and sorted)
    :param output_base_dir: base directory to save dataset
    :param output_names: dataset names to output
    """
    assert isinstance(data_names, list) and isinstance(output_names, list) and len(data_names) == len(output_names) == 3

    logging.info("Starting to load relation list file ...")
    relations = load_relation_list_file(os.path.join(output_base_dir, relation_list_file_name))
    logging.info("Relation list file (num = {}) loaded successfully!".format(len(relations)))

    #  Read train / valid / test simple questions dataset, generate dataset for relation prediction
    logging.info("Starting to generate datasets ...")
    for idx, data_name in enumerate(data_names):
        data_file = os.path.join(data_base_dir, data_name)
        output_file_path = os.path.join(output_base_dir, output_names[idx])
        output_file = open(output_file_path, 'w')
        line_count = 0
        logging.info("Starting to read '{}' to generate '{}' ...".format(data_name, output_names[idx]))

        for line in open(data_file, 'r'):
            line_count += 1
            line = line.strip()
            splits = line.split('\t')
            if len(splits) != 4 or '' == splits[1].strip():
                raise Exception("Invalid format: '{}' at line {} in file {}".format(line, line_count, data_name))
            relation = convert_single_relation_url_to_simplified_format(splits[1])
            if relation not in relations:
                raise Exception("Failed to find relation '{}' (at line '{}' in file '{}') in relation list file '{}'"
                                .format(relation, line_count, data_name, relation_list_file_name))
            question_cleaned, relation_index = format_data_label(splits[3], relations.index(relation))
            output_file.write(question_cleaned + '\t' + str(relation_index) + '\n')
        output_file.close()
        logging.info("'{}' is generated successfully!".format(output_names[idx]))
    logging.info("Datasets are generated successfully!")


if __name__ == '__main__':
    # Save relation list file
    data_base_dir = Config.question_data_base_dir
    data_names = Config.question_data_names
    output_base_dir = Config.out_file_base_dir
    relation_list_file_name = Config.relation_list_file_name

    convert_dataset_to_relation_list(
        data_base_dir=data_base_dir,
        data_names=data_names,
        output_base_dir = output_base_dir,
        relation_list_file_name=relation_list_file_name
    )

    # Generate datasets
    output_names = Config.output_data_names

    generate_datasets(
        relation_list_file_name=relation_list_file_name,
        data_base_dir=data_base_dir,
        data_names=data_names,
        output_base_dir=output_base_dir,
        output_names=output_names
    )

    # cat annotated_fb_data_* | awk -F '\t' '{print $2}' | sort | uniq | wc -l