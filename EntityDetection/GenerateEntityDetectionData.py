import os
from prepare.Utils import convert_single_mid_url_to_simplified_format, clean_sentence
from EntityDetection import Config
from prepare.Utils import load_mid_name_map
from prepare import Log
import logging


def format_data_label(question, subject_mid, mid_to_names):
    """
    Generate label for given question
        Mark the entities (words) in question: Entity is marked as 1, others are marked as 0
    :param question:
    :param subject_mid:
    :param mid_to_names: Dictionary,  Map(mid -> name)
    :return:
    """
    subjects = mid_to_names[subject_mid]

    question_cleaned = clean_sentence(question)
    question_lower = question_cleaned.lower()
    subjects_lower = [clean_sentence(subject).lower() for subject in subjects]

    # label
    question_splits = question_lower.split(' ')
    label = [0 for _ in range(len(question_splits))]

    has_entity = False
    maybe_entity_set = []
    maybe_entity_set.extend(subjects_lower)
    # Assume subject always appear in question, and object is answer

    for entity_set in maybe_entity_set:
        entity_set = entity_set.strip()
        entity_set_splits = entity_set.split(' ')
        for start_idx in range(len(question_splits)):
            if question_splits[start_idx] == entity_set_splits[0]:
                has_entity = True
                same_len = 1
                for entity_idx in range(1, len(entity_set_splits)):
                    if start_idx + entity_idx >= len(question_splits) \
                            or entity_set_splits[entity_idx] != question_splits[start_idx + entity_idx]:
                        break
                    same_len += 1

                if len(entity_set_splits) == 1 or same_len > 1:
                    for i in range(same_len):
                        label[start_idx + i] = 1

    if not has_entity:
        raise Exception("Cannot find entity in '{}' with subject {}: '{}'"
                        .format(question, subject_mid, subjects))

    return question_cleaned, label


def generate_datasets(mid_to_names, data_base_dir, data_names, output_base_dir, output_names, unknown_mids_file_name):
    """
    Convert simple questions dataset to dataset for entity detection
    :param mid_to_names:  Map(mid -> name)
    :param data_base_dir:  Basic directory of simple questions dataset
    :param data_names:  Train / valid / test dataset name
    :param output_base_dir:  Basic directory of output dataset for entity detection
    :param output_names:  Train / valid / test dataset name for entity detection
    :param unknown_mids_file_name: File name to save unknown mids
    :return:
    """
    assert isinstance(data_names, list) and isinstance(output_names, list) and len(data_names) == len(output_names) == 3
    unknown_mids = set()

    #  Read train / valid / test simple questions dataset, generate dataset for relation prediction
    logging.info("Start to generate datasets ...")
    for idx, data_name in enumerate(data_names):
        data_file = os.path.join(data_base_dir, data_name)
        output_file_path = os.path.join(output_base_dir, output_names[idx])
        out_file = open(output_file_path, 'w')
        line_count = 0
        line_error = 0
        logging.info("Start to read '{}' to generate '{}' ...".format(data_name, output_names[idx]))
        for line in open(data_file):
            line_count += 1
            splits = line.split('\t')
            try:
                if len(splits) != 4:
                    raise Exception("Invalid question '{}' at line {}".format(line, line_count))
                subject_mid = convert_single_mid_url_to_simplified_format(splits[0].strip())
                object_mid = convert_single_mid_url_to_simplified_format(splits[2].strip())
                question = splits[3].strip()
                if not mid_to_names.__contains__(subject_mid):
                    unknown_mids.add(subject_mid)
                    raise Exception("Cannot find subject mid '{}' when process '{}' at line {}".format(subject_mid, question, line_count))
                if not mid_to_names.__contains__(object_mid):
                    unknown_mids.add(object_mid)
                question_cleaned, label = format_data_label(question, subject_mid, mid_to_names)

                label_str = str(label[0])
                for label_idx in range(1, len(label)):
                    label_str += ' ' + str(label[label_idx])
                out_file.write(question_cleaned + '\t' + label_str + '\n')
            except Exception as e:
                # logging.debug(str(e))
                line_error += 1
        logging.info("{} : Error {} lines in all {} ({:.6f}%)".format(data_name, line_error, line_count, line_error*100.0/line_count))
        out_file.close()
        logging.info("'{}' is generated successfully!".format(output_names[idx]))
    logging.info("Datasets are generated successfully!")

    # Save unknown mids
    logging.info("Start to save unknown mids ...")
    unknown_file = os.path.join(out_file_base_dir, unknown_mids_file_name)
    with open(unknown_file, 'w') as f:
        for mid in unknown_mids:
            f.write(mid + '\n')
    logging.info("Unknown mids are saved successfully!")


if __name__ == '__main__':
    # Load map file get mid name map
    map_file = Config.map_file

    mid_to_names = load_mid_name_map(map_file)

    # Generate datasets
    question_data_base_dir = Config.question_data_base_dir
    question_data_name = Config.question_data_names
    out_file_base_dir = Config.out_file_base_dir
    output_data_names = Config.output_data_names
    unknown_mids_file_name = Config.unknown_mids_file_name

    generate_datasets(
        mid_to_names=mid_to_names,
        data_base_dir=question_data_base_dir,
        data_names=question_data_name,
        output_base_dir=out_file_base_dir,
        output_names=output_data_names,
        unknown_mids_file_name=unknown_mids_file_name
    )