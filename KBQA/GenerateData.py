import os
from prepare import Log
import logging
import KBQA.Config as Config

if __name__ == '__main__':
    data = {}

    # load data from entity detection
    for data_name in Config.ED_data_names:
        file_path = os.path.join(Config.ED_file_base_dir, data_name)
        line_count = 0
        line_error = 0
        logging.info("Start to read '{}' ...".format(data_name))
        for line in open(file_path):
            line_count += 1
            line = line.strip()
            splits = line.split('\t')
            splits = [v.strip() for v in splits]
            try:
                if len(splits) != 2 or (splits[0] in data and data[splits[0]][0] != splits[1]):
                    raise Exception("Invalid data sample '{}' at line {}".format(line, line_count))
                data[splits[0]] = [splits[1], []]
            except Exception as e:
                logging.debug(str(e))
                line_error += 1
        logging.info("{} : Error {} lines in all {} ({:.6f}%)".format(data_name, line_error, line_count,
                                                                          line_error * 100.0 / line_count))

    # load data from relation prediction
    for data_name in Config.RP_data_names:
        file_path = os.path.join(Config.RP_file_base_dir, data_name)
        line_count = 0
        line_error = 0
        logging.info("Start to read '{}' ...".format(data_name))
        for line in open(file_path):
            line_count += 1
            line = line.strip()
            splits = line.split('\t')
            splits = [v.strip() for v in splits]
            try:
                if len(splits) != 2:
                    raise Warning("Invalid data sample '{}' at line {}".format(line, line_count))
                if splits[0] in data:
                    if len(data[splits[0]]) != 2:
                        raise Exception("Invalid data sample '{}' at line {}".format(line, line_count))
                    if splits[1] not in data[splits[0]][1]:
                        data[splits[0]][1].append(splits[1])
            except Warning as e:
                # logging.debug(str(e))
                line_error += 1
        logging.info("{} : Error {} lines in all {} ({:.6f}%)".format(data_name, line_error, line_count,
                                                                          line_error * 100.0 / line_count))

    # Write data
    out_file = open(os.path.join(Config.output_base_dir, Config.output_data_name), 'w')
    sample_count = 0
    for question, ls in data.items():
        if len(ls) != 2:
            raise Exception("Invalid question '{}' with value {}".format(question, ls))

        if len(ls[1]) == 0:
            continue

        relations = ""
        for relation in ls[1]:
            relations += str(relation) + " "
        relations = relations.strip()
        out_file.write(question + '\t' + ls[0] + '\t' + relations + '\n')
        sample_count += 1
    out_file.close()
    logging.info("Write {} samples to file {} successfully!".format(sample_count, Config.output_data_name))
