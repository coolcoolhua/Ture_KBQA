import os
from prepare.Utils import convert_single_mid_url_to_simplified_format
from prepare import Log
import logging


data_dir = '/home/ubuntu/SimpleQuestions_v2'
data_files = ['annotated_fb_data_test.txt', 'annotated_fb_data_train.txt', 'annotated_fb_data_valid.txt']
mid_file_dir = '/home/ubuntu/test'
mid_file_name = 'fb_data_mid.txt'

for file in data_files:
    _file = os.path.join(data_dir, file)
    if not os.path.isfile(_file):
        raise Exception("ERROR to read data file '{}'".format(_file))
if not os.path.isdir(mid_file_dir):
    raise Exception("ERROR not a dir '{}'".format(mid_file_dir))


mid_set = set()
for file_name in data_files:
    file = os.path.join(data_dir, file_name)
    line_count = 0
    for line in open(file):
        splits = line.split('\t')
        try:
            if len(splits) != 4:
                raise Exception
            mid_set.add(convert_single_mid_url_to_simplified_format(splits[0]))
            mid_set.add(convert_single_mid_url_to_simplified_format(splits[2]))
        except Exception:
            logging.error("ERROR in line {}: '{}'".format(line_count, line))


mid_file = os.path.join(mid_file_dir, mid_file_name)
with open(mid_file, 'w') as f:
    for mid in sorted(mid_set):
        f.write(mid + '\n')

# split -l 2000 fb_data_mid.txt -d -a 3 input_