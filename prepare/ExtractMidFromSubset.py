import os
from prepare.Utils import convert_mids_url_to_simplified_format
from prepare import Log
import logging


subset_file = '/Users/jiecxy/PycharmProjects/ture/test.data'
mid_file_dir = '/Users/jiecxy/PycharmProjects/ture/'
mid_file_name = 'fb_5M_mid.txt'
log_interval = 100000

if not os.path.isfile(subset_file):
    raise Exception("ERROR to read data file '{}'".format(subset_file))
if not os.path.isdir(mid_file_dir):
    raise Exception("ERROR not a dir '{}'".format(mid_file_dir))


line_count = 0
mid_set = set()
for line in open(subset_file):
    line_count += 1
    splits = line.split('\t')
    try:
        if len(splits) != 3:
            raise Exception
        mids = convert_mids_url_to_simplified_format(splits[0])
        mids.extend(convert_mids_url_to_simplified_format(splits[2]))
        for mid in mids:
            mid_set.add(mid)
    except Exception:
        logging.error("ERROR in line {}: '{}'".format(line_count, line))
    if line_count % log_interval == 0:
        logging.debug("Processed {} lines...".format(line_count))


mid_file = os.path.join(mid_file_dir, mid_file_name)
with open(mid_file, 'w') as f:
    for mid in sorted(mid_set):
        f.write(mid + '\n')