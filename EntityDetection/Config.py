

# ======================= Generate entity detection data =======================
question_data_base_dir = '/Users/jiecxy/PycharmProjects/ture/data'
question_data_names = ['annotated_fb_data_train.txt', 'annotated_fb_data_valid.txt', 'annotated_fb_data_test.txt']
out_file_base_dir = '/Users/jiecxy/PycharmProjects/ture/EntityDetection/data'
output_data_names = ['data_train.txt', 'data_valid.txt', 'data_test.txt']
unknown_mids_file_name = 'unknown_mid.txt'


map_file = '/Users/jiecxy/PycharmProjects/ture/data/names.trimmed.5M.txt'


# ================================= Train model =================================
model_save_dir = '/Users/jiecxy/PycharmProjects/ture/EntityDetection/model'
data_base_dir = out_file_base_dir
data_names = output_data_names