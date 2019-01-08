

# ======================= Generate relation prediction data =======================
question_data_base_dir = '/Users/jiecxy/PycharmProjects/ture/data'

question_data_names = ['annotated_fb_data_train.txt', 'annotated_fb_data_valid.txt', 'annotated_fb_data_test.txt']
out_file_base_dir = '/Users/jiecxy/PycharmProjects/ture/RelationPrediction/data'
output_data_names = ['data_train.txt', 'data_valid.txt', 'data_test.txt']
relation_list_file_name = 'relation_list.txt'


# ================================= Train model =================================
model_save_dir = '/Users/jiecxy/PycharmProjects/ture/RelationPrediction/model'
data_base_dir = out_file_base_dir
data_names = output_data_names