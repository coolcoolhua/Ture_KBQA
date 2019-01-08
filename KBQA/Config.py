import EntityDetection.Config
import RelationPrediction.Config

# ======================= Generate relation prediction data =======================
ED_file_base_dir = EntityDetection.Config.out_file_base_dir
ED_data_names = EntityDetection.Config.output_data_names[-1:]


RP_file_base_dir = RelationPrediction.Config.out_file_base_dir
RP_data_names = RelationPrediction.Config.output_data_names[-1:]

output_base_dir = '/Users/jiecxy/PycharmProjects/ture/KBQA/data'
output_data_name = 'data_test.txt'
