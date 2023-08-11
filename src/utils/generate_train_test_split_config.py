import json
import pandas as pd
import random

def generateTrainTestSplitConfig(metadata_filepath='../../data/labels/metadata.csv',
                         metapredictor_config_filepath='../../configs/metapredictor_params.json',
                         label_config_dir='../../configs/',
                         label_config_filename='label_config.csv',
                         random_seed=123):

    print('\nGenerating Train Test Split Config..')
    random.seed(random_seed)

    meta_config = open(metapredictor_config_filepath)
    metapredictor_config = json.load(meta_config)
    meta_config.close()
    percentage_train = metapredictor_config['percentage_train']

    config_df = pd.DataFrame(columns=['caf_id','train_or_test'])
    data_labels = pd.read_csv(metadata_filepath)

    id_iter = 0
    while(id_iter < len(data_labels)):

        random_val = random.uniform(0,1)
        current_row = data_labels.iloc[id_iter,:]
        current_id = current_row['FILENAME'].split('.')[0]
        if(random_val <= percentage_train):
            config_df.loc[id_iter,'caf_id'] = current_id
            config_df.loc[id_iter,'train_or_test'] = 'train'
        if(random_val > percentage_train):
            config_df.loc[id_iter,'caf_id'] = current_id
            config_df.loc[id_iter,'train_or_test'] = 'test'

        id_iter += 1

    config_df.to_csv(label_config_dir + label_config_filename)

    print('Label Config Saved To: ' + label_config_dir + label_config_filename + '\n')

if __name__ == '__main__':

    generateTrainTestSplitConfig()
