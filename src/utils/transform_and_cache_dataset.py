import sys
sys.path.append('../')

from metapredictor import Metapredictor

if __name__ == '__main__':

    pickle_directory = '../../data/intermediate_state/'
    pickle_filename = 'complete_dataset.pkl'

    m = Metapredictor('../../data/raw_coughs/','../../data/labels/metadata.csv',percentage_train=.2)
    m.buildTrainTestSet()
    m.saveDataset(pickle_directory + pickle_filename)
