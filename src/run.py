from metapredictor import Metapredictor

if __name__ == '__main__':
   
    stored_dataset_directory = '../data/intermediate_state/'
    stored_dataset_filename = 'complete_dataset.pkl'

    m = Metapredictor('../data/raw_coughs/','../data/labels/metadata.csv',percentage_train=.8)
    #m.buildTrainTestSet()
    #m.saveDataset(stored_dataset_directory + stored_dataset_filename)
    #m.restoreDataset(stored_dataset_directory + stored_dataset_filename)

    #m.trainClassifier()
    #m.setInferences(model_path='internal',dataset='all')
    #m.setInferences(model_path = '../models/checkpoint-954',dataset='all')

    stored_inference_directory = '../data/intermediate_state/'
    stored_inference_filename = 'complete_dataset_with_inference.pkl'
    #m.saveDataset(stored_inference_directory + stored_inference_filename)
    #m.restoreDataset(stored_inference_directory + stored_inference_filename)

    formatted_logits_directory = '../data/intermediate_state/'
    formatted_logits_filename = 'complete_dataset_with_formatted_logits.pkl'
    #m.formatLogits()
    #m.saveDataset(formatted_logits_directory + formatted_logits_filename)
    m.restoreDataset(formatted_logits_directory + formatted_logits_filename)
    
    m.generatePrecisionRecallMetrics()

