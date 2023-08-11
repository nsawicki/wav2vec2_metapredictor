import librosa
import os
import warnings
import pandas as pd
import random
import pickle as pk
import torch
import matplotlib.pyplot as plt

from transformers import Wav2Vec2Config, Wav2Vec2Model, Wav2Vec2ForCTC, Wav2Vec2Tokenizer, AutoFeatureExtractor,\
        AutoModelForAudioClassification, TrainingArguments, Trainer, pipeline
from utils.wav2vec2transform import wav2vec2TransformDirectory
from utils.traintestsplit import trainTestSplit
from utils.getcovidlabel import getCovidLabel
from utils.computemetrics import compute_metrics,compute_metrics2
from sklearn.metrics import precision_recall_curve, roc_curve

class Metapredictor:

    def __init__(self,
                 cough_dir,
                 metadata_filepath,
                 config_dir='../configs/',
                 percentage_train=.8,
                 random_seed=123):

        random.seed(random_seed)

        print('\nBeginning Metapredictor Training..\n')
        
        self.cough_dir = cough_dir
        self.metadata_filepath = metadata_filepath
        self.config_dir = config_dir

        print('Cough Directory: ' + str(cough_dir))
        print('Labels gathered from: ' + str(metadata_filepath))
        print('Configuration Directory: ' + str(config_dir) + '\n')

        # Dataset initialized to None, percentage_train is the percentage of total coughs randomly sorted into the training set
        self.dataset = None
        self.percentage_train = percentage_train
        self.train_test_ids_dict = self.assignTrainTestIds()
        
        # mapping of data label encodings - required for Huggingface AutoModelForAudioClassification()
        self.id2label, self.label2id = self.generateIDMapping(metadata_filepath)

        self.feature_extractor = self.getFeatureExtractor()
        self.classification_model = self.getModel()
        self.wav2vec2_sampling_rate = 16000

    def getFeatureExtractor(self):

        print('Loading feature extractor from wav2vec2..')
        
        feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
        return feature_extractor

    def assignTrainTestIds(self):

        train_test_ids_dict = {}
        data_labels = pd.read_csv(self.metadata_filepath)

        id_iter = 0
        while(id_iter < len(data_labels)):
            
            random_val = random.uniform(0,1)
            current_row = data_labels.iloc[id_iter,:]
            current_id = current_row['FILENAME'].split('.')[0]
            if(random_val <= self.percentage_train):
                train_test_ids_dict[current_id] = 'train'
            if(random_val > self.percentage_train):
                train_test_ids_dict[current_id] = 'test'

            id_iter += 1

        return train_test_ids_dict

    def getModel(self):

        num_labels = len(self.id2label)

        print('Loading Facebook wav2vec2..')
        classification_model = AutoModelForAudioClassification.from_pretrained(
            "facebook/wav2vec2-base",
            num_labels=num_labels,
            label2id=self.label2id,
            id2label=self.id2label
        )
        
        return classification_model

    def generateIDMapping(self,metadata_filepath):

        data_labels = pd.read_csv(metadata_filepath)
        
        label2id, id2label = dict(), dict()
        id2label['0'] = 'negative'
        id2label['1'] = 'positive'
        label2id['negative'] = '0'
        label2id['positive'] = '1'

        return id2label, label2id

    def trainClassifier(self):

        training_args = TrainingArguments(
            output_dir="../models/",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=3e-5,
            per_device_train_batch_size=32,
            gradient_accumulation_steps=4,
            per_device_eval_batch_size=32,
            num_train_epochs=10,
            warmup_ratio=0.1,
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            push_to_hub=False
        )

        trainer = Trainer(
            model=self.classification_model,
            args=training_args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["test"],
            tokenizer=self.feature_extractor,
            compute_metrics=compute_metrics,
        )
        
        print('\n' + 'Beginning Training..' + '\n')
        trainer.train()

    def buildTrainTestSet(self):

        full_config = pd.read_csv(self.metadata_filepath)

        self.dataset = {}
        self.dataset["train"] = []
        self.dataset["test"] = []

        print('Tokenizing and Generating the Train / Test Split..')

        caf_files = os.listdir(self.cough_dir)
        caf_files.remove('.gitignore')

        caf_iter = 0
        train_iter = 0
        test_iter = 0
        while(caf_iter < len(caf_files)):

            caf_name = caf_files[caf_iter]
            caf_id = caf_name.split("-cough-")[0]
            caf_number = caf_name.split("-cough-")[1]

            if(self.train_test_ids_dict[caf_id] == 'train'):
                self.dataset["train"].append({})
                self.dataset["train"][train_iter]["caf_id"] = caf_id
                self.dataset["train"][train_iter]["caf_number"] = caf_number
                self.dataset["train"][train_iter]["audio"], tmp_fs = librosa.load(self.cough_dir + caf_name,sr = 16000)
                self.dataset["train"][train_iter]["input_values"] = self.feature_extractor(
                    [self.dataset["train"][train_iter]["audio"]], sampling_rate=16000, max_length=16000, truncation=True
                )["input_values"][0]
                self.dataset["train"][train_iter]["tensor"] = self.feature_extractor(
                        self.dataset["train"][train_iter]["audio"],sampling_rate=16000,max_length=16000,truncation=True,return_tensors='pt'
                )
                self.dataset["train"][train_iter]["sampling_rate"] = 16000
                self.dataset["train"][train_iter]["label"] = getCovidLabel(caf_id,full_config)
                train_iter += 1

            if(self.train_test_ids_dict[caf_id] == 'test'):
                self.dataset["test"].append({})
                self.dataset["test"][test_iter]["caf_id"] = caf_id
                self.dataset["test"][test_iter]["caf_number"] = caf_number
                self.dataset["test"][test_iter]["audio"], tmp_fs = librosa.load(self.cough_dir + caf_name,sr = 16000)
                self.dataset["test"][test_iter]["input_values"] = self.feature_extractor(
                    [self.dataset["test"][test_iter]["audio"]], sampling_rate=16000, max_length=16000, truncation=True
                )["input_values"][0]
                self.dataset["test"][test_iter]["tensor"] = self.feature_extractor(
                        self.dataset["test"][test_iter]["audio"],sampling_rate=16000,max_length=16000,truncation=True,return_tensors='pt'
                )
                self.dataset["test"][test_iter]["sampling_rate"] = 16000
                self.dataset["test"][test_iter]["label"] = getCovidLabel(caf_id,full_config)
                test_iter += 1

            caf_iter += 1

        print('Sample Training Data: ' + str(self.dataset["train"][0]))
        print('Training set size: ' + str(len(self.dataset["train"])))
        print('Testing set size: ' + str(len(self.dataset["test"]))) 

    def saveDataset(self,saved_filepath):

        with open(saved_filepath,'wb+') as f:
            pk.dump(self.dataset,f)

        print('Successfully saved dataset: ' + str(saved_filepath))

    def restoreDataset(self,saved_filepath):

        with open(saved_filepath,'rb') as f:
            self.dataset = pk.load(f)

        print('Successfully restored dataset: ' + str(saved_filepath))

    def plotPrecisionRecall(self):
         accuracy.compute(predictions=predictions, references=eval_pred.label_ids)

    def printTestInferences(self):

        dataset_iter = 0
        while(dataset_iter < len(self.dataset["test"])):
            print("Caf id: " + str(self.dataset["test"][dataset_iter]["caf_id"]))
            print(self.dataset["test"][dataset_iter]["logits"])
            dataset_iter += 1

    def setTrainInferences(self):

        dataset_iter = 0
        while(dataset_iter < len(self.dataset["train"])):

            self.dataset["test"][dataset_iter]["logits"] = self.getInference(self.dataset["train"][dataset_iter]["input_values"])
            dataset_iter += 1

    def setTestInferences(self):

        dataset_iter = 0
        while(dataset_iter < len(self.dataset["test"])):
            print('Generating Test Inference..' + str(dataset_iter))
            self.dataset["test"][dataset_iter]["logits"],self.dataset["test"][dataset_iter]["predicted_label"] = \
                    self.getInference(self.dataset["test"][dataset_iter]["tensor"])
            #print(self.dataset["test"][dataset_iter]["logits"])
            dataset_iter += 1

    def getInference(self,datapoint):

        model = AutoModelForAudioClassification.from_pretrained('../models/checkpoint-954')
        #print(datapoint_dict["input_values"])

        #transform = self.feature_extractor(datapoint_dict["audio"], sampling_rate=self.wav2vec2_sampling_rate, return_tensors="pt")
        #print(transform)
        with torch.no_grad():
                logits = model(**datapoint).logits
                predicted_class_ids = torch.argmax(logits).item()
                predicted_label = self.id2label[str(predicted_class_ids)]

        return logits,predicted_label 

    def printInferences(self,model_path,dataset='test'):

        if model_path == 'internal':
            print('Generating inference from recently trained model..')
            classifier = pipeline("audio-classification",model=self.classification_model)
        else:
            print('Generating inference from previously trained model: ' + str(model_path))
            classifier = pipeline("audio-classification",model=model_path)

        if dataset == 'train' or dataset == 'all':
            dataset_iter = 0
            while(dataset_iter < len(self.dataset["train"])):
                print('---------- Cough: ' + str(self.dataset["train"][dataset_iter]["caf_id"]) + '-' + \
                        str(self.dataset["train"][dataset_iter]["caf_number"] + '----------'))
                print('Prediction: ' + str(classifier(self.dataset["train"][dataset_iter]["audio"])))
                print('Ground Truth: ' + str(self.id2label[str(self.dataset["train"][dataset_iter]["label"])]))
                dataset_iter +=1

        if dataset == 'test' or dataset == 'all':
            dataset_iter = 0
            while(dataset_iter < len(self.dataset["test"])):
                print('---------- Cough: ' + str(self.dataset["test"][dataset_iter]["caf_id"]) + '-' + \
                    str(self.dataset["test"][dataset_iter]["caf_number"] + '----------'))
                print('Prediction: ' + str(classifier(self.dataset["test"][dataset_iter]["audio"])))
                print('Ground Truth: ' + str(self.id2label[str(self.dataset["test"][dataset_iter]["label"])]))
                dataset_iter +=1

    def setInferences(self,model_path,dataset='all'):

        if model_path == 'internal':
            print('Generating inference from recently trained model..')
            classifier = pipeline("audio-classification",model=self.classification_model)
        else:
            print('Generating inference from previously trained model: ' + str(model_path))
            classifier = pipeline("audio-classification",model=model_path)

        print('Saving Training Inferences.. ')
        if dataset == 'train' or dataset == 'all':
            dataset_iter = 0
            while(dataset_iter < len(self.dataset["train"])):
                self.dataset["train"][dataset_iter]["logits"] = classifier(self.dataset["train"][dataset_iter]["audio"])
                dataset_iter +=1
            print('Example data entry: ' + str(self.dataset["train"][0]))

        print('Saving Testing Inferences.. ')
        if dataset == 'test' or dataset == 'all':
            dataset_iter = 0
            while(dataset_iter < len(self.dataset["test"])):
                self.dataset["test"][dataset_iter]["logits"] = classifier(self.dataset["test"][dataset_iter]["audio"])
                #self.dataset["test"][dataset_iter]["predicted_label"] = 
                dataset_iter +=1
            print('Example data entry: ' + str(self.dataset["test"][0]))

    def formatLogits(self):

        train_iter = 0
        while(train_iter < len(self.dataset["train"])):
            
            current_logits_1 = self.dataset["train"][train_iter]["logits"][0]
            current_logits_2 = self.dataset["train"][train_iter]["logits"][1]

            current_logits_1_prediction, current_logits_1_score = current_logits_1['label'] , current_logits_1['score']
            current_logits_2_prediction, current_logits_2_score = current_logits_2['label'] , current_logits_2['score']

            if current_logits_1_score >= current_logits_2_score:
                self.dataset["train"][train_iter]["predicted_label"] = current_logits_1_prediction
                self.dataset["train"][train_iter]["predicted_score"] = current_logits_1_score
            elif current_logits_1_score < current_logits_2_score:
                self.dataset["train"][train_iter]["predicted_label"] = current_logits_2_prediction
                self.dataset["train"][train_iter]["predicted_score"] = current_logits_2_score

            train_iter += 1

        print('Example data entry: ' + str(self.dataset["train"][0]))

        test_iter = 0
        while(test_iter < len(self.dataset["test"])):
            
            current_logits_1 = self.dataset["test"][test_iter]["logits"][0]
            current_logits_2 = self.dataset["test"][test_iter]["logits"][1]

            current_logits_1_prediction, current_logits_1_score = current_logits_1['label'] , current_logits_1['score']
            current_logits_2_prediction, current_logits_2_score = current_logits_2['label'] , current_logits_2['score']

            if current_logits_1_score >= current_logits_2_score:
                self.dataset["test"][test_iter]["predicted_label"] = current_logits_1_prediction
                self.dataset["test"][test_iter]["predicted_score"] = current_logits_1_score
            elif current_logits_1_score < current_logits_2_score:
                self.dataset["test"][test_iter]["predicted_label"] = current_logits_2_prediction
                self.dataset["test"][test_iter]["predicted_score"] = current_logits_2_score
            
            test_iter += 1

        print('Example data entry: ' + str(self.dataset["test"][0]))

    def generatePrecisionRecallMetrics(self,precision_recall_plot_filepath='../vizualizations/precision_recall.png',roc_plot_filepath='../vizualizations/roc.png'):

        print('Generating Precision Recall Metrics..')

        y_test = []
        y_score = []

        test_iter = 0
        while(test_iter < len(self.dataset["test"])):

            y_test.append(self.dataset["test"][test_iter]["label"])
            if self.dataset["test"][test_iter]["predicted_label"] == "positive":
                y_score.append(self.dataset["test"][test_iter]["predicted_score"])
            elif self.dataset["test"][test_iter]["predicted_label"] == "negative":
                y_score.append(1 - self.dataset["test"][test_iter]["predicted_score"])
            else:
                print('WARNING no valid label found in test set - expected positive or negative as predicted_label')

            test_iter += 1

        #calculate precision and recall
        precision, recall, thresholds = precision_recall_curve(y_test, y_score)

        print(precision)
        print(recall)
        print(thresholds)

        plt.figure()
        #create precision recall curve
        fig, ax = plt.subplots()
        ax.plot(recall, precision, color='blue')

        #add axis labels to plot
        ax.set_title('Precision-Recall Curve')
        ax.set_ylabel('Precision')
        ax.set_xlabel('Recall')

        plt.savefig(precision_recall_plot_filepath)
        print('Precision Recall Curve saved to: ' + precision_recall_plot_filepath)

        plt.clf()

        plt.figure()
        fpr, tpr, thresholds = roc_curve(y_test, y_score)
        plt.plot(fpr, tpr)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')

        plt.savefig(roc_plot_filepath)
        print('ROC plot saved to: ' + roc_plot_filepath)

    def generateMelImages(self):

       caf_files = os.listdir(self.cough_dir)
       caf_files.remove('.gitignore')
       
       y, sr = librosa.load(audio_name)
       window = np.hanning(window_size)
       out  = librosa.core.spectrum.stft(y, n_fft = window_size, hop_length = hop_length, 
                      window=window)
       out = 2 * np.abs(out) / np.sum(window)

       librosa.display.specshow(librosa.amplitude_to_db(out,ref=np.max),
                              y_axis='log', x_axis='time')

    def wav2vecTransform(self):

        #load model and tokenizer
        tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
        model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

        input_values = tokenizer(speech, return_tensors = 'pt').input_values
        #Store logits (non-normalized predictions)
        logits = model(input_values).logits
