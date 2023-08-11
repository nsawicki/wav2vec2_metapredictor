import cv2
import librosa
import json
import os
import warnings
import pandas as pd
import numpy as np
import random
import pickle as pk
import torch
import matplotlib.pyplot as plt

import torchaudio
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils import data
from torchvision import datasets, models, transforms
import torch.optim as optim

train_on_gpu=torch.cuda.is_available()

from PIL import Image
from utils.getcovidlabel import getCovidLabel
from utils.computemetrics import compute_metrics,compute_metrics2
from utils.audio2melimage2 import audio2MelImage
from sklearn.metrics import precision_recall_curve,roc_curve

class PytorchDataset(data.Dataset):

    def __init__(self,
                 dataset_list):
                 #data_dir,
                 #recording_len,
                 #fft_len,
                 #DecNum=5,
                 #Im_3D=False):
        
        self.dataset_list = dataset_list
        #self.labels = labels
        #self.caf_ids_list = caf_ids_list
        #self.data_dir = data_dir
        #self.recording_len = recording_len # length of most records
        #self.fft_len = fft_len 
        #self.Im_3D = Im_3D
        #self.NFCC_Num = 128
        #self.TimeSamp = 128

    def __len__(self):

        return len(self.dataset_list)

    def __getitem__(self,index):

        caf_id = self.dataset_list[index]["filename"]
        mel_image = self.dataset_list[index]["mel_image"]
        caf_label = self.dataset_list[index]["label"]

        return mel_image,caf_label,caf_id

class CnnAudioNet(nn.Module):
    def __init__(self,NumClasses):
        super(CnnAudioNet,self).__init__()
        self.NumClasses = NumClasses
        self.Fc_features = 128
        self.C1 = nn.Conv2d(1,32,5,padding=1)
        self.C2 = nn.Conv2d(32,32,5,padding=1)
        self.C3 = nn.Conv2d(32,64,5,padding=1)
        self.C4 = nn.Conv2d(64,64,5,padding=1)
                                                                            
        self.BN1 = nn.BatchNorm2d(32)
        self.BN2 = nn.BatchNorm2d(64)
        self.BN3 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(2,2)
        self.maxpool2 = nn.MaxPool2d((1,2),(1,2))
                                                                                                                 
                                                                                                               
        self.fc1 = nn.Linear(64*8*8,128)
        self.fc2 = nn.Linear(128,self.NumClasses)
        self.dropout = nn.Dropout(0.25)
        self.Bat1 = nn.BatchNorm1d(128)

    def forward(self,x):
        # add sequence of convolutional and max pooling layers
        x = F.relu(self.BN1(self.C1(x)))
        x = self.maxpool1(F.relu(self.BN1(self.C2(x))))
        x = F.relu(self.BN2(self.C3(x)))
        x = self.maxpool1(F.relu(self.BN2(self.C4(x))))
        x = F.relu(self.BN2(self.C4(x)))
        x = self.maxpool1(F.relu(self.BN2(self.C4(x))))
        x = F.relu(self.BN2(self.C4(x)))
        x = F.relu(self.BN3(self.C4(x)))
        # flatten image input
        x = self.dropout(x.view(-1,64*8*8))
        # add dropout layer
        x =  self.dropout(self.fc1(x))
        # add 1st hidden layer, with relu activation function
        # add dropout layer
        # add 2nd hidden layer, with relu activation function
        #x = torch.sigmoid(self.fc2(x))
        x = self.fc2(x)
        return x

class CNNClassifier:

    def __init__(self,
                cough_dir='../data/raw_coughs/',
                metadata_filepath='../data/labels/metadata.csv',
                config_dir='../configs/',
                metapredictor_config_filepath='../configs/metapredictor_params.json',
                intermediate_state_dir='../data/intermediate_state/',
                cnn_parameters_filepath = '../configs/cnn_classifier_params.json',
                train_test_labels_filepath='../configs/label_config.csv',
                num_classes=2,
                num_epochs=40):

        print('\nInitializing Mel Spectrum CNN Classifier..\n')

        self.num_classes = num_classes

        self.cnn_parameters_filepath = cnn_parameters_filepath
        self.cough_dir = cough_dir
        self.metadata_filepath = metadata_filepath
        self.config_dir = config_dir
        self.metapredictor_config_filepath = metapredictor_config_filepath
        self.intermediate_state_dir = intermediate_state_dir
        self.train_test_labels_filepath = train_test_labels_filepath
        
        print('CNN parameters gather from: ' + str(cnn_parameters_filepath))
        print('Cough Directory: ' + str(cough_dir))
        print('Labels gathered from: ' + str(metadata_filepath))
        print('Configuration Directory: ' + str(config_dir))
        print('Intermediate Data will be saved to: ' + str(intermediate_state_dir) + '\n')

        # Dataset initialized to None, percentage_train is determined by the metapredictor param json
        self.dataset = None
        self.pytorch_train_dataset = None
        self.pytorch_test_dataset = None
        self.train_test_ids_dict = self.assignTrainTestIds()

        self.num_frequency_bins,self.mel_sampling_rate,self.num_mfcc,self.num_time_samps,self.mel_window_length,self.mel_hop_length\
                = self.loadParameters(cnn_parameters_filepath)

        self.model = None
        self.train_params = None
        self.test_params = None
        self.loss_criterion = None
        self.optimizer = None
        self.num_epochs = num_epochs

    def loadParameters(self,parameter_filepath):

        params = open(parameter_filepath)
        param_json = json.load(params)
        params.close()
        return param_json['num_frequency_bins'],\
                param_json['mel_sampling_rate'],\
                param_json['num_mfcc'],\
                param_json['num_time_samps'],\
                param_json['mel_window_length'],\
                param_json['mel_hop_length']


    def assignTrainTestIds(self):

        if not os.path.exists(self.train_test_labels_filepath):
            generateTrainTestSplitConfig(metadata_filepath=self.metadata_filepath,
                                         metapredictor_config_filepath=self.metapredictor_config_filepath,
                                         label_config_dir=self.config_dir)

        train_test_ids_dict = {}
        data_labels = pd.read_csv(self.train_test_labels_filepath)

        id_iter = 0
        while(id_iter < len(data_labels)):

            current_row = data_labels.iloc[id_iter,:]
            current_id = current_row['caf_id']
            train_or_test = current_row['train_or_test']
            train_test_ids_dict[current_id] = train_or_test

            id_iter += 1

        return train_test_ids_dict

    def buildDataset(self,restore_cached=False,saved_dataset_filename='mel_image_dataset.pkl',convert_to_rgb_images=False):

        if restore_cached == True:
            
            self.restoreDataset(self.intermediate_state_dir + saved_dataset_filename)
            print('Training Example: ' + str(self.dataset["train"][0]))
            print('Test Example: ' + str(self.dataset["test"][0]))

        elif restore_cached == False:

            full_config = pd.read_csv(self.metadata_filepath)

            self.dataset = {}
            self.dataset["train"] = []
            self.dataset["test"] = []

            print('Building Mel Spectrum Image Dataset..\n')

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
                    self.dataset["train"][train_iter]["filename"] = caf_name
                    self.dataset["train"][train_iter]["caf_id"] = caf_id
                    self.dataset["train"][train_iter]["caf_number"] = caf_number
                    tmp_audio, tmp_fs = librosa.load(self.cough_dir + caf_name,sr = self.mel_sampling_rate)
                    self.dataset["train"][train_iter]["audio"] = np.float32(tmp_audio)
                    self.dataset["train"][train_iter]["mel_image"] =\
                                audio2MelImage(tmp_audio,tmp_fs,self.mel_window_length,self.mel_hop_length,self.num_frequency_bins,self.num_time_samps,convert_to_rgb_image=convert_to_rgb_images)
                    self.dataset["train"][train_iter]["sampling_rate"] = tmp_fs
                    covid_label = getCovidLabel(caf_id,full_config)
                    self.dataset["train"][train_iter]["label"] = torch.nn.functional.one_hot(torch.tensor(covid_label),num_classes=self.num_classes)
                    train_iter += 1

                if(self.train_test_ids_dict[caf_id] == 'test'):
                    self.dataset["test"].append({})
                    self.dataset["test"][test_iter]["filename"] = caf_name
                    self.dataset["test"][test_iter]["caf_id"] = caf_id
                    self.dataset["test"][test_iter]["caf_number"] = caf_number
                    tmp_audio, tmp_fs = librosa.load(self.cough_dir + caf_name,sr = self.mel_sampling_rate)
                    self.dataset["test"][test_iter]["audio"] = np.float32(tmp_audio)
                    self.dataset["test"][test_iter]["mel_image"] =\
                            audio2MelImage(tmp_audio,tmp_fs,self.mel_window_length,self.mel_hop_length,self.num_frequency_bins,self.num_time_samps,convert_to_rgb_image=convert_to_rgb_images)
                    self.dataset["test"][test_iter]["sampling_rate"] = tmp_fs
                    covid_label = getCovidLabel(caf_id,full_config)
                    self.dataset["test"][test_iter]["label"] = torch.nn.functional.one_hot(torch.tensor(covid_label),num_classes=self.num_classes)
                    test_iter += 1

                caf_iter += 1

            print('Training Example: ' + str(self.dataset["train"][0]))
            print('Test Example: ' + str(self.dataset["test"][0]))

            self.saveDataset(self.intermediate_state_dir + saved_dataset_filename)

    def buildPytorchTrainTestDataset(self,
                              restore_cached=False,
                              saved_train_dataset_filename='pytorch_cnn_train.pkl',
                              saved_test_dataset_filename='pytorch_cnn_test.pkl'):

        print('\nGenerating Pytorch Train and Test Sets..\n')

        if restore_cached == True:
            
            with open(self.intermediate_state_dir + saved_train_dataset_filename,'rb') as f:
                self.pytorch_train_dataset = pk.load(f)
            print('Successfully restored dataset: ' + str(self.intermediate_state_dir + saved_train_dataset_filename))

            with open(self.intermediate_state_dir + saved_test_dataset_filename,'rb') as f:
                self.pytorch_test_dataset = pk.load(f)
            print('Successfully restored dataset: ' + str(self.intermediate_state_dir + saved_test_dataset_filename))

        elif restore_cached == False:

            self.pytorch_train_dataset = PytorchDataset(self.dataset["train"])
            
            with open(self.intermediate_state_dir + saved_train_dataset_filename,'wb+') as f:
                pk.dump(self.pytorch_train_dataset,f)
            print('Successfully saved dataset: ' + str(self.intermediate_state_dir + saved_train_dataset_filename))

            self.pytorch_test_dataset = PytorchDataset(self.dataset["test"])

            with open(self.intermediate_state_dir + saved_test_dataset_filename,'wb+') as f:
                pk.dump(self.pytorch_test_dataset,f)
            print('Successfully saved dataset: ' + str(self.intermediate_state_dir + saved_test_dataset_filename))

    def saveDataset(self,saved_filepath):

        with open(saved_filepath,'wb+') as f:
            pk.dump(self.dataset,f)

        print('Successfully saved dataset: ' + str(saved_filepath))
    
    def restoreDataset(self,saved_filepath):

        with open(saved_filepath,'rb') as f:
            self.dataset = pk.load(f)

        print('Successfully restored dataset: ' + str(saved_filepath))

    def preprocessPytorchDataset(self):

        # Check image sizing
        # Check mean and std
        print('Measuring Size of Mel Images and Shaping to be Identical..')

        for datapoint in self.dataset["train"]:
            print(datapoint["mel_image"].size())

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])

    def buildModel(self,show_model=False):

        self.train_params = {'batch_size': 32,
                    'shuffle': True,
                    'num_workers': 8}
        self.test_params = {'batch_size': 32,
                    'shuffle': True,
                    'num_workers': 8}

        self.model = CnnAudioNet(self.num_classes)
        if train_on_gpu:
            self.model.cuda()
            if show_model:
                print(self.model)
            
            # specify loss function (MSE)

            #criterion = nn.MSELoss()
            #criterion = nn.BCELoss()
            self.loss_criterion = nn.BCEWithLogitsLoss()
            #self.loss_criterion = nn.CrossEntropyLoss()
            #criterion = nn.MultiLabelSoftMarginLoss()

            self.optimizer = optim.Adam(params=self.model.parameters(),lr=0.001,weight_decay=1e-2)
            #optimizer = optim.Adam(model.parameters(), lr=0.005)

    def trainModel(self):

        import time
        start_time = time.time()
        #Warnings.filterwarnings('ignore')

        # number of epochs to train the model
        n_epochs = self.num_epochs
        MSE_train_by_epoch = []
        MSE_val_by_epoch = []
        print('\nBegin Training - ' + str(n_epochs) + ' epochs..\n')

        dataloader_train = data.DataLoader(self.pytorch_train_dataset,**self.train_params)
        dataloader_val = data.DataLoader(self.pytorch_test_dataset,**self.test_params)

        valid_loss_min = np.Inf # track change in validation loss
        idx = 0 
        for epoch in range(1, n_epochs+1):

            # keep track of training and validation loss
            train_loss = 0.0
            total_MSE_train = 0 
            TotEl = 0

            ###################
            # train the model #
            ###################
            self.model.train()

            for dataBatch, target,_ in dataloader_train:
                                                                
                idx+=1

                # move tensors to GPU if CUDA is available
                if train_on_gpu:
                    #dataBatch, target = dataBatch.unsqueeze(1).float().cuda(), target.cuda()
                    dataBatch, target = dataBatch.permute(0,3,1,2).float().cuda(), target.cuda() 
                
                # clear the gradients of all optimized variables
                self.optimizer.zero_grad()
                # forward pass: compute predicted outputs by passing inputs to the model
                output = self.model(dataBatch)
                # calculate the batch loss
                #loss = criterion(output, torch.squeeze(torch.argmax(target,dim=-1)))
                loss = self.loss_criterion(output,target.float())
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                self.optimizer.step()
                # update training loss
                train_loss += loss.item()*dataBatch.size(0)
                #print(loss.item())
                #print('Finish batch')
                _,pred = torch.max(output,1)
                                                                                                                                                                                                                    
                #Correct = torch.sum(torch.pow(output-target.float(),2))#
                ErrorS = torch.sum(torch.pow(torch.sigmoid(output)-target.float(),2))
                total_MSE_train += ErrorS
                TotEl += output.numel()
                Correct =torch.sum(pred==torch.squeeze(torch.argmax(target,dim=-1)))
                #print('Train batch loss: {:.6f},  Error: {:.4f},  Sum Correct: {} out of {}'.format(loss,ErrorS,Correct,output.shape[0]))
            
            with torch.no_grad():
                self.model.eval()
                TotEl_v = 0
                valid_loss = 0 
                total_MSE_val = 0
                for dataBatch_v, target ,_ in dataloader_val:

                    # move tensors to GPU if CUDA is available
                    if train_on_gpu:
                        #dataBatch_v, target = dataBatch_v.unsqueeze(1).float().cuda(),target.cuda()
                        dataBatch_v, target = dataBatch_v.permute(0,3,1,2).float().cuda(),target.cuda()
                    # forward pass: compute predicted outputs by passing inputs to the model
                    output = self.model(dataBatch_v)
                    # calculate the batch loss
                    loss = self.loss_criterion(output,target.float())
                    #loss = criterion(output, torch.squeeze(torch.argmax(target,dim=-1)))
                    # update average validation loss 
                    output.shape
                    _,pred = torch.max(output,1)
                    Correct = torch.sum(pred==torch.squeeze(torch.argmax(target,dim=-1)))
                    #SumCorrectVal += Correct
                    valid_loss += loss.item()*dataBatch.size(0)
                    #print(TotVal)
                    ErrorS = torch.sum(torch.pow(torch.sigmoid(output)-target.float(),2))#
                    total_MSE_val += ErrorS
                    TotEl_v += output.numel()

            print('Epoch: {} \t  Train Total Square Error: {:.6f} \t Test Total Square Error: {:.6f} '.format(epoch,total_MSE_train,total_MSE_val))
            MSE_train_by_epoch.append(total_MSE_train)
            MSE_val_by_epoch.append(total_MSE_val)

    def buildVGGModel(self):

        self.train_params = {'batch_size': 32,
                             'shuffle': True,
                             'num_workers': 8}
        self.test_params = {'batch_size': 32,
                            'shuffle': True,
                            'num_workers': 8}

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.vgg19(pretrained=True)

        for feat in self.model.features:
            feat.requires_grad = False

        self.model.classifier[6].out_features = self.num_classes
        self.model.classifier[6].weight = torch.nn.Parameter(torch.randn(self.num_classes,4096))
        self.model.classifier[6].bias = torch.nn.Parameter(torch.ones(self.num_classes))

        self.model.classifier.to(device)
        self.model.features.to(device)
        print(self.model)

        self.optimizer = optim.Adam(params=self.model.parameters(),lr=0.001,weight_decay=1e-5)
        self.loss_criterion = nn.BCEWithLogitsLoss()
        #self.loss_criterion = nn.MSELoss()


    def trainClassifier(self):

        print('Started CNN Classifier Training..\n')

    def setInferences(self):

        print('Saving Inferences from Trained Model..\n')

    def generatePrecisionRecallMetrics(self):

        print('Generating Precision & Recall Curves..\n')

    def getTestInferences(self):

        print('Formatting Test Inferences..\n')

    def run(self):

        self.buildDataset(restore_cached=True,convert_to_rgb_images=True)

        self.buildPytorchTrainTestDataset(restore_cached=True)

        #Try Normalizing and Stdev
        #self.preprocessPytorchDataset()

        #self.buildModel()
        self.buildVGGModel()
        self.trainModel()

        #self.trainClassifier()

        #self.setInferences()

        #self.generatePrecisionRecallMetrics()

if __name__ == '__main__':

    c = CNNClassifier(num_epochs=100)
    c.run()
