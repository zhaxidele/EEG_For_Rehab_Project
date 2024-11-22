import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import KFold
import numpy as np
import os
import torchinfo
import pickle as pkl
from collections import Counter
from early_stopping import EarlyStopping
from sklearn.model_selection import train_test_split

import sys
sys.path.append('../')
# Import your custom modules or functions if needed
#from get_data import get_data
import get_data as get
from eeg_reduction import eeg_reduction
import pytorch_model as models # Assuming you have a module containing EEGNet model
import pytorch_eegnet as model_eegnet # Assuming you have a module containing EEGNet model
sys.path.append('subjects/')

import pytorch_online_model as model_sep

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Learning Rate Constant Scheduling
def step_decay(epoch):
    if epoch < 20:
        lr = 0.01
    elif epoch < 50:
        lr = 0.001
    else:
        lr = 0.0001
    return lr


# Save results
def save_results(history):
    # Save metrics
    results = np.zeros((4, len(history['accuracy'])))
    results[0] = history['accuracy']
    results[1] = history['val_accuracy']
    results[2] = history['loss']
    results[3] = history['val_loss']
    #results_str = os.path.join(results_dir, f'stats/global_class_{num_classes}_ds{n_ds}_nch{n_ch}_T{T}_split_{split_ctr}.csv')
    results_str = os.path.join(results_dir, f'stats/Off_line_train_history.csv')

    np.savetxt(results_str, np.transpose(results))
    #return results[0:2, -1]




# Hyperparameters
num_classes_list = [4]
n_epochs = 100
n_ds = 1
n_ch_list = [64]
T_list = [3]
kernLength = int(np.ceil(128 / n_ds))
poolLength = int(np.ceil(8 / n_ds))
num_splits = 5
acc = np.zeros((num_splits, 2))



# CHANGE EXPERIMENT NAME FOR DIFFERENT TESTS!!
experiment_name = 'your-online_experiment_pytorch'

# Data path and result directory
#datapath = "/home/sizhen/PycharmProjects/eegnet-based-embedded-bci-master/dataset/eeg-motor-movementimagery-dataset-1.0.0/files/"
datapath = "/datasets/sibian/eeg/"
datapath_save_subject_data = "/datasets/sibian/eeg/subjects/"

results_dir=f'results/{experiment_name}/'
os.makedirs(f'{results_dir}/stats', exist_ok=True)
os.makedirs(f'{results_dir}/model', exist_ok=True)
os.makedirs(f'{results_dir}/plots', exist_ok=True)

# Load data (you may need to implement your data loading function)
# X, y = get_data(datapath, n_classes=num_classes)


def check_data_balance():
    subjects = range(1,110)
    excluded_subjects = [88, 92, 100, 104]
    subjects = [x for x in subjects if (x not in excluded_subjects)]

    for subject in subjects:
        npzfile = np.load(datapath_save_subject_data + f'{subject}.npz')
        X, y = npzfile['X_Train'], npzfile['y_Train']

        print("Subject: {}, class ratio: {}".format(subject, Counter(y)))




def load_data():
    # Define subjects whose data is taken, namely from 1 to 109 excluding excluded_subjects
    subjects_training = training_subjects
    subjects_training = [x for x in subjects_training if (x not in excluded_subjects)]
    subjects_validation = validation_subjects
    subjects_validation = [x for x in subjects_validation if (x not in excluded_subjects)]
    subjects_testing= testing_subjects
    subjects_testing = [x for x in subjects_testing if (x not in excluded_subjects)]

    # initialize empty arrays to concatanate with itself later
    X_train = np.empty((0, 64, 480))
    y_train = np.empty(0)
    for subject in subjects_training:
        npzfile = np.load(datapath_save_subject_data + f'{subject}.npz')
        X, y = npzfile['X_Train'], npzfile['y_Train']
        # concatenate arrays in order to get the whole data in one input array
        X_train = np.concatenate((X_train,X))
        y_train = np.concatenate((y_train,y))
    print(X_train.shape)
    print(y_train.shape)
    np.savez(datapath_save_subject_data + f'training_' + str(training_subjects[0]) + f'_' + str(training_subjects[-1]), X_Train=X_train, y_Train=y_train)

    X_validation = np.empty((0, 64, 480))
    y_validation = np.empty(0)
    for subject in subjects_validation:
        npzfile = np.load(datapath_save_subject_data + f'{subject}.npz')
        X, y = npzfile['X_Train'], npzfile['y_Train']
        # concatenate arrays in order to get the whole data in one input array
        X_validation = np.concatenate((X_validation, X))
        y_validation = np.concatenate((y_validation, y))
    print(X_validation.shape)
    print(y_validation.shape)
    np.savez(datapath_save_subject_data + f'validation_' + str(validation_subjects[0]) + f'_' + str(validation_subjects[-1]), X_Validation=X_validation, y_Validation=y_validation)

    X_testing = np.empty((0, 64, 480))
    y_testing = np.empty(0)
    for subject in subjects_testing:
        npzfile = np.load(datapath_save_subject_data + f'{subject}.npz')
        X, y = npzfile['X_Train'], npzfile['y_Train']
        # concatenate arrays in order to get the whole data in one input array
        X_testing = np.concatenate((X_testing, X))
        y_testing = np.concatenate((y_testing, y))

        ############### split the test data into online training part and online/offline testing part ###########
        X_train_online, X_test_online, y_train_online, y_test_online = train_test_split(X, y, test_size= (1 - online_train_ratio), random_state=42, stratify=y)
        np.savez(datapath_save_subject_data + 'online_training_testing/' + f'Online_training_{subject}', X_Train=X_train_online, y_Train=y_train_online)
        np.savez(datapath_save_subject_data + 'online_training_testing/' + f'Online_testing_{subject}', X_Train=X_test_online, y_Train=y_test_online)


    print(X_testing.shape)
    print(y_testing.shape)
    np.savez(datapath_save_subject_data + f'testing_' + str(testing_subjects[0]) + f'_' + str(testing_subjects[-1]), X_Validation=X_testing, y_Validation=y_testing)



def load_data_leave_one_session_out():
    # Define subjects whose data is taken, namely from 1 to 109 excluding excluded_subjects
    subjects_training_validation = [x for x in subject_training_validation if (x not in excluded_subjects)]
    #subjects_validation = validation_subjects
    #subjects_validation = [x for x in subjects_validation if (x not in excluded_subjects)]
    subjects_testing= testing_subjects
    subjects_testing = [x for x in subjects_testing if (x not in excluded_subjects)]

    # initialize empty arrays to concatanate with itself later
    X_train = np.empty((0, 64, 480))
    y_train = np.empty(0)
    X_validate = np.empty((0, 64, 480))
    y_validate = np.empty(0)
    for subject in subjects_training_validation:
        npzfile = np.load(datapath_save_subject_data + f'{subject}.npz')
        X, y = npzfile['X_Train'], npzfile['y_Train']

        ############### split the training_validation data into offline training part (0.75) and offline validation part (0.25) ###########
        X_train_offline, X_validate_offline, y_train_offline, y_validate_offline = train_test_split(X, y, test_size= 0.25, random_state=42, stratify=y)

        # concatenate arrays in order to get the whole data in one input array
        X_train = np.concatenate((X_train, X_train_offline))
        y_train = np.concatenate((y_train, y_train_offline))
        X_validate = np.concatenate((X_validate, X_validate_offline))
        y_validate = np.concatenate((y_validate, y_validate_offline))
    print(X_train.shape)
    print(y_train.shape)
    print(X_validate.shape)
    print(y_validate.shape)
    np.savez(datapath_save_subject_data + f'leave_one_session_out_training_' + str(subject_training_validation[0]) + f'_' + str(subject_training_validation[-1]), X_Train=X_train, y_Train=y_train)
    np.savez(datapath_save_subject_data + f'leave_one_session_out_validation_' + str(subject_training_validation[0]) + f'_' + str(subject_training_validation[-1]), X_Train=X_validate, y_Train=y_validate)

    '''
    X_validation = np.empty((0, 64, 480))
    y_validation = np.empty(0)
    for subject in subjects_validation:
        npzfile = np.load(datapath_save_subject_data + f'{subject}.npz')
        X, y = npzfile['X_Train'], npzfile['y_Train']
        # concatenate arrays in order to get the whole data in one input array
        X_validation = np.concatenate((X_validation, X))
        y_validation = np.concatenate((y_validation, y))
    print(X_validation.shape)
    print(y_validation.shape)
    np.savez(datapath_save_subject_data + f'validation_' + str(validation_subjects[0]) + f'_' + str(validation_subjects[-1]), X_Validation=X_validation, y_Validation=y_validation)
    '''


    X_testing = np.empty((0, 64, 480))
    y_testing = np.empty(0)
    for subject in subjects_testing:
        npzfile = np.load(datapath_save_subject_data + f'{subject}.npz')
        X, y = npzfile['X_Train'], npzfile['y_Train']
        # concatenate arrays in order to get the whole data in one input array
        X_testing = np.concatenate((X_testing, X))
        y_testing = np.concatenate((y_testing, y))

        ############### split the test data into online training part and online/offline testing part ###########
        X_train_online, X_test_online, y_train_online, y_test_online = train_test_split(X, y, test_size= (1 - online_train_ratio), random_state=42, stratify=y)
        np.savez(datapath_save_subject_data + 'online_training_testing/' + f'Online_training_{subject}', X_Train=X_train_online, y_Train=y_train_online)
        np.savez(datapath_save_subject_data + 'online_training_testing/' + f'Online_testing_{subject}', X_Train=X_test_online, y_Train=y_test_online)


    print(X_testing.shape)
    print(y_testing.shape)
    np.savez(datapath_save_subject_data + f'testing_' + str(testing_subjects[0]) + f'_' + str(testing_subjects[-1]), X_Validation=X_testing, y_Validation=y_testing)






def save_online_training_testing_data():


    subjects = [x for x in testing_subjects if (x not in excluded_subjects)]

    for subject in subjects:

        ####################################
        # This rate (online_train_ratio) of test data will be used for online training, the left is used for testing
        ####################################
        npzfile = np.load(datapath_save_subject_data + 'online_training_testing/' + f'Online_testing_{subject}' + '.npz')
        X_test_online, y_test_online = npzfile['X_Train'], npzfile['y_Train']

        npzfile = np.load(datapath_save_subject_data + 'online_training_testing/' + f'Online_training_{subject}' + '.npz')
        X_train_online, y_train_online = npzfile['X_Train'], npzfile['y_Train']

        print(subject)
        print("train X shape: ", X_train_online.shape)
        print("test X shape: ", X_test_online.shape)
        print(f'# samples in incu: {X_train_online.shape[0]}, # samples in test: {X_test_online.shape[0]}')

        X_train_online = np.expand_dims(X_train_online, axis=-1)
        X_train_online = np.reshape(X_train_online, (X_train_online.shape[0], 1, 64, 480))

        X_test_online = np.expand_dims(X_test_online, axis=-1)
        X_test_online = np.reshape(X_test_online, (X_test_online.shape[0], 1, 64, 480))

        print("train X shape after expand and reshape: ", X_train_online.shape)
        print("test X shape after expand and reshape: ", X_test_online.shape)

        if not os.path.exists(incu_tar_root := os.path.join(datapath_save_subject_data, "online_training_testing", 'c_input', f'subject_{subject}', 'incu')):
            os.makedirs(incu_tar_root)

        if not os.path.exists(test_tar_root := os.path.join(datapath_save_subject_data,  "online_training_testing", 'c_input', f'subject_{subject}', 'test')):
            os.makedirs(test_tar_root)


        for i, (x, y) in enumerate(zip(X_train_online, y_train_online)):
            x = x.reshape(-1)
            y = y.reshape(-1)
            with open(os.path.join(incu_tar_root, f'{i}.input'), 'wb') as file:
                file.write(x.astype(np.float32).tobytes())
                file.write(y.astype(np.float32).tobytes())

        for i, (x, y) in enumerate(zip(X_test_online, y_test_online)):
            x = x.reshape(-1)
            y = y.reshape(-1)
            with open(os.path.join(test_tar_root, f'{i}.input'), 'wb') as file:
                file.write(x.astype(np.float32).tobytes())
                file.write(y.astype(np.float32).tobytes())




def training_offline(leave_one_session_out = False):

    # Load data
    #X, y = get.get_data(datapath, n_classes=num_classes)

    ######## If you want to save the data after loading once from .edf (faster)
    # np.savez(datapath+f'{num_classes}class',X_Train = X_Train, y_Train = y_Train)
    # np.savez(datapath + f'{num_classes}class', X_Train=X, y_Train=y)

    if leave_one_session_out:
        ### leave one session out train/test data set ###
        npzfile = np.load(datapath_save_subject_data + f'leave_one_session_out_training_' + str(subject_training_validation[0]) + f'_' + str(subject_training_validation[-1]) + '.npz')
        X_train, y_train = npzfile['X_Train'], npzfile['y_Train']
        npzfile = np.load(datapath_save_subject_data + f'leave_one_session_out_validation_' + str(subject_training_validation[0]) + f'_' + str(subject_training_validation[-1]) + '.npz')
        X_test, y_test = npzfile['X_Train'], npzfile['y_Train']
    else:
        ### leave one user out train/test data set ###
        npzfile = np.load(datapath_save_subject_data + f'training_' + str(training_subjects[0]) + f'_' + str(training_subjects[-1]) + '.npz')
        X_train, y_train = npzfile['X_Train'], npzfile['y_Train']
        npzfile = np.load(datapath_save_subject_data + f'validation_' + str(validation_subjects[0]) + f'_' + str(validation_subjects[-1]) + '.npz')
        X_test, y_test = npzfile['X_Validation'], npzfile['y_Validation']



    # reduce EEG data (downsample, number of channels, time window)
    #X = eeg_reduction(X, n_ds=n_ds, n_ch=n_ch, T=T)  # Implement this function

    # Expand dimensions to match expected EEGNet input
    print("X.shape before expanding: ", X_train.shape)
    X_train = np.expand_dims(X_train, axis=-1)
    print("X.shape after expanding: ", X_train.shape)
    X_train = np.reshape(X_train, (X_train.shape[0], 1, 64, 480))
    print("X.shape after reshaping: ", X_train.shape)


    n_samples = X_train.shape[2]
    y_cat_train = torch.tensor(y_train, dtype=torch.long)
    y_cat_train = nn.functional.one_hot(y_cat_train, 4)

    X_test = np.expand_dims(X_test, axis=-1)
    print("X.shape: ", X_test.shape)
    X_test = np.reshape(X_test, (X_test.shape[0], 1, 64, 480))
    n_samples = X_test.shape[2]
    y_cat_test = torch.tensor(y_test, dtype=torch.long)
    y_cat_test = nn.functional.one_hot(y_cat_test, 4)

    # using 5 folds
    #kf = KFold(n_splits=num_splits)

    #split_ctr = 0
    #for train, test in kf.split(X, y):
    # init model
    #model = models.EEGNet(nb_classes=num_classes, Chans=n_ch, Samples=n_samples, regRate=0.25, dropoutRate=0.2, kernLength=kernLength, poolLength=poolLength, numFilters=8, dropoutType='Dropout')

    #model = model_eegnet.EEGNet(chunk_size = 480, num_electrodes= 64, F1 = 8, F2 = 16, num_classes = 4, kernel_1 = 128, kernel_2 = 16, dropout = 0.2)
    #model = model_eegnet.EEGNet_2()

    backbone = model_sep.EEGNet_backbone()
    classifier = model_sep.Classifier()

    torchinfo.summary(backbone, (16, 1, 64, 480), col_names=("input_size", "output_size", "num_params", "kernel_size", "mult_adds"), verbose=1)
    torchinfo.summary(classifier, (16, 480), col_names=("input_size", "output_size", "num_params", "kernel_size", "mult_adds"), verbose=1)
    #torchinfo.summary(model, (16, 64, 480, 1), col_names=("input_size", "output_size", "num_params", "kernel_size", "mult_adds"), verbose=1)
    backbone.to(device)
    classifier.to(device)

    # Set Learning Rate
    #optimizer = optim.Adam(model.parameters(), lr=0.01)
    optimizer_backbone = optim.Adam(backbone.parameters(), lr=0.01)
    optimizer_classifier = optim.Adam(classifier.parameters(), lr=0.01)
    scheduler_backbone = StepLR(optimizer_backbone, step_size=40, gamma=0.1, verbose=0)
    scheduler_classifier = StepLR(optimizer_classifier, step_size=40, gamma=0.1, verbose=0)

    loss_fn = nn.CrossEntropyLoss().to(device)

    X_train, y_train = torch.tensor(X_train, dtype=torch.float32), y_cat_train
    X_test, y_test = torch.tensor(X_test, dtype=torch.float32), y_cat_test

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    history = {'epoch':[], 'train_loss':[], 'train_accuracy':[], 'valid_loss':[], 'valid_accuracy':[]}

    if leave_one_session_out:
        backbone_path = os.path.join(results_dir, f'model/leave_one_session_out_pre_offline_training_backbone.pt')
        classifier_path = os.path.join(results_dir, f'model/leave_one_session_out_pre_offline_training_classifier.pt')
    else:
        backbone_path = os.path.join(results_dir, f'model/pre_offline_training_backbone.pt')
        classifier_path = os.path.join(results_dir, f'model/pre_offline_training_classifier.pt')

    early_stopping = EarlyStopping(patience=10, verbose=True, delta=0.1, path=[backbone_path, classifier_path])


    # do training
    for epoch in range(n_epochs):
        #--------------------------------------------------------------------#
        #                              training                              #
        #--------------------------------------------------------------------#

        #model.train()
        train_loss_sum = 0.0
        train_correct_sum = 0.0
        train_total_sum = 0.0
        backbone.train()
        classifier.train()
        #for batch_X, batch_y in train_loader:
        for i_batch, (batch_X, batch_y) in enumerate(train_loader):
            #batch_X = batch_X.permute(0, 3, 1, 2)  # from NHWC to NCHW
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            #optimizer.zero_grad()
            optimizer_backbone.zero_grad()
            optimizer_classifier.zero_grad()
            #output = model(batch_X)
            x_embedding = backbone(batch_X)
            y_pred = classifier(x_embedding)
            #loss = nn.CrossEntropyLoss()(y_pred, torch.argmax(batch_y, dim=1))
            loss = loss_fn(y_pred, torch.argmax(batch_y, dim=1))
            loss.backward()
            optimizer_backbone.step()
            optimizer_classifier.step()

            y_pred_cls = torch.argmax(nn.Softmax(dim=1)(y_pred), dim=1).detach()
            train_loss_sum += loss.item()
            train_correct_sum += (y_pred_cls == torch.argmax(batch_y, dim=1)).sum().item()
            train_total_sum += y_pred_cls.shape[0]

            #print(f'[epoch-batch=[{epoch}-{i_batch}] loss: {train_loss_sum / (i_batch + 1)}, accuracy: {train_correct_sum / train_total_sum}')
        #print(f'[epoch-batch=[{epoch}-{i_batch}] loss: {train_loss_sum / (i_batch + 1)}, accuracy: {train_correct_sum / train_total_sum}')

        #--------------------------------------------------------------------#
        #                             validation                             #
        #--------------------------------------------------------------------#
        valid_loss_sum = 0.0
        valid_correct_sum = 0.0
        valid_total_sum = 0.0
        backbone.eval()
        classifier.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            #for batch_X, batch_y in test_loader:
            for i_batch, (batch_X, batch_y) in enumerate(test_loader):

                #batch_X = batch_X.permute(0, 3, 1, 2)  # from NHWC to NCHW
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                #output = model(batch_X)
                x_embedding = backbone(batch_X)
                y_pred = classifier(x_embedding)
                loss = loss_fn(y_pred, torch.argmax(batch_y, dim=1))
                y_pred_cls = torch.argmax(nn.Softmax(dim=1)(y_pred), dim=1).detach()

                #val_loss += nn.CrossEntropyLoss()(y_pred, torch.argmax(batch_y, dim=1)).item()
                #pred = y_pred.argmax(dim=1, keepdim=True)
                #correct += pred.eq(torch.argmax(batch_y, dim=1).view_as(pred)).sum().item()

                valid_loss_sum += loss.item()
                valid_correct_sum += (y_pred_cls == torch.argmax(batch_y, dim=1)).sum().item()
                valid_total_sum += y_pred_cls.shape[0]




        #--------------------------------------------------------------------#
        #                           epoch summary                            #
        #--------------------------------------------------------------------#
        #print(f'-------------------------- epoch-{epoch} --------------------------')
        print(f'epoch-{epoch}, train_loss: {train_loss_sum / len(train_loader):.3}, train_acc: {train_correct_sum / train_total_sum:.3}, valid_loss: {valid_loss_sum / len(test_loader):.3}, valid_acc: {valid_correct_sum / valid_total_sum:.3}')
        #print(f'valid_loss: {valid_loss_sum / len(test_loader):.3}, valid_acc: {valid_correct_sum / valid_total_sum:.3}')

        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss_sum / len(train_loader))
        history['train_accuracy'].append(train_correct_sum / train_total_sum)
        history['valid_loss'].append(valid_loss_sum / len(test_loader))
        history['valid_accuracy'].append(valid_correct_sum / valid_total_sum)

        # --------------------------------------------------------------------#
        #                        check early stopping                        #
        # --------------------------------------------------------------------#
        early_stopping(valid_loss_sum / len(test_loader), [backbone, classifier])
        if early_stopping.early_stop:
            print('early stopping.')
            break


        #val_loss /= len(test_loader.dataset)
        #accuracy = correct / len(test_loader.dataset)

        #print(f'Epoch {epoch + 1}/{n_epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}, Val Acc: {accuracy:.4f}')

        #scheduler.step()
        scheduler_backbone.step()
        scheduler_classifier.step()



    #acc[split_ctr] = save_results({'accuracy': [], 'val_accurary': [], 'loss': [], 'val_loss': []}, num_classes, n_ds, n_ch, T, split_ctr)  # Replace with actual history
    save_results({'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []})  # Replace with actual history
    #print(f'Fold {split_ctr}\t{acc[split_ctr, 0]}\t{acc[split_ctr, 1]}')

    #--------------------------------------------------------------------#
    #                            save history                            #
    #--------------------------------------------------------------------#
    if leave_one_session_out:
        history_path = os.path.join(results_dir, f'stats/leave_one_session_out_offline_training_history.pkl')
    else:
        history_path = os.path.join(results_dir, f'stats/offline_training_history.pkl')
    print(f'saving history to {history_path}...')
    with open(history_path, 'wb') as file:
        pkl.dump(history, file)

    print('pre-training finished.')

    # Save model
    if leave_one_session_out:
        torch.save(backbone.state_dict(), os.path.join(results_dir, f'model/leave_one_session_out_offline_training_backbone.pth'))
        torch.save(classifier.state_dict(), os.path.join(results_dir, f'model/leave_one_session_out_offline_training_classifier.pth'))
    else:
        torch.save(backbone.state_dict(), os.path.join(results_dir, f'model/offline_training_backbone.pth'))
        torch.save(classifier.state_dict(), os.path.join(results_dir, f'model/offline_training_classifier.pth'))

    #split_ctr += 1

    #print(f'AVG \t {acc[:, 0].mean()}\t{acc[:, 1].mean()}')
    #'''




Testing_offline_accuracy = []
Testing_online_accuracy = []

def testing_offline():
    #subjects_testing = range(1, 10)
    #testing_subjects = testing_subjects
    subjects = [x for x in testing_subjects if (x not in excluded_subjects)]

    for subject in subjects:
        npzfile = np.load(datapath_save_subject_data + f'{subject}.npz')
        X_test, y_test = npzfile['X_Train'], npzfile['y_Train']

        ####################################
        # The first half will be used for online training, the second half is used for testing
        ####################################
        #X_test = X_test[int(X_test.shape[0] * online_train_ratio):, :, :]
        #y_test = y_test[int(y_test.shape[0] * online_train_ratio):]


        ####################################
        # This rate (online_train_ratio) of test data will be used for online training, the left is used for testing
        ####################################
        npzfile = np.load(datapath_save_subject_data + 'online_training_testing/' + f'Online_testing_{subject}' + '.npz')
        X_test, y_test = npzfile['X_Train'], npzfile['y_Train']


        #'''
        X_test = np.expand_dims(X_test, axis=-1)
        X_test = np.reshape(X_test, (X_test.shape[0], 1, 64, 480))
        #print("X.shape: ", X_test.shape)
        n_samples = X_test.shape[2]
        y_cat_test = torch.tensor(y_test, dtype=torch.long)
        y_cat_test = nn.functional.one_hot(y_cat_test, 4)

        X_test, y_test = torch.tensor(X_test, dtype=torch.float32), y_cat_test

        test_dataset = TensorDataset(X_test, y_test)

        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


        # Load
        #model = model_eegnet.EEGNet_2()
        #model.load_state_dict(torch.load(os.path.join(results_dir, f'model/offline_training.pth')))
        #model.to(device)

        backbone = model_sep.EEGNet_backbone()
        classifier = model_sep.Classifier()
        backbone.load_state_dict(torch.load(os.path.join(results_dir, f'model/offline_training_backbone.pth')))
        classifier.load_state_dict(torch.load(os.path.join(results_dir, f'model/offline_training_classifier.pth')))
        backbone.to(device)
        classifier.to(device)

        #model.eval()
        backbone.eval()
        classifier.eval()
        correct = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                #batch_X = batch_X.permute(0, 3, 1, 2)  # from NHWC to NCHW
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                output = backbone(batch_X)
                y_pred = classifier(output)
                y_pred_cls = torch.argmax(nn.Softmax(dim=1)(y_pred), dim=1).detach()
                #pred = y_pred_cls.argmax(dim=1, keepdim=True)
                correct += y_pred_cls.eq(torch.argmax(batch_y, dim=1).view_as(y_pred_cls)).sum().item()
        accuracy = correct / len(test_loader.dataset)
        Testing_offline_accuracy.append(accuracy)
        print("Subject: {}, testing accuracy: {}".format(subject, accuracy))
    print("All Testing Subject, averaged offline testing accuracy: {}".format(np.mean(Testing_offline_accuracy)))
        #'''






def training_online():

    #subjects_testing = range(1, 10)
    subjects = [x for x in testing_subjects if (x not in excluded_subjects)]

    for subject in subjects:
        npzfile = np.load(datapath_save_subject_data + f'{subject}.npz')
        X_train_online, y_train_online = npzfile['X_Train'], npzfile['y_Train']

        ####################################
        # The first half will be used for online training, the second half is used for testing
        ####################################
        #X_train_online = X_train_online[0: int(X_train_online.shape[0] * online_train_ratio), :, :]
        #y_train_online = y_train_online[0: int(y_train_online.shape[0] * online_train_ratio)]

        ####################################
        # This rate (online_train_ratio) of test data will be used for online training, the left is used for testing
        ####################################
        npzfile = np.load(datapath_save_subject_data + 'online_training_testing/' + f'Online_training_{subject}' + '.npz')
        X_train_online, y_train_online = npzfile['X_Train'], npzfile['y_Train']


        print(X_train_online.shape)
        print(Counter(y_train_online))
        # '''
        X_train_online = np.expand_dims(X_train_online, axis=-1)
        X_train_online = np.reshape(X_train_online, (X_train_online.shape[0], 1, 64, 480))

        y_cat_train_online = torch.tensor(y_train_online, dtype=torch.long)
        y_cat_train_online = nn.functional.one_hot(y_cat_train_online, 4)

        X_train_online, y_train_online = torch.tensor(X_train_online, dtype=torch.float32), y_cat_train_online

        train_online_dataset = TensorDataset(X_train_online, y_train_online)

        #train_online_loader = DataLoader(train_online_dataset, batch_size=16, shuffle=False)
        train_online_loader = DataLoader(train_online_dataset, batch_size=1, shuffle=False)

        backbone = model_sep.EEGNet_backbone()
        classifier = model_sep.Classifier()
        backbone.load_state_dict(torch.load(os.path.join(results_dir, f'model/offline_training_backbone.pth')))
        classifier.load_state_dict(torch.load(os.path.join(results_dir, f'model/offline_training_classifier.pth')))
        backbone.to(device)
        classifier.to(device)


        # Set Learning Rate
        #optimizer = optim.Adam(model.parameters(), lr=0.01)
        #optimizer_backbone = optim.Adam(backbone.parameters(), lr=0.01)

        #optimizer_classifier = optim.Adam(classifier.parameters(), lr=0.005)
        optimizer_classifier = optim.SGD(classifier.parameters(), lr=0.002, momentum=0.9)
        #optimizer_classifier = optim.Adam(classifier.parameters(), lr=0.005, betas=(0.9, 0.99), eps=1e-8)

        # optimizer_backbone = optim.Adam(backbone.parameters(), lr=0.002, betas=(0.9, 0.99), eps=1e-08)

        #scheduler_backbone = StepLR(optimizer_backbone, step_size=30, gamma=0.2, verbose=0)
        #scheduler_classifier = StepLR(optimizer_classifier, step_size=30, gamma=0.2, verbose=0)

        loss_fn = nn.CrossEntropyLoss().to(device)

        # history = {'conf_mat': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []}


        # do training
        for epoch in range(online_epoch):
            #--------------------------------------------------------------------#
            #                              training                              #
            #--------------------------------------------------------------------#

            #model.train()
            train_loss_sum = 0.0
            train_correct_sum = 0.0
            train_total_sum = 0.0
            backbone.eval()
            classifier.train()
            #for batch_X, batch_y in train_loader:
            for i_batch, (batch_X, batch_y) in enumerate(train_online_loader):
                #batch_X = batch_X.permute(0, 3, 1, 2)  # from NHWC to NCHW
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                #optimizer.zero_grad()
                #optimizer_backbone.zero_grad()
                optimizer_classifier.zero_grad()
                #output = model(batch_X)
                x_embedding = backbone(batch_X)
                y_pred = classifier(x_embedding)
                #loss = nn.CrossEntropyLoss()(y_pred, torch.argmax(batch_y, dim=1))
                loss = loss_fn(y_pred, torch.argmax(batch_y, dim=1))
                loss.backward()
                #optimizer_backbone.step()
                optimizer_classifier.step()

                #y_pred_cls = torch.argmax(nn.Softmax(dim=1)(y_pred), dim=1).detach()
                #train_loss_sum += loss.item()
                #train_correct_sum += (y_pred_cls == torch.argmax(batch_y, dim=1)).sum().item()
                #train_total_sum += y_pred_cls.shape[0]

                #print(f'[epoch-batch=[{epoch}-{i_batch}] loss: {train_loss_sum / (i_batch + 1)}, accuracy: {train_correct_sum / train_total_sum}')
                #print(f'[epoch-batch=[{epoch}-{i_batch}] loss: {train_loss_sum / (i_batch + 1)}, accuracy: {train_correct_sum / train_total_sum}')

        print('online training finished.')

        # Save model

        torch.save(backbone.state_dict(), os.path.join(results_dir, f'model/online_training_backbone.pth'))
        torch.save(classifier.state_dict(), os.path.join(results_dir, f'model/online_training_classifier_{subject}.pth'))



def testing_online():
    #subjects_testing = range(1, 10)
    subjects = [x for x in testing_subjects if (x not in excluded_subjects)]

    i = 0
    for subject in subjects:
        i += 1
        #npzfile = np.load(datapath_save_subject_data + f'{subject}.npz')
        #X_test_online, y_test_online = npzfile['X_Train'], npzfile['y_Train']

        ####################################
        # The first half will be used for online training, the second half is used for testing
        ####################################
        #X_test_online = X_test_online[int(X_test_online.shape[0] * online_train_ratio):, :, :]
        #y_test_online = y_test_online[int(y_test_online.shape[0] * online_train_ratio):]

        ####################################
        # This rate (online_train_ratio) of test data will be used for online training, the left is used for testing
        ####################################
        npzfile = np.load(datapath_save_subject_data + 'online_training_testing/' + f'Online_testing_{subject}' + '.npz')
        X_test_online, y_test_online = npzfile['X_Train'], npzfile['y_Train']


        print(X_test_online.shape)
        print(Counter(y_test_online))


        #'''
        X_test_online = np.expand_dims(X_test_online, axis=-1)
        X_test_online = np.reshape(X_test_online, (X_test_online.shape[0], 1, 64, 480))

        y_cat_test_online = torch.tensor(y_test_online, dtype=torch.long)
        y_cat_test_online = nn.functional.one_hot(y_cat_test_online, 4)

        X_test_online, y_test_online = torch.tensor(X_test_online, dtype=torch.float32), y_cat_test_online

        test_dataset_online = TensorDataset(X_test_online, y_test_online)

        #test_loader_online = DataLoader(test_dataset_online, batch_size=16, shuffle=False)
        test_loader_online = DataLoader(test_dataset_online, batch_size=1, shuffle=False)


        backbone = model_sep.EEGNet_backbone()
        classifier = model_sep.Classifier()
        backbone.load_state_dict(torch.load(os.path.join(results_dir, f'model/online_training_backbone.pth')))
        classifier.load_state_dict(torch.load(os.path.join(results_dir, f'model/online_training_classifier_{subject}.pth')))
        backbone.to(device)
        classifier.to(device)

        backbone.eval()
        classifier.eval()
        correct = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader_online:
                #batch_X = batch_X.permute(0, 3, 1, 2)  # from NHWC to NCHW
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                output = backbone(batch_X)
                y_pred = classifier(output)
                y_pred_cls = torch.argmax(nn.Softmax(dim=1)(y_pred), dim=1).detach()
                #pred = y_pred_cls.argmax(dim=1, keepdim=True)
                correct += y_pred_cls.eq(torch.argmax(batch_y, dim=1).view_as(y_pred_cls)).sum().item()
        accuracy = correct / len(test_loader_online.dataset)
        Testing_online_accuracy.append(accuracy)
        print("Subject: {}, testing accuracy: {}, changed:  {}".format(subject, accuracy, accuracy - Testing_offline_accuracy[i-1]))
    print("All Testing Subject, averaged online testing accuracy: {}".format(np.mean(Testing_online_accuracy)))

        #'''


def combine_backbone_classifier():
    backbone = model_sep.EEGNet_backbone()
    classifier = model_sep.Classifier()
    backbone.load_state_dict(torch.load(os.path.join(results_dir, f'model/offline_training_backbone.pth')))
    classifier.load_state_dict(torch.load(os.path.join(results_dir, f'model/offline_training_classifier.pth')))
    backbone.to(device)
    classifier.to(device)

    backbone.eval()
    classifier.eval()
    combination = model_sep.EEGnet(backbone=backbone, classifier=classifier).to(device)
    dummy_input_backbone = torch.randn(1, 1, 64, 480).to(device)
    cmb_pre_onnx_path = os.path.join(results_dir, f'model/offline_training_combined.onnx')
    torch.onnx.export(combination,
                     dummy_input_backbone,
                     cmb_pre_onnx_path,
                     export_params=True,
                     opset_version=10,
                     do_constant_folding=True,
                     input_names=['modelInput'],
                     output_names=['modelOutput'],
                     dynamic_axes={'modelInput': {0: 'batch_size'}, 'modelOutput': {0: 'batch_size'}}
                     )
    print(f'combination has been converted to onnx and stored in {cmb_pre_onnx_path}')



if __name__ == '__main__':
    training_subjects = range(90, 100)
    validation_subjects = range(100, 110)
    subject_training_validation = list(training_subjects) + list(validation_subjects)
    excluded_subjects = [88, 92, 100, 104]
    testing_subjects = range(1, 90)
    online_train_ratio = 0.5
    online_epoch = 5

    #load_data()
    #load_data_leave_one_session_out()
    #training_offline(leave_one_session_out=True)
    #training_offline()
    testing_offline()
    #training_online()
    testing_online()
    #check_data_balance()
    #combine_backbone_classifier()

    #save_online_training_testing_data()