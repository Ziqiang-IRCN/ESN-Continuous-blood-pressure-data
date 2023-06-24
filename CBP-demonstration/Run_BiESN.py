import numpy as np
from dataloader import timeseries_loader
import ESN
from tqdm import tqdm
import torch
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
from utility import decision_function
from utility import random_index_generator
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--random_selection', type=int, default=10)
parser.add_argument('--random_init', type=int, default=10)
parser.add_argument('--num_series', type = int, default = 10)
parser.add_argument('--num_select_seg',type=int,default=100)
parser.add_argument('--res_size',type=int,default=100)
parser.add_argument('--rho', type = float, default = 1.3)
parser.add_argument('--alpha', type = float, default = 0.5)
parser.add_argument('--training_ratio', type = float, default = 0.8)
parser.add_argument('--validation_ratio', type=float, default = 0.1)
parser.add_argument('--testing_ratio',type=float,default=0.1)
parser.add_argument('--mode',type=str,default="paper",help="Same with paper's settings")
args = parser.parse_args()

random_selection = args.random_selection
random_init = args.random_init
num_series = args.num_series
num_select_seg = args.num_select_seg
res_size = args.res_size
rho = args.rho
alpha = args.alpha
training_ratio = args.training_ratio
validation_ratio = args.validation_ratio
testing_ratio = args.testing_ratio

aggregated_y = np.linspace(0,num_series, num = num_series, endpoint=False)
if args.mode == "paper":
    random_dict = np.load('./index_dict/random_dict_index.npy', allow_pickle=True).item()

Bi_ESN_acc = []
Bi_ESN_acc_agg_1 = []
Bi_ESN_acc_agg_2 = []
Bi_ESN_acc_agg_3 = []
Bi_ESN_acc_agg_4 = []
Bi_ESN_acc_agg_5 = []
Bi_ESN_acc_agg_6 = []
Bi_ESN_acc_agg_7 = []
Bi_ESN_acc_agg_8 = []
Bi_ESN_acc_agg_9 = []
Bi_ESN_acc_agg_10 = []

for s in tqdm(range(random_selection)):
    if args.mode == "paper":
        random_list = random_dict['random_index_{}'.format(s)]
    else:
        random_list = random_index_generator(num_index = num_series)
    training_length_per_series = int(num_select_seg*training_ratio)
    validation_length_per_series = int(num_select_seg*validation_ratio)
    testing_length_per_series = int(num_select_seg*testing_ratio)
    dataloader = timeseries_loader(random_list = random_list,num_series=num_series,validation=False,length=200000,num_select_seg=num_select_seg,flipping=False, training_ratio=training_ratio, validation_ratio = validation_ratio, testing_ratio=testing_ratio,peakwindow = 0.261,beatwindow =0.767)
    dataloader.from_3d_to_2d()
    dataloader_re = timeseries_loader(random_list = random_list,num_series=num_series,validation=False,length=200000,num_select_seg=num_select_seg,flipping=True, training_ratio=training_ratio, validation_ratio = validation_ratio, testing_ratio=testing_ratio, peakwindow = 0.261,beatwindow =0.767)
    dataloader_re.from_3d_to_2d()
    for r in tqdm(range(random_init)):
        train_state_BiESN = np.zeros((training_length_per_series * num_series, dataloader.min_length, 2 * res_size))
        test_state_BiESN = np.zeros((testing_length_per_series * num_series, dataloader.min_length, 2 * res_size))
        ESN_encoder_1 = ESN.ESN(input=1, reservoir=res_size, sr=rho, density=1, scale_in=1.0, leaking_rate=1, Nepochs=10,
                                eta=1e-3, mu=0, sigma=0.1, threshold=0.1, W_assign='Uniform', Win_assign='Uniform')
        ESN_encoder_2 = ESN.ESN(input=1, reservoir=res_size, sr=rho, density=1, scale_in=1.0, leaking_rate=1, Nepochs=10,
                                eta=1e-3, mu=0, sigma=0.1, threshold=0.1, W_assign='Uniform', Win_assign='Uniform')

        for i in range(len(dataloader.train_x)):
            for j in range(len(dataloader.train_x[i])):
                input = torch.Tensor(dataloader.train_x[i][j])
                input = torch.unsqueeze(input, 1)
                input_re = torch.Tensor(dataloader_re.train_x[i][j])
                input_re = torch.unsqueeze(input_re, 1)
                state_Bi_ESN_1 = ESN.state_transform(ESN_encoder_1(input, h_0=None, useIP=False))
                state_Bi_ESN_1_re = ESN.state_transform(ESN_encoder_2(input_re, h_0=None, useIP=False))
                train_state_BiESN[i * training_length_per_series + j,] = np.concatenate((state_Bi_ESN_1.numpy()[
                                                                                         state_Bi_ESN_1.shape[
                                                                                             0] - dataloader.min_length:],
                                                                                         state_Bi_ESN_1_re.numpy()[
                                                                                         state_Bi_ESN_1_re.shape[
                                                                                             0] - dataloader.min_length:]),
                                                                                        1)
        for i in range(len(dataloader.test_x)):
            for j in range(len(dataloader.test_x[i])):
                input = torch.Tensor(dataloader.test_x[i][j])
                input = torch.unsqueeze(input, 1)
                input_re = torch.Tensor(dataloader_re.test_x[i][j])
                input_re = torch.unsqueeze(input_re, 1)
                state_Bi_ESN_1 = ESN.state_transform(ESN_encoder_1(input, h_0=None, useIP=False))
                state_Bi_ESN_1_re = ESN.state_transform(ESN_encoder_2(input_re, h_0=None, useIP=False))
                test_state_BiESN[i * testing_length_per_series + j,] = np.concatenate((state_Bi_ESN_1.numpy()[
                                                                                       state_Bi_ESN_1.shape[
                                                                                           0] - dataloader.min_length:],
                                                                                       state_Bi_ESN_1_re.numpy()[
                                                                                       state_Bi_ESN_1_re.shape[
                                                                                           0] - dataloader.min_length:]),
                                                                                      1)
        LR_Bi = RidgeClassifier(alpha=alpha, solver='svd')
        LR_Bi.fit(train_state_BiESN.reshape(train_state_BiESN.shape[0], -1), dataloader.train_y)
        decision_y = LR_Bi.decision_function(test_state_BiESN.reshape(test_state_BiESN.shape[0], -1))
        Bi_ESN_acc_agg_1.append(accuracy_score(y_true=aggregated_y,
                                               y_pred=decision_function(decision_value=decision_y, window=1,
                                                                        num_series=10)))
        Bi_ESN_acc_agg_2.append(accuracy_score(y_true=aggregated_y,
                                               y_pred=decision_function(decision_value=decision_y, window=2,
                                                                        num_series=10)))
        Bi_ESN_acc_agg_3.append(accuracy_score(y_true=aggregated_y,
                                               y_pred=decision_function(decision_value=decision_y, window=3,
                                                                        num_series=10)))
        Bi_ESN_acc_agg_4.append(accuracy_score(y_true=aggregated_y,
                                               y_pred=decision_function(decision_value=decision_y, window=4,
                                                                        num_series=10)))
        Bi_ESN_acc_agg_5.append(accuracy_score(y_true=aggregated_y,
                                               y_pred=decision_function(decision_value=decision_y, window=5,
                                                                        num_series=10)))
        Bi_ESN_acc_agg_6.append(accuracy_score(y_true=aggregated_y,
                                               y_pred=decision_function(decision_value=decision_y, window=6,
                                                                        num_series=10)))
        Bi_ESN_acc_agg_7.append(accuracy_score(y_true=aggregated_y,
                                               y_pred=decision_function(decision_value=decision_y, window=7,
                                                                        num_series=10)))
        Bi_ESN_acc_agg_8.append(accuracy_score(y_true=aggregated_y,
                                               y_pred=decision_function(decision_value=decision_y, window=8,
                                                                        num_series=10)))
        Bi_ESN_acc_agg_9.append(accuracy_score(y_true=aggregated_y,
                                               y_pred=decision_function(decision_value=decision_y, window=9,
                                                                        num_series=10)))
        Bi_ESN_acc_agg_10.append(accuracy_score(y_true=aggregated_y,
                                                y_pred=decision_function(decision_value=decision_y, window=10,
                                                                         num_series=10)))
        pred_y = LR_Bi.predict(test_state_BiESN.reshape(test_state_BiESN.shape[0], -1))
        Bi_ESN_acc.append(accuracy_score(y_true=dataloader.test_y, y_pred=pred_y))

print('Segment-by-segment accuracy:' + str(np.mean(np.array(Bi_ESN_acc)))+ 'std: '+ str(np.std(np.array(Bi_ESN_acc))))
print('Aggregated accuracy with N_te=1 ' + str(np.mean(np.array(Bi_ESN_acc_agg_1))))
print('Aggregated accuracy with N_te=2 ' + str(np.mean(np.array(Bi_ESN_acc_agg_2))))
print('Aggregated accuracy with N_te=3 ' + str(np.mean(np.array(Bi_ESN_acc_agg_3))))
print('Aggregated accuracy with N_te=4 ' + str(np.mean(np.array(Bi_ESN_acc_agg_4))))
print('Aggregated accuracy with N_te=5 ' + str(np.mean(np.array(Bi_ESN_acc_agg_5))))
print('Aggregated accuracy with N_te=6 ' + str(np.mean(np.array(Bi_ESN_acc_agg_6))))
print('Aggregated accuracy with N_te=7 ' + str(np.mean(np.array(Bi_ESN_acc_agg_7))))
print('Aggregated accuracy with N_te=8 ' + str(np.mean(np.array(Bi_ESN_acc_agg_8))))
print('Aggregated accuracy with N_te=9 ' + str(np.mean(np.array(Bi_ESN_acc_agg_9))))
print('Aggregated accuracy with N_te=10 ' + str(np.mean(np.array(Bi_ESN_acc_agg_10))))




