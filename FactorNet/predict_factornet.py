# import transformers
# from transformers import AdamW, get_linear_schedule_with_warmup
import torch, glob
import numpy as np
import pandas as pd
# import seaborn as sns
# from pylab import rcParams
# import matplotlib.pyplot as plt
# from matplotlib import rc
# from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
# from textwrap import wrap
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import swifter, pickle
import sys, random
import time
import torch.nn.functional as F
from tqdm import tqdm

if len(sys.argv) < 4:
    print('python file.py input_fname model_pth output_fname')
    exit(0)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)

input_fname = sys.argv[1]
model_pth = sys.argv[2]
output_fname = sys.argv[3]

print('input_fname:', input_fname,
      'model_pth:', model_pth,
      'output_fname:', output_fname
      )

BATCH_SIZE = 15000

df = pd.read_csv(input_fname, header=None)
df.columns = ['tag', 'species', 'sequence', 'seq_len']
print(df.head())
print(df.shape)

df = df[df.seq_len>=700]
print('df:', df.shape)

#list_models = [item.strip() for item in open('list-models').readlines()]
#print('list_models:', list_models)
# exit(0)

# read dataset
mapper = {'A': [True, False, False, False],
              'C': [False, True, False, False],
              'G': [False, False, True, False],
              'T': [False, False, False, True],
              'P': [False, False, False, False]
              }

rev_comp_mapper = {'A': 'T',
                   'C': 'G',
                   'G': 'C',
                   'T': 'A'
                   }


class FactorNet(nn.Module):

    def __init__(self, device):
        super(FactorNet, self).__init__()
        self.device = device

        # self.OUT_CH = OUT_CH
        # self.FC = FC

        # conv layer 1
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=32, stride=1, kernel_size=26,
                               padding=0)
        self.relu1 = nn.ReLU(inplace=False)
        # define dropout layer in __init__
        self.drop_layer1 = nn.Dropout(p=0.1)

        # pooling layer 1
        self.max_pool = nn.MaxPool1d(kernel_size=13, stride=13)

        # bidirectional lstm layer 1
        self.lstm = nn.LSTM(input_size=75,
                            hidden_size=32,
                            num_layers=1,
                            bidirectional=True,
                            batch_first=True
                            )
        self.drop_layer2 = nn.Dropout(p=0.5)

        # fully connected layer 1

        self.fc1 = nn.Linear(32 * 64, 128)
        self.relu2 = nn.ReLU(inplace=False)
        self.drop_layer3 = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(128, 1)

        self.sigmoid = nn.Sigmoid()

        # Convolution1D(input_dim=4, nb_filter=32,
        #               filter_length=26, border_mode='valid', activation='relu',
        #               subsample_length=1),
        # Dropout(0.1),
        # TimeDistributed(Dense(num_motifs, activation='relu')),
        # MaxPooling1D(pool_length=w2, stride=w2),
        # Bidirectional(LSTM(num_recurrent, dropout_W=0.1, dropout_U=0.1, return_sequences=True)),
        # Dropout(dropout_rate),
        # Flatten(),
        # Dense(num_dense, activation='relu'),
        # Dropout(dropout_rate),
        # Dense(num_tfs, activation='sigmoid')

    def forward_one(self, x):
        # if self.device == "cuda":
        #   x = x.type(torch.cuda.FloatTensor)
        # else:
        #   x.type(torch.FloatTensor)
        # print('x:', x.shape)
        x = self.conv1(x)
        # print('after conv1d x:', x.shape)
        x = self.relu1(x)
        x = self.drop_layer1(x)
        x = self.max_pool(x)
        # print('after max pool x:', x.shape)

        x, (hn, cn) = self.lstm(x)
        # print('x:', x.shape)
        x = self.drop_layer2(x)
        # print('x:', x.shape)

        x = x.reshape(-1, 32 * 64)
        # print('x:', x.shape)

        x = self.fc1(x)
        # print('x:', x.shape)
        x = self.relu2(x)
        x = self.drop_layer3(x)

        x = self.fc2(x)
        # print('x:', x.shape)

        x = self.sigmoid(x)
        # print('x:', x.shape)

        return x

    def forward(self, input1, input2):
        output1 = self.forward_one(input1)  # .reshape(-1)
        output2 = self.forward_one(input2)  # .reshape(-1)
        # print('output1:', output1)
        # print('output2:', output2)
        cat_output = torch.cat((output1, output2), 1)
        # print('cat_output:', cat_output)
        out = torch.mean(cat_output, dim=1)
        # print(out); exit(0)

        return out

# load model

class_names = [0, 1]
OUT_CH = 128
FC = 32
DROPOUT = 0
n_classes = 2
# exit(0)


def get_features(df):
    # fix length to max len
    max_len = 1000

    df['fixed_len_seq'] = df['sequence']. \
        swifter.apply(lambda x: x + ('P' * (max_len - len(x))) if len(x) <= max_len else x[:max_len])
    # print(df['fixed_len_seq'])

    # get features
    df['features'] = df['fixed_len_seq']. \
        swifter.apply(lambda row: np.array([mapper[item] for item in row], dtype=np.bool_).reshape(-1, 4).T)
    # print(df['features'])
    # exit(0)

    # get rev comp
    df['rev_comp'] = df['sequence']. \
        swifter.apply(lambda x: x[::-1])
    # print(df.rev_comp); exit(0)

    df['rev_comp'] = df['rev_comp']. \
        swifter.apply(lambda x: ''.join([rev_comp_mapper[item] for item in x]))
    # print(df.rev_comp)

    df['rev_comp_fixed_len_seq'] = df['rev_comp']. \
        swifter.apply(lambda x: x + ('P' * (max_len - len(x))) if len(x) <= max_len else x[:max_len])
    # print(df.rev_comp_fixed_len_seq)

    df['rev_comp_features'] = df['rev_comp_fixed_len_seq']. \
        swifter.apply(lambda row: np.array([mapper[item] for item in row], dtype=np.bool_).reshape(-1, 4).T)
    # print(df.rev_comp_features)

    return df['features'].values, df['rev_comp_features'].values


def get_predictions(df, model):
    # get features from sequences
    input1, input2 = get_features(df)
    input1 = torch.tensor(np.stack(input1), dtype=torch.float).to(device)
    input2 = torch.tensor(np.stack(input2), dtype=torch.float).to(device)
    # print('input1:', input1.shape)
    # print('input2:', input2.shape); exit(0)
    # predict on features
    outputs = model(input1, input2).reshape(-1)
    print('outputs:', outputs.shape)
    return outputs

#for model_pth in list_models:

print('model_pth:', model_pth)

tf = model_pth.split('/')[-1].split('-')[2]
print('tf:', tf)

cell_type = '-'.join(model_pth.split('/')[-1].split('-')[3:]).split('.')[0]
print('cell_type:', cell_type)


print('model_pth:', model_pth,
      'tf:', tf,
      'cell_type:', cell_type,
      'df:', df.shape
      )


print('df:', df.shape)
model = FactorNet(device=device, )
model = model.to(device)
#print('device:', device, type(device), str(device)); exit(0)
if str(device) == 'cpu':
    model.load_state_dict(torch.load(model_pth, map_location=torch.device('cpu')))
else:
    model.load_state_dict(torch.load(model_pth))
print('Done. Model loaded')

def get_batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


pred_df = pd.DataFrame()
for counter, batch_ids in enumerate(get_batch(df.index, BATCH_SIZE)):
    # print('counter:', counter,
    #       'batch_ids:', len(batch_ids))

    # compute predictions
    curr_curr_df = df.loc[batch_ids, :]
    print('curr_curr_df:', curr_curr_df.shape)
    curr_predictions = get_predictions(curr_curr_df, model)
    curr_curr_df['predictions'] = curr_predictions.cpu().detach().numpy()
    del curr_predictions
    torch.cuda.empty_cache()
    curr_curr_df['label'] = curr_curr_df['tag'].apply(lambda x: x.split('-')[-1])
    pred_df = pd.concat([pred_df, curr_curr_df[['tag', 'species', 'predictions', 'label']]])
    print('pred_df:', pred_df.shape)
    # print('Time:', time.time())
    # reset curr_lines and curr_df
    del curr_curr_df

print('pred_df:', pred_df.shape)
pred_df.to_csv(output_fname, index=False)

del model

