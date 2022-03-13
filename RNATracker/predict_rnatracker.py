import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
import swifter
import sys, random
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

fname = sys.argv[1]
model_pth = sys.argv[2]
out_name = sys.argv[3]
print('fname:', fname,
      'model_pth:', model_pth,
      'out_name:', out_name
      )

# exit(0)

BATCH_SIZE=1000
EARLY_STOP=20
EPOCHS=200

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


def get_df(df):

    num_pos = df[df['label']==1].shape[0]
    num_neg = df[df['label']==0].shape[0]

    print('num_pos:', num_pos, 'num_neg:', num_neg)

    # print('df:', df.head())

    # get rev comp
    df['req_sequence'] = df[['sequence', 'strand']]. \
        swifter.apply(lambda x: x[0][::-1] if x[1] == '-' else x[0],
                      axis =1)

    # print(df.iloc[0, :])
    # print(df.iloc[0, :])

    df['req_sequence'] = df[['req_sequence', 'strand']]. \
        swifter.apply(
        lambda x: ''.join([rev_comp_mapper[item] for item in x[0]] if x[1]=='-' else x[0]),
        axis = 1
    )

    # print(df.iloc[0, :])
    # print(df.iloc[0, :])
    # # exit(0)

    max_len = 101
    df['fixed_len_seq'] = df['req_sequence']. \
        swifter.apply(lambda x: ('P' * (max_len - len(x))) + x if len(x) <= max_len else x[-max_len:])

    # get features
    df['features'] = df['fixed_len_seq']. \
        swifter.apply(lambda row: np.array([mapper[item] for item in row], dtype=np.bool_).reshape(-1, 4).T)

    # print(df.iloc[0, :])
    # # exit(0)


    return df


n_classes = 1

class RNATracker(nn.Module):
    def __init__(self, device, OUT_CH, FC, DROPOUT, decoder_unroll_steps=10):
        super(RNATracker, self).__init__()
        self.device = device

        self.OUT_CH = OUT_CH
        self.FC = FC

        # First Conv layer
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=self.OUT_CH, stride=1, kernel_size=10,
                               padding=5,
                               )
        self.relu1 = nn.ReLU(inplace=False)

        # optional
        self.max_pool1 = nn.MaxPool1d(kernel_size=3, stride=3) #, padding=1)
        self.drop_layer1 = nn.Dropout(p=DROPOUT)

        # Second Conv layer
        self.conv2 = nn.Conv1d(in_channels=self.OUT_CH, out_channels=self.OUT_CH, stride=1, kernel_size=10,
                               padding=5)
        self.relu2 = nn.ReLU(inplace=False)

        self.max_pool2 = nn.MaxPool1d(kernel_size=3, stride=3) #, padding=1)
        self.drop_layer2 = nn.Dropout(p=DROPOUT)

        # bidirectional LSTM
        # self.lstm = nn.LSTM(input_size=102, hidden_size=100, num_layers=1, bidirectional=True, batch_first=True)
        # self.lstm = nn.LSTM(input_size=401, hidden_size=100, num_layers=1, bidirectional=True, batch_first=True)
        # self.lstm = nn.LSTM(input_size=444, hidden_size=100, num_layers=1, bidirectional=True, batch_first=True)
        self.lstm = nn.LSTM(input_size=11, hidden_size=100, num_layers=1, bidirectional=True, batch_first=True)
        # self.lstm = nn.LSTM(input_size=101, hidden_size=100, num_layers=1, bidirectional=True, batch_first=True)

        # TODO
        # decoder lstm unroll it for fixed amount time, g=10, with attention
        # 200 is the hidden states dimensionality from the bi-directional lstm
        # 400 is the dimension of inputs to the decoder lstm
        self.decoder_lstm = nn.LSTMCell(400, 200)
        self.decoder_unroll_steps = decoder_unroll_steps

        # self.lstm = nn.LSTM(101, 101, bidirectional=True)
        self.avg_pool1 = nn.AvgPool1d(200)
        self.drop_layer3 = nn.Dropout(p=DROPOUT)


        # output layer
        # self.out = nn.Linear(self.OUT_CH, n_classes)
        self.out = nn.Linear(400, n_classes)
        # self.out = nn.Linear(400, 2) #n_classes)

        # sigmoid layer
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.device == "cuda":
            x = x.type(torch.cuda.FloatTensor)
        else:
            x.type(torch.FloatTensor)

        # x input is of shape [batch_size, length, in_channels]
        # you want to permute the last two dimensions as per pytorch's requirement
        # use transpose instead

        # GPUtil.showUtilization()

        # first conv layer
        # print('x:', x.shape)
        x = self.conv1(x)
        # print('x:', x.shape)
        x = self.relu1(x)
        # print('x:', x.shape)
        x = self.max_pool1(x)
        # print('x: after max_pool1', x.shape)
        x = self.drop_layer1(x)
        # print('x:', x.shape)
        # exit(0)

        # second conv layer
        x = self.conv2(x)
        # print('after second conv x:', x.shape)
        x = self.relu2(x)
        # print('x:', x.shape)
        x = self.max_pool2(x)
        # print('x:', x.shape)
        x = self.drop_layer2(x)
        # print('before lstm x:', x.shape)
        # exit(0)

        # here you have x of shape [batch_size, out_channels, length]

        # third layer lstm
        x, (hn, cn) = self.lstm(x)

        # print('x:', x.shape)#; exit(0)
        #
        batch_size = x.size(0)
        nb_features = x.size(-1) * 2
        hn = hn.transpose(1, 2).reshape(batch_size, -1)
        cn = cn.transpose(1, 2).reshape(batch_size, -1)

        # set2set pooling unrolling deocder lstm
        token = torch.zeros(batch_size, nb_features, device=self.device)

        for _ in range(self.decoder_unroll_steps):
            (hn, cn) = self.decoder_lstm(token, (hn, cn))

            # compute attention
            scores = torch.matmul(x, hn[:, None, :].transpose(1, 2))[:, :, 0]
            attention_weights = torch.softmax(scores, dim=-1)
            context_vector = torch.sum(x * attention_weights[:, :, None], dim=1)
            token = torch.cat([context_vector, hn], dim=-1)

        # # print('before avg pool1:', x.shape)
        # x = self.avg_pool1(x)
        # # print('after avg pool1:', x.shape); exit(0)
        x = self.drop_layer3(token)
        #
        # print('after drop_layer3 x:', x.shape)
        # exit(0)


        # x = self.drop_layer3(x)

        # # if using avg pool
        # x = x.reshape(-1, self.OUT_CH)
        # # print('x:', x.shape); exit(0)

        x = self.out(x)

        # print('x:', x.shape); #exit(0)

        # TODO use softmax (maybe)
        x = self.sigmoid(x)
        # x = F.softmax(x)

        # print('after sigmoid x:', x.shape); exit(0)

        return x


model = RNATracker(device, 32, 32, 0.1)
model = model.to(device)
model.load_state_dict(torch.load(model_pth))
print('Done. Model loaded')
model = model.eval()


df_vals = []
df_data = pd.DataFrame()
for line_index, line in enumerate(open(fname)):
    if line_index == 0:
        continue
    line = line.strip().split(',')

    df_vals.append(line)

    if len(df_vals) % 2000 == 0:
        print('line_index:', line_index)
        df = pd.DataFrame(df_vals)
        df.columns = ['chrom', 'start', 'stop', 'tag',
                      'species', 'sequence', 'seq_len', 'label',
                      'strand'
                      ]
        # df.columns = ['tag', 'species', 'sequence', 'label']
        df['label'] = df['label'].astype(float)
        # df['strand'] = df['tag'].apply(lambda x: x.split('_')[-1])
        df = get_df(df)
        # get predictions
        features = torch.tensor(np.stack(df.features.values),
                                dtype=torch.float).to(device)
        predictions = model(features).reshape(-1).cpu().detach().numpy()
        df['predictions'] = predictions
        df_data = pd.concat([df_data,
                             df[['tag',  'species', 'predictions', 'label']]
                             ])
        del df
        df_vals = []

if len(df_vals) > 0:
    df = pd.DataFrame(df_vals)
    df.columns = ['chrom', 'start', 'stop', 'tag',
                  'species', 'sequence', 'seq_len', 'label',
                  'strand'
                  ]
    # df.columns = ['tag', 'species', 'sequence', 'label']
    df['label'] = df['label'].astype(float)
    # df['strand'] = df['tag'].apply(lambda x: x.split('_')[-1])

    df = get_df(df)

    # get predictions
    features = torch.tensor(np.stack(df.features.values),
                            dtype=torch.float).to(device)
    predictions = model(features).reshape(-1).cpu().detach().numpy()
    df['predictions'] = predictions
    df_data = pd.concat([df_data,
                         df[['tag', 'species', 'predictions', 'label']]
                         ])
    del df
    df_vals = []

print('df_data:', df_data.shape)

df_data = df_data[['tag', 'species', 'predictions', 'label']]
df_data.to_csv(out_name,
               index=False
               )
