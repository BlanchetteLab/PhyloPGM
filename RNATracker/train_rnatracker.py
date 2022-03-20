import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
import swifter
import sys, random
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

if len(sys.argv)<3:
    print('python file.py input_fname model_pth')
    exit(0)

#expt_name = sys.argv[1]
#fname = expt_name+'/ortho-train.csv'
fname = sys.argv[1]
#model_pth = expt_name+'/base_model.pth'
model_pth = sys.argv[2]
print('fname:', fname,
      'model_pth:', model_pth)

# exit(0)

BATCH_SIZE=1000
EARLY_STOP=20
EPOCHS=200

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# read data
given_df = pd.read_csv(fname, header=None)
given_df.columns = ['chrom', 'start', 'stop', 'tag', 'species', 'sequence', 'seq_len', 'label', 'strand']
print('given_df:', given_df.shape)
df_hg38 = given_df[given_df.species=='hg38']
print('df_hg38:', df_hg38.shape)
df_train, df_val = train_test_split(df_hg38, test_size=0.2, random_state=42)
print('df_train:', df_train.shape,
      'df_val:', df_val.shape
      )

ortho_train = pd.merge(given_df, df_train, on='tag', how='right')
ortho_train = ortho_train[['chrom_x', 'start_x', 'stop_x', 'tag', 'species_x', 'sequence_x',
       'seq_len_x', 'label_x', 'strand_x']]
ortho_train.columns = ['chrom', 'start', 'stop', 'tag', 'species', 'sequence',
       'seq_len', 'label', 'strand']

ortho_val = pd.merge(given_df, df_val, on='tag', how='right')
ortho_val = ortho_val[['chrom_x', 'start_x', 'stop_x', 'tag', 'species_x', 'sequence_x',
       'seq_len_x', 'label_x', 'strand_x']]
ortho_val.columns = ['chrom', 'start', 'stop', 'tag', 'species', 'sequence',
       'seq_len', 'label', 'strand']


#ortho_train.to_csv(expt_name+'/sampled-ortho-train.csv',index=False)
#ortho_val.to_csv(expt_name+'/sampled-ortho-val.csv', index=False)
print('ortho_train:', ortho_train.shape,
      'ortho_val:', ortho_val.shape
      )
# exit(0)


class SequenceDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets
        # print('features:',features.shape, features[0].shape, type(features), type(features[0]))
        # print('targets:', targets.shape); exit(0)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, item):
        review = self.features[item]
        target = self.targets[item]

        return {
            'review_text': torch.tensor(review, dtype=torch.float),
            'targets': torch.tensor(target, dtype=torch.long)
        }


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


NUM_WORKERS = 0 if str(device) == 'cpu' else 16
def create_data_loader(df, batch_size):
    # print('ok1')
    ds = SequenceDataset(
        # features=df.features.to_numpy(),
        features=np.stack(df['features'].values),
        targets=df.label.to_numpy(),
    )
    # print('ok2'); exit(0)
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=NUM_WORKERS,
        shuffle=True
    )

df_train = get_df(df_train)
df_val = get_df(df_val)

train_data_loader = create_data_loader(df_train, 1000)
val_data_loader = create_data_loader(df_val, 1000)

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

optimizer = torch.optim.Adam(model.parameters())


def train_epoch(
        model,
        data_loader,
        optimizer,
        device,
        n_examples,
):
    model = model.train()

    losses = []
    correct_predictions = 0
    for d in data_loader:
        features = d['review_text'].to(device)
        targets = d["targets"].to(device)

        # print('features:', features)
        # print('targets:', targets)
        # exit(0)

        outputs = model(features).reshape(-1)
        preds = (outputs >= 0.5).float() * 1
        loss = F.binary_cross_entropy(outputs, targets.float())

        correct_predictions += torch.sum(preds == targets)
        # correct_predictions = loss #+= torch.sum(preds == targets)
        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        # scheduler.step()
        optimizer.zero_grad()
    return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0
    with torch.no_grad():
        for d in data_loader:
            features = d['review_text'].to(device)
            targets = d["targets"].to(device)

            outputs = model(features).reshape(-1)
            preds = (outputs >= 0.5).float() * 1
            loss = F.binary_cross_entropy(outputs, targets.float())

            correct_predictions += torch.sum(preds == targets)

            losses.append(loss.item())
    return correct_predictions.double() / n_examples, np.mean(losses)


# %%time
best_accuracy = -np.inf
best_epoch=-np.inf
best_loss = np.inf
did_not_improve=0
toggler = 0; flag = True
for epoch in range(EPOCHS):
    O_CTR = random.randint(50, 150)
    print('O_CTR:', O_CTR)
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)
    # exit(0)

    run_label = True

    train_acc, train_loss = train_epoch(
        model,
        train_data_loader,
        optimizer,
        device,
        len(df_train),
    )
    print(f'Train loss {train_loss} accuracy {train_acc}')
    # exit(0)
    val_acc, val_loss = eval_model(
        model,
        val_data_loader,
        device,
        len(df_val)
    )
    print(f'Val   loss {val_loss} accuracy {val_acc}')

    if best_loss > val_loss :
        did_not_improve = 0
        torch.save(model.state_dict(), model_pth)
        best_accuracy = val_acc
        best_epoch=epoch
        best_loss = val_loss
    else:
        did_not_improve += 1

    if did_not_improve == EARLY_STOP:
        print('Did not Improve for :', EARLY_STOP, 'iterations')
        break


print('best_epoch:', best_epoch, 'best_accuracy:', best_accuracy, 'best_loss:', best_loss)



