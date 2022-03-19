# import transformers
# from transformers import AdamW, get_linear_schedule_with_warmup
import torch
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
import swifter
import sys
import torch.nn.functional as F


if len(sys.argv)<6:
    print('python file.py train_csv val_csv test_csv batch_size epochs model_pth')
    exit(0)


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_csv = sys.argv[1]
val_csv = sys.argv[2]
test_csv = sys.argv[3]
BATCH_SIZE = int(sys.argv[4])
EPOCHS = int(sys.argv[5])
model_pth = sys.argv[6]


class SequenceDataset(Dataset):
    def __init__(self, features, rev_comp_features, targets):
        self.features = features
        self.rev_comp_features = rev_comp_features
        self.targets = targets
        # print('features:',features.shape, features[0].shape, type(features), type(features[0]))
        # print('targets:', targets.shape); exit(0)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, item):
        review = self.features[item]
        review_rev_comp = self.rev_comp_features[item]
        target = self.targets[item]

        return {
        'review_text': torch.tensor(review, dtype=torch.float),
        'review_text_rev_comp': torch.tensor(review_rev_comp, dtype=torch.float),
        'targets': torch.tensor(target, dtype=torch.long)
        }

# read dataset
def get_df(fname, species='hg38'):
    df = pd.read_csv(fname, sep=',', header=None,)
    print('df:', df.head())
    print('df:', df.shape)

    if species == 'hg38':
        df = df[df[1]=='hg38']

    #print('df:', df.head())
    #print('df:', df.shape)
    #exit(0)

    #df['label'] = df[0].swifter.apply(lambda x: 1 if '-1-chr' in x else 0,
    #                                  axis=1)
    df['label'] = df[0].swifter.apply(lambda x: float(x.split('-')[-1]))

    #print('df:', df.head())
    #print('df:', df.shape)
    #exit(0)
    
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

    # fix length to max len
    max_len = 1000

    df['fixed_len_seq'] = df[2]. \
      swifter.apply(lambda x: x + ('P'* (1000-len(x))) if len(x)<=1000 else x[:1000])
    # df['fixed_len_seq'] = df[2]. \
    #   swifter.apply(lambda x: x + ('P'* (1000-len(x))) if len(x)<=1000 else x[(len(x)-1000)//2:((len(x)-1000)//2)+1000])

    # get features
    df['features'] = df['fixed_len_seq'].\
        swifter.apply(lambda row: np.array([mapper[item] for item in row], dtype=np.bool_).reshape(-1, 4).T)

    # get rev comp
    df['rev_comp'] = df[2].\
        swifter.apply(lambda x: x[::-1])

    df['rev_comp'] = df['rev_comp'].\
        swifter.apply(lambda x: ''.join([rev_comp_mapper[item] for item in x]))

    df['rev_comp_fixed_len_seq'] = df['rev_comp']. \
      swifter.apply(lambda x: x + ('P'* (1000-len(x))) if len(x)<=1000 else x[:1000])
    # df['rev_comp_fixed_len_seq'] = df['rev_comp']. \
    #   swifter.apply(lambda x: x + ('P'* (1000-len(x))) if len(x)<=1000 else x[(len(x)-1000)//2:((len(x)-1000)//2)+1000])

    df['rev_comp_features'] = df['rev_comp_fixed_len_seq']. \
      swifter.apply(lambda row: np.array([mapper[item] for item in row], dtype=np.bool_).reshape(-1, 4).T)

    return df

df_train = get_df(train_csv)
df_val = get_df(val_csv)
df_test = get_df(test_csv)

print('df_train:', df_train.shape,
      'df_val:', df_val.shape,
      'df_test:', df_test.shape)

NUM_WORKERS = 0 if str(device)=='cpu' else 16

def create_data_loader(df, batch_size):
  # print('ok1')
  ds = SequenceDataset(
    # features=df.features.to_numpy(),
    features=np.stack(df['features'].values),
    rev_comp_features=np.stack(df['rev_comp_features'].values),
    targets=df.label.to_numpy(),
  )
  # print('ok2'); exit(0)
  return DataLoader(
    ds,
    batch_size=batch_size,
    #num_workers=16,
    num_workers=NUM_WORKERS,
    shuffle=True
  )


train_data_loader = create_data_loader(df_train, BATCH_SIZE)
# pos_train_data_loader = create_data_loader(df_train_pos, BATCH_SIZE//2)
# neg_train_data_loader = create_data_loader(df_train_neg, BATCH_SIZE//2)
val_data_loader = create_data_loader(df_val, BATCH_SIZE)
test_data_loader = create_data_loader(df_test, BATCH_SIZE)



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

    self.fc1 = nn.Linear(32*64, 128)
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

    x = x.reshape(-1, 32*64)
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
    output1 = self.forward_one(input1)#.reshape(-1)
    output2 = self.forward_one(input2)#.reshape(-1)
    # print('output1:', output1)
    # print('output2:', output2)
    cat_output = torch.cat((output1, output2), 1)
    # print('cat_output:', cat_output)
    out = torch.mean(cat_output, dim=1)
    # print(out); exit(0)

    return out


class_names=[0, 1]
OUT_CH=128
FC=32
DROPOUT=0
n_classes=2
model = FactorNet(device=device, )
model = model.to(device)

# EPOCHS = int(sys.argv[3]) #10
# optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
# total_steps = len(train_data_loader) * EPOCHS
# total_steps = len(pos_train_data_loader) * EPOCHS
scheduler = ''
# scheduler = get_linear_schedule_with_warmup(
#   optimizer,
#   num_warmup_steps=0,
#   num_training_steps=total_steps
# )
loss_fn = nn.CrossEntropyLoss().to(device)
# loss_fn = nn.BCELoss().to(device)

# optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
optimizer = torch.optim.Adam(model.parameters())

def train_epoch(
  model,
  data_loader,
  loss_fn,
  optimizer,
  device,
  scheduler,
  n_examples
):
  model = model.train()
  losses = []
  correct_predictions = 0
  for d in data_loader:
    # input_ids = d["input_ids"].to(device)
    # attention_mask = d["attention_mask"].to(device)
    features = d['review_text'].to(device)
    rev_comp_features = d['review_text_rev_comp'].to(device)
    targets = d["targets"].to(device)
    # print('pos:', len(targets[targets==1]))
    # print('neg:', len(targets[targets==0]))
    # print('input_ids:', input_ids.shape)
    # print('attention_mask:', attention_mask.shape)
    # print('targets:', targets.shape); #exit(0)
    # outputs = model(
    #   input_ids=input_ids,
    #   attention_mask=attention_mask
    # )
    outputs = model(features, rev_comp_features).reshape(-1)
    # print('outputs:', outputs.shape, targets.shape); exit(0)
    preds = (outputs >= 0.5).float()*1
    # print('preds:', preds); exit(0)
    # _, preds = torch.max(outputs, dim=1)
    # loss = loss_fn(outputs, targets)
    # use factornet loss function
    loss = F.binary_cross_entropy(outputs, targets.float())

    correct_predictions += torch.sum(preds == targets)
    # correct_predictions = loss #+= torch.sum(preds == targets)
    losses.append(loss.item())
    loss.backward()
    # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    # scheduler.step()
    optimizer.zero_grad()
  return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
  model = model.eval()
  losses = []
  correct_predictions = 0
  with torch.no_grad():
    for d in data_loader:
      # input_ids = d["input_ids"].to(device)
      # attention_mask = d["attention_mask"].to(device)
      features = d['review_text'].to(device)
      rev_comp_features = d['review_text_rev_comp'].to(device)
      targets = d["targets"].to(device)
      # outputs = model(
      #   input_ids=input_ids,
      #   attention_mask=attention_mask
      # )


      # outputs = model(features)
      # _, preds = torch.max(outputs, dim=1)
      # loss = loss_fn(outputs, targets)

      outputs = model(features, rev_comp_features).reshape(-1)
      preds = (outputs >= 0.5).float() * 1
      loss = F.binary_cross_entropy(outputs, targets.float())

      correct_predictions += torch.sum(preds == targets)

      losses.append(loss.item())
  return correct_predictions.double() / n_examples, np.mean(losses)


# %%time
history = defaultdict(list)
best_accuracy = 0
for epoch in range(EPOCHS):
  print(f'Epoch {epoch + 1}/{EPOCHS}')
  print('-' * 10)
  train_acc, train_loss = train_epoch(
    model,
    train_data_loader,
    loss_fn,
    optimizer,
    device,
    scheduler,
    len(df_train)
  )
  print(f'Train loss {train_loss} accuracy {train_acc}')
  val_acc, val_loss = eval_model(
    model,
    val_data_loader,
    loss_fn,
    device,
    len(df_val)
  )
  print(f'Val   loss {val_loss} accuracy {val_acc}')
  print()
  history['train_acc'].append(train_acc)
  history['train_loss'].append(train_loss)
  history['val_acc'].append(val_acc)
  history['val_loss'].append(val_loss)
  if val_acc > best_accuracy:
    # torch.save(model.state_dict(), 'best_model_state.bin')
    torch.save(model.state_dict(), model_pth)
    best_accuracy = val_acc

test_acc, _ = eval_model(
  model,
  test_data_loader,
  loss_fn,
  device,
  len(df_test)
)
print(test_acc.item())
print('best_val_acc:', best_accuracy)


