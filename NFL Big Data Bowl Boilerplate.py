# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), '../../../../../var/folders/r8/scstr3q94_7db3_dqvs8l25h0000gn/T'))
	print(os.getcwd())
except:
	pass

# %%
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
import gc
from tqdm import tqdm_notebook as tqdm
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


# %%
train_path = './train.csv'
train = pd.read_csv(train_path)
print(train.shape)


# %%
use_cols = [
    'GameId', 
    'PlayId', 
    'Team',
    'Yards',
    'X',
    'Y',
    'PossessionTeam',
    'HomeTeamAbbr',
    'VisitorTeamAbbr',
    'Position',
]
train = train[use_cols]
train.head()


# %%
"""
Find an offense team.
ref: https://www.kaggle.com/c/nfl-big-data-bowl-2020/discussion/112314#latest-648026
"""
def fe_is_offence(row):
    if row["Team"] == "home":
        if row["HomeTeamAbbr"] == row["PossessionTeam"]:
            return 1
        else:
            return 0
    elif row["Team"] == "away":
        if row["VisitorTeamAbbr"] == row["PossessionTeam"]:
            return 1
        else:
            return 0

def fe_is_offence_from_position(row, off_team):
    if row["Team"] == off_team:
        return 1
    else:
        return 0
        
# def run_fe_is_offence(df):
#     df['is_offence'] = df.apply(lambda row: fe_is_offence(row), axis=1)
    
#     if (df['is_offence'].values == 0).all():
#         off_team = df[df['Position']=='QB']['Team'].values[0]
#         df['is_offence'] = df.apply(lambda row: fe_is_offence_from_position(row, off_team), axis=1)

"""
bugfix
"""
def run_fe_is_offence(df):
    df['is_offence'] = df.apply(lambda row: fe_is_offence(row), axis=1)
    
    check_is_offence = df.groupby('PlayId')['is_offence'].nunique()
    is_offence_not_found_idx = check_is_offence[check_is_offence!=2].index
    not_found_df = df[df['PlayId'].isin(is_offence_not_found_idx)]
    found_df = df[~df['PlayId'].isin(is_offence_not_found_idx)]
#     print('is_offence found: {}'.format(len(found_df)))
#     print('is_offence not found: {}'.format(len(not_found_df)))

    for u_play_id in not_found_df['PlayId'].unique():
        tmp_df = not_found_df[not_found_df['PlayId']==u_play_id]
        pos_list = [pos for pos in tmp_df['Position'].unique() if pos in ['QB', 'RB', 'WR', 'TE']]
        
        if len(pos_list) > 0:
            off_team = tmp_df[tmp_df['Position']==pos_list[0]]['Team'].values[0]
#         else:
#             print('Offence position not found')
#             import pdb;pdb.set_trace()

        target_idx = not_found_df.query('PlayId==@u_play_id and Team==@off_team').index
        not_found_df.loc[target_idx, 'is_offence'] = 1
    
    df = pd.concat([found_df, not_found_df], sort=False)
#     print('done df: {}'.format(df.shape))
    return df


# %%
def run_group_fe(df, group_key, aggs):
    
    group_df = df.groupby(group_key).agg(aggs)

    new_cols = [col[0]+'_'+col[1] for col in group_df.columns]
    group_df.columns = new_cols
    group_df.reset_index(inplace=True)
        
    return group_df

def adjust_group_df(group_df, is_train):
    offence_df = group_df[group_df['is_offence']==1]
    deffence_df = group_df[group_df['is_offence']==0]

    del group_df['is_offence']
    del offence_df['is_offence']
    del deffence_df['is_offence']
    
    if is_train:
        off_cols = ['off_{}'.format(col) if col not in ['GameId', 'PlayId', 'Yards'] else col for col in group_df.columns]
        deff_cols = ['deff_{}'.format(col) if col not in ['GameId', 'PlayId', 'Yards'] else col for col in group_df.columns]
    else:
        off_cols = ['off_{}'.format(col) if col not in ['GameId', 'PlayId'] else col for col in group_df.columns]
        deff_cols = ['deff_{}'.format(col) if col not in ['GameId', 'PlayId'] else col for col in group_df.columns]
        
    offence_df.columns = off_cols
    deffence_df.columns = deff_cols
    if is_train: del deffence_df['Yards']
    
    adjusted_group_df = pd.merge(offence_df, deffence_df, on=['GameId', 'PlayId'])
    
    return adjusted_group_df


# %%
train = run_fe_is_offence(train)
train.head()


# %%
train_group_key = ['GameId', 'PlayId', 'is_offence', 'Yards']
aggs = {
    'X': ['mean', 'max', 'min', 'median'],
    'Y': ['mean', 'max', 'min', 'median'],
}
is_train = True
group_df = run_group_fe(train, train_group_key, aggs)
adjusted_group_df = adjust_group_df(group_df, is_train)
print(adjusted_group_df.shape)
adjusted_group_df.head()


# %%
# random.seed(1234)
# os.environ['PYTHONHASHSEED'] = str(1234)
# np.random.seed(1234)
# torch.manual_seed(1234)
# torch.cuda.manual_seed(1234)
# torch.backends.cudnn.deterministic = True

# %% [markdown]
# ## Full K-Fold

# %%
class NFL_NN(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.fc1 = nn.Linear(in_features, 216)
        self.bn1 = nn.BatchNorm1d(216)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(216, 512)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(512, 216)
        self.relu3 = nn.ReLU()
        self.dout3 = nn.Dropout(0.2)
        self.out = nn.Linear(216, out_features)
        self.out_act = nn.Sigmoid()
        
    def forward(self, input_):
        a1 = self.fc1(input_)
        bn1 = self.bn1(a1)
        h1 = self.relu1(bn1)
        a2 = self.fc2(h1)
        h2 = self.relu2(a2)
        a3 = self.fc3(h2)
        h3 = self.relu3(a3)
        dout3 = self.dout3(h3)
        a5 = self.out(dout3)
        y = self.out_act(a5)
        return a5

for param in model.parameters():
    print(param.shape)


# %%
epoch = 10
batch_size = 1012


# %%
oof_crps_list = []
fold = GroupKFold(n_splits=5)


y = np.zeros(shape=(adjusted_group_df.shape[0], 199))
for i, yard in enumerate(adjusted_group_df['Yards'].values):
#     print(i, yard)
    y[i, yard+99:] = np.ones(shape=(1, 100-yard))

oof_preds = np.ones((len(adjusted_group_df), train_y.shape[1]))

feats = [
        "off_X_mean","off_X_max","off_X_min","off_X_median","off_Y_mean","off_Y_max","off_Y_min","off_Y_median",
        "deff_X_mean","deff_X_max","deff_X_min","deff_X_median","deff_Y_mean","deff_Y_max","deff_Y_min","deff_Y_median",
    ]

print('use feats: {}'.format(len(feats)))


# %%
for n_fold, (train_idx, valid_idx) in enumerate(fold.split(adjusted_group_df, y, groups=adjusted_group_df['GameId'])):
        print('Fold: {}'.format(n_fold+1))
        
        train_x, train_y = adjusted_group_df[feats].iloc[train_idx].values, y[train_idx]
        valid_x, valid_y = adjusted_group_df[feats].iloc[valid_idx].values, y[valid_idx] 

        train_x = torch.from_numpy(train_x)
        train_y = torch.from_numpy(train_y)
        valid_x = torch.from_numpy(valid_x)
        valid_y = torch.from_numpy(valid_y)

        train_dataset = TensorDataset(train_x, train_y)
        valid_dataset = TensorDataset(valid_x, valid_y)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        
        print('train: {}, valid: {}'.format(len(train_dataset), len(valid_dataset)))
        
        in_features = adjusted_group_df[feats].shape[1]
        out_features = y.shape[1]
        
        model = NFL_NN(in_features, out_features)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        
        for idx in range(epoch):
            print('Training epoch {}'.format(idx+1))
            train_batch_loss_sum = 0
            
            for param in model.parameters():
                param.requires_grad = True
                
            model.train()
            for x_batch, y_batch in tqdm(train_loader):
                y_pred = model(x_batch.float())
                loss = torch.sqrt(criterion(y_pred.float(), y_batch.view((len(y_batch), out_features)).float()))
                train_batch_loss_sum += loss.item()
                
                del x_batch
                del y_batch

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                torch.cuda.empty_cache()
                gc.collect()
                
            train_epoch_loss = train_batch_loss_sum / len(train_loader)
            
            model.eval()
            preds = np.zeros((len(valid_dataset), out_features))
            with torch.no_grad():
                for i, eval_x_batch in enumerate(valid_loader):
                    eval_values = eval_x_batch[0].float()
                    pred = model(eval_values)
                    preds[i * batch_size:(i + 1) * batch_size] = pred
                    
            valid_y_pred = preds
            valid_crps = np.sum(np.power(valid_y_pred - valid_dataset[:][1].data.cpu().numpy(), 2))/(199*len(valid_dataset))
            oof_preds[valid_idx] = valid_y_pred
            
            print('Train Epoch Loss: {:.5f}, Valid CRPS: {:.5f}'.format(train_epoch_loss, valid_crps))
            
        del model, criterion, optimizer
        gc.collect()

print('DONE OOF ALL CRPS: {:.5f}'.format(np.sum(np.power(oof_preds - y, 2))/(199*len(oof_preds))))
    

# %% [markdown]
# ## One Batch

# %%
del model, criterion, optimizer


# %%
xb_train, yb_train = next(iter(train_loader))
xb_eval, yb_eval = next(iter(valid_loader))


# %%
xb_train, yb_train = next(iter(train_loader))
xb_eval, yb_eval = next(iter(valid_loader))

model = NFL_NN(in_features, out_features)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

for param in model.parameters():
    param.requires_grad = True

model.train()
y_pred = model(xb_train.float())

loss = torch.sqrt(criterion(y_pred.float(), yb_train.view((len(yb_train), out_features)).float()))
train_batch_loss_sum += loss.item()

del xb_train
del yb_train

optimizer.zero_grad()
loss.backward()
optimizer.step()

torch.cuda.empty_cache()
gc.collect()

train_epoch_loss = train_batch_loss_sum / len(train_loader)

model.eval()
preds = np.zeros((len(valid_dataset), out_features))
with torch.no_grad():
    eval_values = xb_eval.float()
    pred = model(eval_values)
    preds[0 * batch_size:(0 + 1) * batch_size] = pred
        
valid_y_pred = preds
valid_crps = np.sum(np.power(valid_y_pred - valid_dataset[:][1].data.cpu().numpy(), 2))/(199*len(valid_dataset))

print('Train Epoch Loss: {:.5f}, Valid CRPS: {:.5f}'.format(train_epoch_loss, valid_crps))


# %%
Train Epoch Loss: 0.23357, Valid CRPS: 1.19845
Train Epoch Loss: 0.27025, Valid CRPS: 0.82464
Train Epoch Loss: 0.30727, Valid CRPS: 0.60142
Train Epoch Loss: 0.34372, Valid CRPS: 0.86409
Train Epoch Loss: 0.38006, Valid CRPS: 0.96284

# %% [markdown]
# ## All Batches

# %%
train_batch_loss_sum = 0
            
for param in model.parameters():
    param.requires_grad = True

model.train()
for x_batch, y_batch in tqdm(train_loader):
    y_pred = model(x_batch.float())
    loss = torch.sqrt(criterion(y_pred.float(), y_batch.view((len(y_batch), out_features)).float()))
    train_batch_loss_sum += loss.item()

    del x_batch
    del y_batch

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    torch.cuda.empty_cache()
    gc.collect()

train_epoch_loss = train_batch_loss_sum / len(train_loader)

model.eval()
preds = np.zeros((len(valid_dataset), out_features))
with torch.no_grad():
    for i, eval_x_batch in enumerate(valid_loader):
        eval_values = eval_x_batch[0].float()
        pred = model(eval_values)
        preds[i * batch_size:(i + 1) * batch_size] = pred

valid_y_pred = preds
valid_crps = np.sum(np.power(valid_y_pred - valid_dataset[:][1].data.cpu().numpy(), 2))/(199*len(valid_dataset))
oof_preds[valid_idx] = valid_y_pred

print('Train Epoch Loss: {:.5f}, Valid CRPS: {:.5f}'.format(train_epoch_loss, valid_crps))
            


# %%
Train Epoch Loss: 0.38006, Valid CRPS: 0.96284

# %% [markdown]
# ## Two Epochs

# %%
for idx in range(2):
    print('Training epoch {}'.format(idx+1))
    train_batch_loss_sum = 0

    for param in model.parameters():
        param.requires_grad = True

    model.train()
    for x_batch, y_batch in tqdm(train_loader):
        y_pred = model(x_batch.float())
        loss = torch.sqrt(criterion(y_pred.float(), y_batch.view((len(y_batch), out_features)).float()))
        train_batch_loss_sum += loss.item()

        del x_batch
        del y_batch

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.empty_cache()
        gc.collect()

    train_epoch_loss = train_batch_loss_sum / len(train_loader)

    model.eval()
    preds = np.zeros((len(valid_dataset), out_features))
    with torch.no_grad():
        for i, eval_x_batch in enumerate(valid_loader):
            eval_values = eval_x_batch[0].float()
            pred = model(eval_values)
            preds[i * batch_size:(i + 1) * batch_size] = pred

    valid_y_pred = preds
    valid_crps = np.sum(np.power(valid_y_pred - valid_dataset[:][1].data.cpu().numpy(), 2))/(199*len(valid_dataset))
    oof_preds[valid_idx] = valid_y_pred

    print('Train Epoch Loss: {:.5f}, Valid CRPS: {:.5f}'.format(train_epoch_loss, valid_crps))


# %%



# %%



# %%



# %%



# %%



# %%



# %%



# %%



# %%



# %%



# %%



# %%



# %%



# %%



# %%



# %%


