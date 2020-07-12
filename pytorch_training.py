
#%%

import os, time
import pandas as pd
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib

# %%
SEED = 42
MAX_LENGTH = 1024

DATA_PATH =  "./data/"
SST_TRAIN = "train.csv"
SST_TEST  = "test.csv"

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

# %%

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(SEED)

# %%

import logging
logging.basicConfig(level=logging.ERROR)

# %%
df_train = pd.read_csv(os.path.join(DATA_PATH, SST_TRAIN))
df_train.head()


# %%
df_test  = pd.read_csv(os.path.join(DATA_PATH, SST_TEST))
df_test.head()


# %%

from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig

class TitleDataset(Dataset):

    def __init__(self, df, maxlen):
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
        self.df = df
        self.maxlen = maxlen

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        abstract = self.df.loc[index, 'abstract']
        title = self.df.loc[index, 'title']

        abstract_enc = self.tokenizer.encode_plus(
            abstract,
            return_tensors='pt',
            # add_special_tokens=True,
            max_length=self.maxlen,
            # pad_to_max_length=True
        )

        title_enc = self.tokenizer.encode_plus(
            title,
            return_tensors='pt',
            # add_special_tokens=True,
            max_length=self.maxlen,
            # pad_to_max_length=True
        )

        return torch.tensor(abstract_enc['input_ids']), \
            torch.tensor(title_enc['input_ids'])

#%%

# def evaluate(net, dataloader):
#     net.eval()

#     mean_acc, mean_auc, mean_loss = 0, 0, 0
#     count = 0

#     with torch.no_grad():
#         for abs_ids, tit_ids in dataloader:
#             abs_ids, tit_ids = abs_ids.cuda(), tit_ids.cuda()
#             loss, scores = net(abs_ids, decoder_input_ids=tit_ids) # TODO
#             mean_loss += loss
#             count += 1

#     return mean_acc / count, mean_auc / count, mean_loss / count


# def train(net, opti, train_loader, max_eps, print_every):
#     net.train()

#     for ep in range(max_eps):
#         for it, (abs_ids, tit_ids) in enumerate(train_loader):
#             opti.zero_grad()
            
#             abs_ids, tit_ids = abs_ids.cuda(), tit_ids.cuda()
#             loss = net(abs_ids, decoder_input_ids=tit_ids)[0]
#             loss = loss.mean()
#             loss.backward()
#             opti.step()

#             if (it + 1) % print_every == 0:
#                 #acc = get_accuracy_from_logits(logits, labels)
#                 # auc = get_auc_from_logits(logits, labels)
#                 print(f"Iteration {it+1} of epoch {ep+1} completed. Loss : {loss.item()} AUC : {auc}")
#                 # if (it + 1) % (print_every * 5) == 0:
#                 #     _, val_auc, val_loss = evaluate(net, criterion, val_loader, gpu)
#                 #     print(f"Preliminary validation, AUC : {val_auc}, Loss : {val_loss}")

#         # val_acc, val_auc, val_loss = evaluate(net, criterion, val_loader, gpu)
#         # print(f"Epoch {ep+1} complete! Validation AUC : {val_auc}, Validation Loss : {val_loss}")
#         # if val_auc > best_auc:
#         #     print("Best validation auc improved from {} to {}, saving model...".format(best_auc, val_auc))
#         #     best_auc = val_auc
#         #     torch.save(net.state_dict(), 'bert_ep{}_auc{:.2f}_freeze.dat'.format(ep, val_auc))



# %%

TRAIN_BATCH_SIZE = 1 # TODO Tune
EVAL_BATCH_SIZE = 1

train_set = TitleDataset(df_train, maxlen=MAX_LENGTH)
test_set = TitleDataset(df_test, maxlen=MAX_LENGTH)

train_loader = DataLoader(train_set, batch_size=TRAIN_BATCH_SIZE, num_workers=5)
test_loader = DataLoader(test_set, batch_size=EVAL_BATCH_SIZE, num_workers=5)


# %%

net = BartForConditionalGeneration \
    .from_pretrained('facebook/bart-large-cnn') \
    .to(torch_device)

# criterion = nn.BCEWithLogitsLoss()
opti = optim.Adam(net.parameters())
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

# %%

max_eps = 1
print_every = 100

net.train()

for ep in range(max_eps):
    for it, (abs_ids, tit_ids) in enumerate(train_loader):
        opti.zero_grad()
        
        abs_ids = torch.squeeze(abs_ids, dim=1).to(torch_device)
        tit_ids = torch.squeeze(tit_ids, dim=1).to(torch_device)
        # print(abs_ids.shape, tit_ids.shape)
        # abs_ids, tit_ids = abs_ids.to(torch_device), tit_ids.to(torch_device)

        # TEST That shapes are ok
        # summary_ids = net.generate(abs_ids, num_beams=4, max_length=5, early_stopping=True)
        # print([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids])

        loss = net(abs_ids, decoder_input_ids=tit_ids)[0]
        loss = loss.mean()
        loss.backward()
        opti.step()

        break

        if (it + 1) % print_every == 0:
            print(f"Iteration {it+1} of epoch {ep+1} completed. Loss : {loss.item()} AUC : {auc}")


# %%


from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig

# see ``examples/summarization/bart/run_eval.py`` for a longer example
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

ARTICLE_TO_SUMMARIZE = "My friends are cool but they eat too many carbs."
inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors='pt')

# Generate Summary
summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=5, early_stopping=True)
print([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids])


# %%

v = model(inputs['input_ids'], decoder_input_ids=inputs['input_ids'])
for i in v:
    print("NEW")

# %%

print(inputs['input_ids'])

# %%
