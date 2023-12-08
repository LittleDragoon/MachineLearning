# A partir de la liste de mails, on va créer un vocabulaire
# Les mots du mails sont remplacés par leur index dans le vocabulaire
# Ensuites, on va créer une matrice d'embedding à partir du vocabulaire
# Cette matrice d'embedding sera utilisée comme poids de la première couche du réseau de neurones
# Elle sera donc entrainée en même temps que le reste du réseau et permettra au réseau de mieux comprendre les mots

# penser à enlever les caractères spéciaux, les \n (à remplacer par des espaces),...

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer
from Enron_dataset.file_reader import File_reader

# 1) Create dataloader from enron dataset

fr = File_reader()
X_data, Y_label = fr.load_ham_and_spam(ham_paths = "default", spam_paths = "default", max = 3000)


### 2) From data (string) to Tokenized data and Attention masks tensors before padding

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
data_batch_size = 512 
X_tokenized_before_padding = []
X_tokenized_attention_masks_before_padding = []
#Tokenize data using BertTokenizer with batches (Bert has a limit of 512 tokens)
for i in range(len(X_data)):  
    mail_i = X_data[i]
    tokenized_batches = []
    attention_masks_batches = []

    for batch_number in range(0, len(mail_i), data_batch_size):
        # tokenize data
        batch_texts = mail_i[batch_number:batch_number+data_batch_size]
        tokenized_batch = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
        tokenized_batches.append(tokenized_batch['input_ids'])

        # add attention mask 
        batch_attention_mask = tokenized_batch['attention_mask']
        attention_masks_batches.append(batch_attention_mask)


    X_tokenized_before_padding.append(torch.cat(tokenized_batches, dim=1))
    X_tokenized_attention_masks_before_padding.append(torch.cat(attention_masks_batches, dim=1))


### 3) Add padding (to size of longest data)

max_len = max([len(tensor[0]) for tensor in X_tokenized_before_padding])

for i in range(len(X_tokenized_before_padding)):
    X_tokenized_before_padding[i] = nn.ConstantPad1d((0, max_len - len(X_tokenized_before_padding[i][0])),0)(X_tokenized_before_padding[i])
    X_tokenized_attention_masks_before_padding[i] = nn.ConstantPad1d((0, max_len - len(X_tokenized_attention_masks_before_padding[i][0])),0)(X_tokenized_attention_masks_before_padding[i])


X_tokenized_after_padding = torch.cat(X_tokenized_before_padding, dim=0)
X_attention_masks_after_padding = torch.cat(X_tokenized_attention_masks_before_padding, dim=0).unsqueeze(2)
flattened_features = X_attention_masks_after_padding.clone().detach().to(dtype=torch.float32).mean(dim=1) #pooling

### 4) We now have X_tokenized_after_padding (Tokenized data with padding) and flattened_features (Attention masks pooled)
### We can now create a PyTorch dataset and data loaders

# Question : Here, do we need to standardize the data/features ?

X = X_tokenized_after_padding.clone().detach().to(dtype=torch.float32)
y = Y_label.clone().detach().to(dtype=torch.float32)


# Manual split into train and test sets
SPLIT_RATIO = 0.8
split_idx = int(SPLIT_RATIO * len(X))

X_train, y_train = X[:split_idx], y[:split_idx]
X_test, y_test = X[split_idx:], y[split_idx:]

# Create a PyTorch dataset and data loaders
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

for batch in train_dataset:
    print(batch)
