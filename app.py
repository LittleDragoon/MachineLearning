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

# Create dataloader from enron dataset

fr = File_reader()
X_data, Y_label = fr.load_ham_and_spam(ham_paths = "default", spam_paths = "default", max = 3000)


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

data_batch_size = 512 
X_data_attention_masks = []
#Tokenize data using BertTokenizer (which has a limit of 512 tokens)
for i in range(len(X_data)):  
    mail = X_data[i]
    tokenized_batches = []
    attention_masks_batches = []
    for batch_number in range(0, len(mail), data_batch_size):
        batch_texts = mail[batch_number:batch_number+data_batch_size]
        batch_X_data = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
        tokenized_batches.append(batch_X_data['input_ids'])

        # add attention mask 
        batch_attention_mask = batch_X_data['attention_mask']
        attention_masks_batches.append(batch_attention_mask)


    X_data[i] = torch.cat(tokenized_batches, dim=1)
    X_data_attention_masks.append(torch.cat(attention_masks_batches, dim=1))



# truncate data to size of longest data
max_len = max([len(tensor[0]) for tensor in X_data])


for i in range(len(X_data)):
    X_data[i] = nn.ConstantPad1d((0, max_len - len(X_data[i][0])),0)(X_data[i])
    X_data_attention_masks[i] = nn.ConstantPad1d((0, max_len - len(X_data_attention_masks[i][0])),0)(X_data_attention_masks[i])


X_data = torch.cat(X_data, dim=0)
X_data_attention_masks = torch.cat(X_data_attention_masks, dim=0).unsqueeze(2)
flattened_features = X_data_attention_masks.clone().detach().to(dtype=torch.float32).mean(dim=1) #pooling

# Question : Do we need to standardize the data/features ?

# Convert data to PyTorch tensors
X = X_data.clone().detach().to(dtype=torch.float32)
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
