# A partir de la liste de mails, on va créer un vocabulaire
# Les mots du mails sont remplacés par leur index dans le vocabulaire
# Ensuites, on va créer une matrice d'embedding à partir du vocabulaire
# Cette matrice d'embedding sera utilisée comme poids de la première couche du réseau de neurones
# Elle sera donc entrainée en même temps que le reste du réseau et permettra au réseau de mieux comprendre les mots

# penser à enlever les caractères spéciaux, les \n (à remplacer par des espaces),...

import torch
import torch.nn as nn
from transformers import BertModel
from transformers import AutoTokenizer
from Enron_dataset.file_reader import File_reader
from preprocess_text import preprocess_text
from sklearn.utils import shuffle
import gc
from memory_profiler import profile

# use gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 1) Create dataloader from enron dataset
bert_model_name = "bert-base-uncased"
fr = File_reader()

number_spam = 2000
X_data, Y_label = fr.load_ham_and_spam(ham_paths = "default", spam_paths = "default", max = number_spam)

X_data = [preprocess_text(mail) for mail in X_data]
### 2) From data (string) to Tokenized data and Attention masks tensors before padding

tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
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
        tokenized_batch = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
        tokenized_batches.append(tokenized_batch['input_ids'].to(device))

        # add attention mask 
        batch_attention_mask = tokenized_batch['attention_mask'].to(device)
        attention_masks_batches.append(batch_attention_mask)


    X_tokenized_before_padding.append(torch.cat(tokenized_batches, dim=1))
    X_tokenized_attention_masks_before_padding.append(torch.cat(attention_masks_batches, dim=1))

### 3) Add padding (to size of longest data)

# max_len = max([len(tensor[0]) for tensor in X_tokenized_before_padding])
max_len = 20480

for i in range(len(X_tokenized_before_padding)):
    if X_tokenized_before_padding[i].size(1) > max_len:
        # Truncate the sequence if its length is greater than max_len
        X_tokenized_before_padding[i] = X_tokenized_before_padding[i][:,:max_len].clone().detach()
        X_tokenized_attention_masks_before_padding[i] = X_tokenized_attention_masks_before_padding[i][:,:max_len].clone().detach()
    else:
        # Pad the sequence if its length is less than max_len
        X_tokenized_before_padding[i] = nn.ConstantPad1d((0, max_len - X_tokenized_before_padding[i].size(1)), 0)(X_tokenized_before_padding[i])
        X_tokenized_attention_masks_before_padding[i] = nn.ConstantPad1d((0, max_len - X_tokenized_attention_masks_before_padding[i].size(1)), 0)(X_tokenized_attention_masks_before_padding[i])

X_tokenized_after_padding = torch.cat(X_tokenized_before_padding, dim=0)
X_attention_masks_after_padding = torch.cat(X_tokenized_attention_masks_before_padding, dim=0).unsqueeze(2)
flattened_features = X_attention_masks_after_padding.clone().detach().to(dtype=torch.float32).mean(dim=1) #pooling


### 4) We now have X_tokenized_after_padding (Tokenized data with padding) and flattened_features (Attention masks pooled)
### We can now create a PyTorch dataset and data loaders

# Question : Here, do we need to standardize the data/features ?

X = X_tokenized_after_padding.clone().detach().to(dtype=torch.int64)
y = Y_label.clone().detach().to(dtype=torch.int64)

X, flattened_features, y = shuffle(X, flattened_features, y, random_state=42)

# Manual split into train and test sets
SPLIT_RATIO = 0.8
split_idx = int(SPLIT_RATIO * len(X))
number_train = int(number_spam * 2 * SPLIT_RATIO)
number_test = int(number_spam * 2 * (1 - SPLIT_RATIO))


X_train, attention_mask_train, y_train = X[:split_idx].to(device), flattened_features[:split_idx].to(device), y[:split_idx].to(device)
X_test, attention_mask_test, y_test = X[split_idx:].to(device), flattened_features[split_idx:].to(device), y[split_idx:].to(device)

# Compute the embeddings
bert_model = BertModel.from_pretrained(bert_model_name)
bert_model.to(device)
def extract_bert_embeddings(model, input_ids, attention_mask):
    outputs = model(input_ids, attention_mask)
    embeddings = outputs.last_hidden_state[:,0,:] # CLS token, representation of the whole sequence
    return embeddings

torch.save(y_train, "./embeddings/y_train.pt")
torch.save(y_test, "./embeddings/y_test.pt")


step = 10
column_size = 512


with torch.no_grad():
    for i_train in range(0,number_train,step): 
        total_train_embeddings = []
        for column in range(0,max_len,column_size):
            X_train_split = torch.narrow(X_train[i_train:(i_train+step)], 1, column, column_size) 
            train_embeddings = extract_bert_embeddings(bert_model, X_train_split, attention_mask_train[i_train:(i_train+step)]) 
            total_train_embeddings.append(train_embeddings)
        total_train_embeddings = torch.cat(total_train_embeddings, dim=1)
        torch.save(total_train_embeddings.clone().detach(), "./embeddings/train_embeddings"+str(i_train)+".pt")


    for i_test in range(0,number_test,step):
        total_test_embeddings = []
        for column in range(0,max_len,column_size):
            X_test_split = torch.narrow(X_test[i_test:(i_test+step)], 1, column, column_size)
            test_embeddings = extract_bert_embeddings(bert_model, X_test_split, attention_mask_test[i_test:(i_test+step)])
            total_test_embeddings.append(test_embeddings)
        total_test_embeddings = torch.cat(total_test_embeddings, dim=1)
        torch.save(total_test_embeddings.clone().detach(), "./embeddings/test_embeddings"+str(i_test)+".pt")
