# A partir de la liste de mails, on va créer un vocabulaire
# Les mots du mails sont remplacés par leur index dans le vocabulaire
# Ensuites, on va créer une matrice d'embedding à partir du vocabulaire
# Cette matrice d'embedding sera utilisée comme poids de la première couche du réseau de neurones
# Elle sera donc entrainée en même temps que le reste du réseau et permettra au réseau de mieux comprendre les mots

# penser à enlever les caractères spéciaux, les \n (à remplacer par des espaces),...
import pdb
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer
from Enron_dataset.file_reader import File_reader
import numpy as np
import matplotlib.pyplot as plt

# 1) Create dataloader from enron dataset

fr = File_reader()
X_data, Y_label = fr.load_ham_and_spam(ham_paths = "default", spam_paths = "default", max = 50)

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

# max_len = max([len(tensor[0]) for tensor in X_tokenized_before_padding])
max_len = 512

for i in range(len(X_tokenized_before_padding)):
    if X_tokenized_before_padding[i].size(1) > max_len:
        # Truncate the sequence if its length is greater than max_len
        X_tokenized_before_padding[i] = torch.tensor(X_tokenized_before_padding[i][:,:max_len])
        X_tokenized_attention_masks_before_padding[i] = torch.tensor(X_tokenized_attention_masks_before_padding[i][:,:max_len])
    else:
        # Pad the sequence if its length is less than max_len
        X_tokenized_before_padding[i] = nn.ConstantPad1d((0, max_len - X_tokenized_before_padding[i].size(1)), 0)(X_tokenized_before_padding[i])
        X_tokenized_attention_masks_before_padding[i] = nn.ConstantPad1d((0, max_len - X_tokenized_attention_masks_before_padding[i].size(1)), 0)(X_tokenized_attention_masks_before_padding[i])

    
    # X_tokenized_before_padding[i] = nn.ConstantPad1d((0, max_len - len(X_tokenized_before_padding[i][0])),0)(X_tokenized_before_padding[i])
    # X_tokenized_attention_masks_before_padding[i] = nn.ConstantPad1d((0, max_len - len(X_tokenized_attention_masks_before_padding[i][0])),0)(X_tokenized_attention_masks_before_padding[i])


X_tokenized_after_padding = torch.cat(X_tokenized_before_padding, dim=0)
X_attention_masks_after_padding = torch.cat(X_tokenized_attention_masks_before_padding, dim=0).unsqueeze(2)
flattened_features = X_attention_masks_after_padding.clone().detach().to(dtype=torch.float32).mean(dim=1) #pooling


### 4) We now have X_tokenized_after_padding (Tokenized data with padding) and flattened_features (Attention masks pooled)
### We can now create a PyTorch dataset and data loaders

# Question : Here, do we need to standardize the data/featnures ?

X = X_tokenized_after_padding.clone().detach().to(dtype=torch.int64)
y = Y_label.clone().detach().to(dtype=torch.int64)


# Manual split into train and test sets
SPLIT_RATIO = 0.8
split_idx = int(SPLIT_RATIO * len(X))

X_train, attention_mask_train, y_train = X[:split_idx], flattened_features[:split_idx], y[:split_idx]
X_test, attention_mask_test, y_test = X[split_idx:], flattened_features[split_idx:], y[split_idx:]



# pdb.set_trace()

# Create a PyTorch dataset and data loaders
train_dataset = TensorDataset(X_train, attention_mask_train,  y_train)
test_dataset = TensorDataset(X_test, attention_mask_test, y_test)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# for batch in train_dataset:
#     print(batch)

# Model
class BERTMLPClassifier(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', mlp_input_dim=768, mlp_hidden_dim=256, num_classes=2):
        super(BERTMLPClassifier, self).__init__()

        # BERT model for embedding
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)

        # MLP for classification
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, mlp_hidden_dim), # input layer
            nn.ReLU(), # activation function
            nn.Dropout(0.1), # dropout added as regularizer
            nn.Linear(mlp_hidden_dim, num_classes) # classification layer
        )

    def forward(self, input_ids, attention_mask):
        # BERT embedding
        outputs = self.bert(input_ids, attention_mask=attention_mask) 
        pooled_output = outputs['pooler_output']
        pooled_output = self.dropout(pooled_output)

        # MLP classification
        logits = self.mlp(pooled_output)
        return logits


# training
model = BERTMLPClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10
loss_values = []
for epoch in range(num_epochs):
    index = 0
    print("epoch : ", epoch)
    for batch_x, attention_mask, batch_y in train_loader:
        optimizer.zero_grad() # 
        new_batch_x = torch.split(batch_x, 512,1)[0]
        new_attention_mask = torch.split(attention_mask, 512,1)[0]
        logits = model(new_batch_x, new_attention_mask)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
    loss_values.append(loss.item())
    print(f"Epoch {epoch+1}/{num_epochs} : loss = {loss.item():.4f}")

print(loss_values)
plt.figure(figsize=(10, 5))
plt.plot(loss_values, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()

plt.tight_layout()
plt.show()