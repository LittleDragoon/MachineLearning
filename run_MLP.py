import torch
import os
import glob
import pdb
import torch
import torch.nn as nn
from transformers import AlbertModel
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Specify the path to the saved model file

def load_files (PATH, prefix):
    matching_files = glob.glob(os.path.join(PATH, f"{prefix}*"))
    total_embeddings = []
    for file in matching_files:
        total_embedding = torch.load(file)
        total_embeddings.append(total_embedding)
    
    total_embeddings = torch.cat(total_embeddings, dim=0)
    return total_embeddings


train_embeddings = load_files("./embeddings", "train")
test_embeddings = load_files("./embeddings", "test")
y_test = torch.load("./embeddings/y_test.pt")
y_train = torch.load("./embeddings/y_train.pt")


train_dataset = TensorDataset(train_embeddings, y_train)
test_dataset = TensorDataset(test_embeddings, y_test)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

bert_model_name = "albert-base-v2"
bert_model = AlbertModel.from_pretrained(bert_model_name)
input_size = bert_model.config.hidden_size
hidden_size = 100
num_classes = 2
model = MLP(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
num_epochs = 100
loss_values = []
for epoch in range(num_epochs):
    index = 0
    print("epoch : ", epoch)
    for embeddings, y_labels in train_loader:
        optimizer.zero_grad() # 

        logits = model(embeddings)
        loss = criterion(logits, y_labels)
        loss.backward()
        optimizer.step()
    loss_values.append(loss.item())
    print(f"Epoch {epoch+1}/{num_epochs} : loss = {loss.item():.4f}")
    # compute accuracy on test set
    with torch.no_grad():
        correct = 0
        total = 0
        for embeddings, y_labels in test_loader:
            logits = model(embeddings)
            _, predicted = torch.max(logits.data, 1)
            total += y_labels.size(0)
            correct += (predicted == y_labels).sum().item()
        print(f"Accuracy on test set: {100 * correct / total}%")

print(loss_values)
plt.figure(figsize=(10, 5))
plt.plot(loss_values, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()

plt.tight_layout()
plt.show()
