import torch
import os
import glob
import pdb
import torch
import torch.nn as nn
from transformers import AlbertModel
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# use gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

batch_size = 500
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, hidden_size_3, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size_1) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)  
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size_2, hidden_size_3)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_size_3, num_classes)

    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        return out

input_size = train_embeddings.shape[1]
hidden_size = 500
hidden_size_2 = 100
hidden_size_3 = 20
num_classes = 2
model = MLP(input_size=input_size, hidden_size_1=hidden_size, hidden_size_2=hidden_size_2, hidden_size_3=hidden_size_3, num_classes=num_classes)
model.to(device)
# def init_weights(m):
#     if isinstance(m, nn.Linear):
#         nn.init.kaiming_normal_(m.weight.data) 

# model.apply(init_weights)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
num_epochs = 500
loss_values = []
accuracy_values = []
for epoch in range(num_epochs):
    index = 0
    print("epoch : ", epoch)
    for embeddings, y_labels in train_loader:
        optimizer.zero_grad() # 
        logits = model(embeddings)
        loss = criterion(logits, y_labels)
        loss.backward()
        optimizer.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
        accuracy_values.append(100 * correct / total)


plt.figure(figsize=(10, 5))
plt.plot(loss_values, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()

plt.tight_layout()
plt.show()

# display accuracy
plt.figure(figsize=(10, 5))
plt.plot(accuracy_values, label='Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')
plt.legend()

plt.tight_layout()
plt.show()