from google.colab import drive
drive.mount('/content/drive')
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

train_data = pd.read_csv('/content/drive/My Drive/cure_the_princess_train.csv')
val_data = pd.read_csv('/content/drive/My Drive/cure_the_princess_validation.csv')
test_data = pd.read_csv('/content/drive/My Drive/cure_the_princess_test.csv')

%%timeit


# Rastgele sayı üreteci ayarla
SEED = 180401027
torch.manual_seed(SEED)
from sklearn.preprocessing import MinMaxScaler,RobustScaler
skaler = MinMaxScaler()
# Verileri tensor'lara dönüştür
train_X = torch.tensor(train_data.values[:, :-1], dtype=torch.float32)
train_X = torch.tensor(skaler.fit_transform(train_X),dtype=torch.float32)
train_y = torch.tensor(train_data.values[:, -1], dtype=torch.float32)
val_X = torch.tensor(val_data.values[:, :-1], dtype=torch.float32)
val_X = torch.tensor(skaler.fit_transform(val_X),dtype=torch.float32)
val_y = torch.tensor(val_data.values[:, -1], dtype=torch.float32)
test_X = torch.tensor(test_data.values[:, :-1], dtype=torch.float32)
test_X = torch.tensor(skaler.fit_transform(test_X),dtype=torch.float32)
test_y = torch.tensor(test_data.values[:, -1], dtype=torch.float32)
train_y = train_y.reshape(-1, 1)
test_y = test_y.reshape(-1, 1)
val_y = val_y.reshape(-1, 1)

%load_ext autotime
# MLP modelini tanımla
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt



# Rastgele sayı üreteci ayarla
SEED = 180401027
torch.manual_seed(SEED)
from sklearn.preprocessing import MinMaxScaler,RobustScaler
skaler = MinMaxScaler()
# Verileri tensor'lara dönüştür
train_X = torch.tensor(train_data.values[:, :-1], dtype=torch.float32)
train_X = torch.tensor(skaler.fit_transform(train_X),dtype=torch.float32)
train_y = torch.tensor(train_data.values[:, -1], dtype=torch.float32)
val_X = torch.tensor(val_data.values[:, :-1], dtype=torch.float32)
val_X = torch.tensor(skaler.fit_transform(val_X),dtype=torch.float32)
val_y = torch.tensor(val_data.values[:, -1], dtype=torch.float32)
test_X = torch.tensor(test_data.values[:, :-1], dtype=torch.float32)
test_X = torch.tensor(skaler.fit_transform(test_X),dtype=torch.float32)
test_y = torch.tensor(test_data.values[:, -1], dtype=torch.float32)
train_y = train_y.reshape(-1, 1)
test_y = test_y.reshape(-1, 1)
val_y = val_y.reshape(-1, 1)
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out

# Modeli eğit
input_size = train_X.shape[1]
hidden_size1 = 100
hidden_size2 = 50
output_size = 1

model = MLP(input_size, hidden_size1, hidden_size2, output_size)

criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.02)
batch_size = 16
num_epochs = 500
patience = 5

# Eğitim ve doğrulama seti için kayıp değerlerini tutacak listeleri oluştur
train_loss_list = []
val_loss_list = []
best_val_loss=None
patience_counter=0
# Eğitim döngüsü
for epoch in range(num_epochs):
    # Batch'lerden eğitim kayıplarını tutacak bir liste oluştur
    epoch_train_loss = []
    for i in range(0, len(train_X), batch_size):
        inputs = train_X[i:i+batch_size]
        labels = train_y[i:i+batch_size].reshape(-1, 1)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        # Batch kaybını ekle
        epoch_train_loss.append(loss.item())

    # Eğitim kaybını epoch sonunda hesapla ve listeye ekle
    train_loss = np.mean(epoch_train_loss)
    train_loss_list.append(train_loss)

 # Doğrulama kaybını hesapla ve listeye ekle
    val_loss = 0.0
    """val_correct = 0.0
    val_total = 0.0"""
    #validation_count = 0.0
    with torch.no_grad():
        model.eval()
        for validation_X, validation_y in zip(val_X,val_y):
            outputs = model(validation_X)
            loss = criterion(outputs, validation_y)
            #validation_count += 1.0
            val_loss += loss.item()
            """, predicted = torch.max(outputs, 1) # burada validation accuracy hesabı var.
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()"""

    model.train()

    # calculate metrics
  
    val_loss /= val_X.shape[0]

    #val_acc = 100 * val_correct / val_total

    print("Epoch", epoch, "Training loss", train_loss,"Validation Loss :",val_loss)

    
    val_loss_list.append(val_loss)

    val_score = val_loss
    
    if best_val_loss is None:
        best_val_loss = val_score # hafızada patience boyu tutmaya başla
        torch.save(model.state_dict(), "checkpoint.pt")
    elif best_val_loss < val_score: # patience counter
        patience_counter += 1
        print("Earlystopping Patience Counter:",patience_counter)
        if patience_counter == patience:
            break
    else:
        best_val_loss = val_score
        torch.save(model.state_dict(), "checkpoint.pt") # to keep the best model
        patience_counter = 0
plt.plot(train_loss_list, label="Training loss")
plt.plot(val_loss_list, label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()


print('Finished Training')
model = MLP(input_size, hidden_size1, hidden_size2, output_size)
model.load_state_dict(torch.load('checkpoint.pt'))
model.eval()
predicts =[]
real_labels = list()
with torch.no_grad():
    for inputs,label in zip(test_X,test_y):
        outputs = model(inputs)
        predict = round(float(outputs.data))
        predicts.append(predict)


        real_labels.extend(label.tolist())
from sklearn.metrics import f1_score,accuracy_score,classification_report
print("Accuracy score of this model: {}".format(accuracy_score(real_labels,predicts)))
print(classification_report(real_labels,predicts))
