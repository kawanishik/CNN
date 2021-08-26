import os
import sys
import torch
import cv2
import pandas as pd
import numpy as np
import scipy.misc as smp
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from PIL import Image, ImageDraw
import seaborn as sns
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
os.environ['KMP_DUPLICATE_LIB_OK']='True'

fname = "icml_face_data.csv"
df = pd.read_csv(fname)
#print(len(df))

# 使用する感情の数
emotions = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

# 学習に使用するデータを作成
def prepare_data(data):
    image_array = np.zeros(shape=(len(data), 1, 48, 48))
    image_label = np.array(list(map(int, data['emotion'])))

    for i,row in enumerate(data.index):
        image = np.fromstring(data.loc[row, ' pixels'], dtype=int, sep=' ')
        image = np.reshape(image, (48, 48))
        image_array[i] = image
    return image_array, image_label

# データを作成
train_image_array_tmp, train_image_label_tmp = prepare_data(df[df[' Usage']=='Training'])
val_image_array_tmp, val_image_label_tmp = prepare_data(df[df[' Usage']=='PrivateTest'])
test_image_array_tmp, test_image_label_tmp = prepare_data(df[df[' Usage']=='PublicTest'])
train_image_array = []
train_image_label = []
val_image_array = []
val_image_label = []
test_image_array = []
test_image_label = []
for (d1, d2) in zip(train_image_array_tmp, train_image_label_tmp):
  train_image_array.append(d1)
  train_image_label.append(d2)
for (d1, d2) in zip(val_image_array_tmp, val_image_label_tmp):
  val_image_array.append(d1)
  val_image_label.append(d2)
for (d1, d2) in zip(test_image_array_tmp, test_image_label_tmp):
  test_image_array.append(d1)
  test_image_label.append(d2)

# Data Augmentation
datagen = ImageDataGenerator(
           rotation_range=0,
           width_shift_range=0,
           height_shift_range=0,
           shear_range=0,
           zoom_range=0,
           horizontal_flip=True,
           vertical_flip=False)

d = datagen.flow(train_image_array_tmp, batch_size=1, shuffle=False)

for i in range(len(d)):
  train_image_array.append(d[i][0])
  label = train_image_label[i]
  train_image_label.append(label)
#print(len(train_image_array))
#print(len(train_image_label))

"""
datagen = ImageDataGenerator(
           rotation_range=0,
           width_shift_range=0,
           height_shift_range=0,
           shear_range=0,
           zoom_range=0,
           horizontal_flip=False,
           vertical_flip=True,
           )

d = datagen.flow(train_image_array_tmp, batch_size=1, shuffle=False)
#print(len(d))

for i in range(len(d)):
  train_image_array.append(d[i][0])
  label = train_image_label[i]
  train_image_label.append(label)
print(len(train_image_array))
print(len(train_image_label))
"""

# バッチ処理を適用するための関数
def train2batch(data, category, batch_size=32):
    data_batch = []
    cat_batch = []
    data, category = shuffle(data, category)
    for i in range(0, len(data), batch_size):
        data_batch.append(data[i:i+batch_size])
        cat_batch.append(category[i:i+batch_size])
    return data_batch, cat_batch

# データの作成
train_data, train_label = train2batch(train_image_array, train_image_label)
val_data, val_label = train2batch(val_image_array, val_image_label)
test_data, test_label = train2batch(test_image_array, test_image_label)
#train_data = list(map(lambda x: x / 255.0, train_data))

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 32, 5)
        self.fc1 = nn.Linear(32 * 9 * 9, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 7)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        #print(x.shape)
        x = self.pool(F.relu(self.conv2(x)))
        #print(x.shape)
        #x = x.view(-1, 32*9*9)
        x = x.view(x.shape[0], -1)
        #print(x.shape)
        x = F.relu(self.fc1(x))
        #print(x.shape)
        x = F.relu(self.fc2(x))
        #print(x.shape)
        x = self.fc3(x)
        #print(x.shape)
        return x

def evaluate(data, label):
    epoch_test_l = 0
    epoch_test_a = 0
    with torch.no_grad():
        for i in range(len(data)):
            inputs = torch.tensor(data[i]).to(device, dtype=torch.float)
            #inputs = inputs.unsqueeze(1)
            labels = torch.tensor(label[i]).to(device, dtype=torch.long)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            epoch_test_l += loss.item() / inputs.shape[0]
            test_acc = (outputs.argmax(dim=1) == labels).float().mean()
            epoch_test_a += test_acc / inputs.shape[0]
    return epoch_test_l, epoch_test_a

def evaluate_value(data, label):
    epoch_test_l = 0
    epoch_test_a = 0
    ans = 0
    cnt = 0
    with torch.no_grad():
        for i in range(len(data)):
            inputs = torch.tensor(data[i]).to(device, dtype=torch.float)
            #inputs = inputs.unsqueeze(1)
            labels = torch.tensor(label[i]).to(device, dtype=torch.long)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            epoch_test_l += loss.item() / inputs.shape[0]
            cnt += inputs.shape[0]
            test_acc = (outputs.argmax(dim=1) == labels).float().mean()
            #print(outputs.argmax(dim=1))
            #print(labels)
            #print((outputs.argmax(dim=1) == labels).sum())
            ans += (outputs.argmax(dim=1) == labels).sum()
            epoch_test_a += test_acc / inputs.shape[0]
    return epoch_test_l, epoch_test_a, ans, cnt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("device : ", device)
model = CNN().to(device)
#print(model)

criterion = nn.CrossEntropyLoss()
#optimizer = optim.Adam(model.parameters(), lr=0.005)
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.005)
#model_path = "model_expand.pth"
model_path = "model_expand.pth"

epochs = 20
train_loss = []
test_loss = []
min_loss = 100.0
for epoch in range(epochs):
    epoch_train_loss = 0
    epoch_train_acc = 0
    epoch_test_loss = 0
    epoch_test_acc = 0

    model.train()
    for i in range(len(train_data)):
        inputs = torch.tensor(train_data[i], requires_grad=True).to(device, dtype=torch.float)
        #inputs = inputs.unsqueeze(1)
        labels = torch.tensor(train_label[i]).to(device, dtype=torch.long)
        #labels = torch.from_numpy(train_label[i]).to(device).clone()
        #print(labels)
        #print('inputs :', inputs.shape)
        #print('lables :', labels.shape)
        #print(inputs.shape[0])
        #print(inputs[0][0])

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        #print(loss)
        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.item() / inputs.shape[0]
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        epoch_train_acc += acc / inputs.shape[0]
        #print("acc :", acc, "epoch_train_acc :", epoch_train_acc)
    #if(epoch + 1) % 10 == 0:
    epoch_test_loss, epoch_test_acc = evaluate(val_data, val_label)
    print(f'Epoch {epoch+1} : train acc. {epoch_train_acc:.2f} train loss {epoch_train_loss:.2f}')
    print(f'Epoch {epoch+1} : test acc. {epoch_test_acc:.2f} test loss {epoch_test_loss:.2f}')
    if min_loss > epoch_test_loss:
      min_loss = epoch_test_loss
      torch.save(model.state_dict(), model_path)
    train_loss.append(epoch_train_loss)
    test_loss.append(epoch_test_loss)

model.load_state_dict(torch.load(model_path))
epoch_test_loss, epoch_test_acc, ans, cnt = evaluate_value(test_data, test_label)
print(f'test acc. {epoch_test_acc:.2f} test loss {epoch_test_loss:.2f}')
print("正解率 :", ans / cnt)

