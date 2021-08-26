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

test_image_array_tmp, test_image_label_tmp = prepare_data(df[df[' Usage']=='PublicTest'])
test_image_array = []
test_image_label = []
for (d1, d2) in zip(test_image_array_tmp, test_image_label_tmp):
  test_image_array.append(d1)
  test_image_label.append(d2)

def train2batch(data, category, batch_size=32):
    data_batch = []
    cat_batch = []
    data, category = shuffle(data, category)
    for i in range(0, len(data), batch_size):
        data_batch.append(data[i:i+batch_size])
        cat_batch.append(category[i:i+batch_size])
    return data_batch, cat_batch

test_data, test_label = train2batch(test_image_array, test_image_label)

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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("device : ", device)
model = CNN().to(device)
model_path = "model_expand.pth"
model.load_state_dict(torch.load(model_path))

class pycolor:
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    PURPLE = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    RETURN = '\033[07m' #反転
    ACCENT = '\033[01m' #強調
    FLASH = '\033[05m' #点滅
    RED_FLASH = '\033[05;41m' #赤背景+点滅
    END = '\033[0m'

def sigmoid(a):
  return 1 / (1 + np.exp(-a))

def softmax(a):
  #print(np.sum(np.exp(a)))
  return np.exp(a) / np.sum(np.exp(a))

def result(id):
  with torch.no_grad():
    input = torch.tensor(test_image_array[id]).to(device, dtype=torch.float)
    input = input.unsqueeze(0)
    outputs = model(input)
    ans = outputs.argmax(dim=1).item()
    if ans == test_image_label[id]:
      print(pycolor.RED+"Success"+pycolor.END)
    else:
      print(pycolor.RED+"Failed :"+pycolor.END, pycolor.RED+emotions[test_image_label[id]]+pycolor.END)
    #print(ans, test_image_label[id])
    x = outputs.to('cpu').detach().numpy().copy()
    sum = softmax(x)
    index_tmp = np.argpartition(sum, -4)[-4:]
    index = []
    for i in range(len(index_tmp[0])):
      index.insert(0, index_tmp[0][i])
    #print(sum)
    #print(index)
  if sum[0][index[0]] > 0.7:
    print(pycolor.RED+"Very"+pycolor.END, pycolor.RED+emotions[index[i]]+pycolor.END)
  elif sum[0][index[0]] > 0.5:
    print(pycolor.RED+emotions[index[0]]+pycolor.END)
  else:
    print(pycolor.RED+"A little"+pycolor.END, pycolor.RED+emotions[index[0]]+pycolor.END)
  for i in range(4):
    print(emotions[index[i]], ":", sum[0][index[i]])
  #print(sum)
  data = test_image_array[id][0]
  plt.subplot(1, 2, 1), plt.imshow(np.uint8(data))
  plt.gray()
  plt.show()

result(13)