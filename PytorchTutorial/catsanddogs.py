import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from torchvision import transforms
from PIL import Image
from torch.autograd import Variable
from os import listdir


normalize = transforms.Normalize(
    mean = [0.485, 0.456, 0.406],
    std= [0.229, 0.224, 0.225]
    )

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    normalize
    ])


        

class Netz(nn.Module):
    def __init__(self):
        super(Netz, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3)
        self.conv2 = nn.Conv2d(6, 12, kernel_size=3)
        self.conv3 = nn.Conv2d(12, 24, kernel_size=3)
        self.conv4 = nn.Conv2d(24, 48, kernel_size=3)
        self.conv5 = nn.Conv2d(48, 96, kernel_size=3)
        self.conv6 = nn.Conv2d(96, 192, kernel_size=3)
        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.max_pool2d(x,2)
        x = F.relu(x)
        x = self.conv6(x)
        x = F.max_pool2d(x,2)
        x = F.relu(x)
        x = self.dropout1(x)
        x = x.view(-1,768)
        x = F.relu(self.dropout2(self.fc1(x)))
        x = self.fc2(x)
        return F.sigmoid(x)
       

model = Netz()
model.cuda()


def train(epoch):
    model.train()
    batch_id = 0
    for data, target in train_data:
        data = data.cuda()
        target = torch.Tensor(target).cuda()
        data = Variable(data)
        target = Variable(target)
        optimizer.zero_grad()
        out = model(data)
        criterion = F.binary_cross_entropy
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:0.6f}'.format(epoch, batch_id, len(train_data), 100. * batch_id / len(train_data), loss.item()))
        batch_id = batch_id + 1

def test():
    model.eval()
    image_path = r"D:\Bilder\CatsVsDogs\test\\";
    files = listdir(image_path)
    f = random.choice(files)
    img = Image.open(image_path + f)
    image_eval_tensor = transform(img)
    image_eval_tensor.unsqueeze_(0)
    data = Variable(image_eval_tensor.cuda())
    out = model(data)
    print(out.data.max(1, keepdim=True)[1])
    img.show()
    x = input('')

def eval_image(path):
    model.eval()
    img = Image.open(path)
    image_eval_tensor = transform(img)
    image_eval_tensor.unsqueeze_(0)
    data = Variable(image_eval_tensor.cuda())
    out = model(data)
    print(out.data.max(1, keepdim=True)[1])
    img.show()
    x = input('')

if os.path.isfile('catdogNetz.pt'):
    model = torch.load('catdogNetz.pt')
else:
    #Target [isCat, isDog]
    image_path = r"D:\Bilder\CatsVsDogs\train\\";
    train_data_list = []
    target_list = []
    train_data = []
    files = listdir(image_path)
    total_files = len(files)

    for i in range(len(files)):
        f = random.choice(files)
        files.remove(f)
        img = Image.open(image_path + f)
        img_tensor = transform(img)
        train_data_list.append(img_tensor)
        isCat = 1 if 'cat' in f else 0
        isDog = 1 if 'dog' in f else 0
        target = [isCat, isDog]
        target_list.append(target)
        if (len(train_data_list) >= 64): #batch size
            train_data.append((torch.stack(train_data_list), target_list))
            train_data_list = []
            target_list = []
            print ('Loaded batch ', len(train_data), ' of ', int(total_files/64))
            print ('Percentage Done: ', round((len(train_data)/(int(total_files)/64)*100), 2), '%')

    model = Netz()
    model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(1, 50):
        train(epoch)
    torch.save(model, 'catdogNetz.pt')

#for wdh in range(1, 10):
    #test()
eval_image(r"D:\Bilder\CatsVsDogs\test3.jpg")
