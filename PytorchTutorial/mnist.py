import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms

kwargs = {'num_workers': 0, 'pin_memory': True}
train_data = torch.utils.data.DataLoader(datasets.MNIST('data', train=True, download=True,
                                                    transform=transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize([0.1307], [0.3081])])),
                                     batch_size=256, shuffle=True, **kwargs) #Handschrift Train Datensatz(MNIST)

test_data = torch.utils.data.DataLoader(datasets.MNIST('data', train=False,
                                                    transform=transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize((0.1307,), (0.3081,))])),
                                     batch_size=64, shuffle=True, **kwargs) #Handschrift Datensatz zu evaluieren (MNIST)

class Netz(nn.Module):
    def __init__(self):
        super(Netz, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv_dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 60)
        self.fc2 = nn.Linear(60, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.conv_dropout(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = x.view(-1, 320)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x)




def train(epoch):
    model.train()
    for batch_id, (data, target) in enumerate(iter(train_data), 0):
        data = data.cuda()
        target = target.cuda()

        data = Variable(data)
        target = Variable(target)
        optimizer.zero_grad()
        out = model(data)
        criterion = F.nll_loss
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:0.6f}'.format(epoch, batch_id * len(data), len(train_data.dataset), 100. * batch_id / len(train_data), loss.item()))

def test():
    model.eval()
    loss = 0
    correct = 0
    for data, target in test_data:
        data = Variable(data.cuda(), volatile=True)
        target = Variable(target.cuda())
        out = model(data)
        loss += F.nll_loss(out, target, reduction='sum').item()
        prediction = out.data.max(1, keepdim=True)[1]
        correct += prediction.eq(target.data.view_as(prediction)).cpu().sum()

    loss = loss/len(test_data.dataset)
    correct = 100.*correct/len(test_data.dataset)
    print('Durchschnitts-loss: ', loss)
    print('Genauigkeit: ', correct, '%')


if os.path.isfile('mnistNetz.pt'):
    model = torch.load('mnistNetz.pt')
else:
    model = Netz()
    model = model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.8)
    for epoch in range(1, 5):
        train(epoch)
    torch.save(model, 'mnistNetz.pt')

test()

    