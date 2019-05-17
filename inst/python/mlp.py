import torch
import torch.utils.data
from torch import nn, optim

import numpy as np
import random


def mlp(traindata, testdata=None,D_in=1,D_H1=7,D_H2=5,D_out=1, batch_size=128, epochs=30,cuda=False, seed=0,
          log_interval=10, learning_rate=1e-2,verbose=False, debug=False):

    torch.set_num_threads(1)
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    model = MLP(D_in,D_H1,D_H2,D_out).to(device)

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)


    #kwargs={}
    kwargs = {'num_workers': 1, 'pin_memory': False}
    traindata = torch.from_numpy(traindata).float()

    D_row, D_col = traindata.size()

    train_loader = torch.utils.data.DataLoader(traindata, batch_size=batch_size, shuffle=True,**kwargs)
    if testdata is not None:
        testdata = torch.from_numpy(testdata).float()
        test_loader = torch.utils.data.DataLoader(testdata, batch_size=batch_size, shuffle=True,**kwargs)

    optimizer = optim.Adam([{'params': model.parameters()}],
                           lr=learning_rate)

    loss_function=nn.MSELoss()
    score = []
    score_test = []
    for epoch in range(1, epochs + 1):
        # train(epoch)
        model.train()
        train_loss = 0
        for batch_idx,data in enumerate(train_loader):

            data = data.to(device)
            optimizer.zero_grad()
            x=data[:,0].view(-1,1)
            y=data[:,1].view(-1,1)
            yhat = model(x)
            loss = loss_function(yhat,y)
            loss.backward()

            train_loss += loss.item()
            optimizer.step()
            if verbose and batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader),
                           loss.item() / len(data)))

        train_loss /= len(train_loader.dataset)

        score.append(-train_loss)
        if verbose:
            print('====> Epoch: {} Average loss: {:.4f}'.format(
                epoch, train_loss))



    if debug:
        return model
    else:
        yhat=model(traindata[:,0]).detach().numpy()
        return yhat



class MLP(nn.Module):
    def __init__(self, D_in,D_H1,D_H2,D_out):
        super(MLP, self).__init__()
        self.D_in=D_in
        self.D_H1 = D_H1
        self.D_H2 = D_H2
        self.D_out = D_out
        self.fc1 = nn.Linear(D_in, D_H1)
        self.fc2 = nn.Linear(D_H1, D_H2)
        self.fc3 = nn.Linear(D_H2, D_out)
        self.relu = nn.ReLU()

    def forward(self, data):
        data = data.view(-1, self.D_in)
        h1=self.relu(self.fc1(data))
        h2=self.relu(self.fc2(h1))

        return self.fc3(h2)


