import torch
import torch.utils.data
from torch import nn, optim
from scipy.stats import gaussian_kde

import numpy as np
import random



def fit(traindata, testdata=None, N=1, beta=1, batch_size=128, epochs=50, logpx=None, cuda=False, seed=0,
        log_interval=10, learning_rate=1e-2, prior_sdy=0.5, update_sdy=True,preload=False, warming_up=False, verbose=False,
        debug=False):
    """

    :param traindata: Traning data of causal pair X, Y.
    :param testdata: Testing data of causal pair X, Y.
    :param N: The number of latent intermediated variables
    :param beta: The beta parameters in term of beta-VAE for controlling the size of KL-divergence in ELBO
    :param batch_size: The batch size.
    :param epochs: The training epochs
    :param logpx: The average log-likelihood for p(x) at each sample.
    :param cuda: Whether use GPU.
    :param seed: The random seed
    :param log_interval: The option of verbose that output the training detail at each intervals.
    :param learning_rate: The learning rate.
    :param prior_sdy: The initialization of the standard error at noise distribution.
    :param update_sdy: Whether update the noise distribution by using gradient decent.
    :param preload: If preload=True, the traindata and testdata must be the object of the torch.utils.data.DataLoader.
    :param warming_up: Using a warming up strategy for the beta.
    :param verbose: print output
    :param debug: If debug=True, the output will return the model object for debug.
    :return:

    """
    torch.set_num_threads(1)
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    model = CANM(N).to(device)

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

    # kwargs={}
    kwargs = {'num_workers': 1, 'pin_memory': False}
    if logpx is None:
        pde = gaussian_kde(traindata[0,:])
        logpx=np.log(pde(traindata[0,:])).mean()
    if preload:
        train_loader = traindata
        test_loader = testdata

    if not preload:
        traindata = torch.from_numpy(traindata).float()
        train_loader = torch.utils.data.DataLoader(traindata, batch_size=batch_size, shuffle=True, **kwargs)
    if testdata is not None and not preload:
        testdata = torch.from_numpy(testdata).float()
        test_loader = torch.utils.data.DataLoader(testdata, batch_size=batch_size, shuffle=True, **kwargs)

    if update_sdy:
        sdy = torch.tensor([prior_sdy], device=device, dtype=torch.float, requires_grad=True)
        optimizer = optim.Adam([{'params': model.parameters()}, {'params': sdy}],
                               lr=learning_rate)  # Use Adam
    else:
        sdy = torch.tensor([prior_sdy], device=device, dtype=torch.float, requires_grad=False)
        optimizer = optim.Adam([{'params': model.parameters()}], lr=learning_rate)


    score = []
    score_test = []
    for epoch in range(1, epochs + 1):
        # train(epoch)
        model.train()
        train_loss = 0
        if warming_up:
            wu_beta = beta / epoch
        else:
            wu_beta = beta
        for batch_idx, data in enumerate(train_loader):

            data = data.to(device)
            optimizer.zero_grad()
            # x = data[:, 0]
            y = data[:, 1].view(-1, 1)
            yhat, mu, logvar = model(data)

            loss = loss_function(y, yhat, mu, logvar, sdy, wu_beta) - logpx * len(data)
            loss.backward()

            train_loss += loss.item()
            optimizer.step()
            if update_sdy and sdy < 0.01: # Ensuring the sdy larger than 0.01 to avoid the NAN loss.
                sdy = sdy + 0.01
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

        # test(epoch)
        if testdata is not None:
            model.eval()
            test_loss = 0
            with torch.no_grad():
                for i, data in enumerate(test_loader):
                    data = data.to(device)
                    yhat, mu, logvar = model(data)
                    # x = data[:, 0]
                    y = data[:, 1].view(-1, 1)
                    test_loss += loss_function(y, yhat, mu, logvar, sdy, wu_beta).item() - logpx * len(data)

                test_loss /= len(test_loader.dataset)

                score_test.append(-test_loss)
                if verbose:
                    print('====> Test set loss: {:.4f}'.format(test_loss))

    if testdata is not None:
        output={'train_likelihood': -np.float(train_loss), 'test_likelihood': -np.float(test_loss),
                'train_score': score, 'test_score': score_test, 'sdy': sdy.detach().numpy()}
    else:
        output={'train_likelihood': -np.float(train_loss), 'train_score': score, 'sdy': sdy.detach().numpy()}
    if debug:
        output['model']=model

    return output


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(y, yhat, mu, logvar, sdy, beta):
    # BCE = F.binary_cross_entropy(recon_x, x.view(-1, 2), size_average=False)
    # D=D.add_([xhat,yhat])
    N = y - yhat

    if sdy.item() <= 0:
        sdy = -sdy + 0.01

    n = torch.distributions.Normal(0, sdy)
    BCE = - torch.sum(n.log_prob(N)) # Compute the log-likelihood of noise distribution.

    # BCE=F.mse_loss(torch.cat((xhat,yhat),1),D)
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) * beta
    # KLD=0
    return BCE + KLD



class CANM(nn.Module):
    def __init__(self, N):  # N: The number of latent variables
        super(CANM, self).__init__()
        self.N = N
        # encoder 输入只有y
        self.fc1 = nn.Linear(2, 20)
        # 均值网络
        self.fc21 = nn.Linear(20, 12)
        self.fc22 = nn.Linear(12, 7)
        self.fc23 = nn.Linear(7, N)

        # 方差网络
        self.fc31 = nn.Linear(20, 12)
        self.fc32 = nn.Linear(12, 7)
        self.fc33 = nn.Linear(7, N)

        # decoder
        self.fc4 = nn.Linear(1 + N, 10)  # yhat=f(x,z)
        # self.fc4 = nn.Linear(1, 10)  # yhat=f(x,z)

        self.fc5 = nn.Linear(10, 7)
        self.fc6 = nn.Linear(7, 5)
        self.fc7 = nn.Linear(5, 1)  # yhat

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def encode(self, xy):
        xy = xy.view(-1, 2)
        h1 = self.relu(self.fc1(xy))

        h21 = self.relu(self.fc21(h1))
        h22 = self.relu(self.fc22(h21))
        mu = self.fc23(h22)

        h31 = self.relu(self.fc31(h1))
        h32 = self.relu(self.fc32(h31))
        logvar = self.fc33(h32)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)  # e^(0.5*logvar)=e^(logstd)=std
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, x, z):
        x = x.view(-1, 1)
        z = z.view(-1, self.N)
        h4 = self.relu(self.fc4(torch.cat((x, z), 1)))
        h5 = self.relu(self.fc5(h4))
        h6 = self.relu(self.fc6(h5))
        yhat = self.fc7(h6)
        return yhat

    def forward(self, data):
        data = data.view(-1, 2)
        x = data[:, 0]
        y = data[:, 1]

        mu, logvar = self.encode(data)
        z = self.reparameterize(mu, logvar)
        yhat = self.decode(x, z)
        return yhat, mu, logvar

if __name__ == "__main__":
    np.random.seed(0)
    X=np.random.normal(0,1,10000)
    Z=np.power(X,3)+0.5*np.random.normal(0,1,10000)
    Y=np.tanh(2*Z)+0.5*np.random.normal(0,1,10000)
    traindata1=np.array([X, Y]).transpose()
    result1=fit(traindata1,verbose=True)
    traindata2=np.array([Y, X]).transpose()
    result2=fit(traindata2,N=1,verbose=True)
    if np.max(result1['train_score'])>np.max(result2['train_score']):
        print('X->Y')
    else:
        print('Y->X')
