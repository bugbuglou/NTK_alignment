# code for getting input distributions/covariance/hessian matrices eigenvalues
!pip install pyhessian
from pyhessian import hessian # Hessian computation

datas = []
targets = []
i = 0
length = 500
for d, t in iter(dataloaders['micro_train']):
    datas.append(d.to('cuda'))
    targets.append(t.to('cuda'))
    i += d.size(0)
    if i >= length:
        break
datas = torch.cat(datas)[:length]
targets = torch.cat(targets)[:length]
# dataset = TensorDataset(datas.to('cuda'), targets.to('cuda'))
h = []
a = datas.clone().detach()
T = list(model.features)
Index = []
for i in range(len(T)):
    # get the i-th layer
    if T[i].__class__.__name__ in ['Linear', 'Conv2d', 'BatchNorm1d',
                             'BatchNorm2d', 'GroupNorm']:
        h.append(a.clone().detach())
        Index.append(i)
    a = nn.Sequential(*list(T[i:i+1]))(a)
    
    
dics = [] # a list of dictionaries, each one with key 0-9 each corresponding to a distribution of input vectors with a certain label
means, covs = [],[]  # means:a list of mean 
mean_dists, cov_frobs = [], []  # mean_dists: 
for j in range(len(h)):
    dic = {}
    mean, cov = [],[]
    cov_frob = []
    for i in range(10):
        l = list(targets == i)
        dic[i] = h[j][l,:]
        mean.append(dic[i].mean(dim = 0))
        d = dic[i].view(dic[i].size(0),-1)
        cov_frob.append(cov_frobenius(d))

    dics.append(dic)
    means.append(mean)
    mean_dists.append(torch.trace(mean_distance(mean)))
    cov_frobs.append(cov_frob)

for j in range(len(h)):
    for i in range(10):
        def criterion(outputs, targets):
            outputs = nn.Sequential(*list(model.features[Index[j]+1:]))(outputs)
            return outputs[0,i]
        model_c = nn.Sequential(*list(model.features[Index[j]:Index[j]+1])).train()
        hessian_comp = hessian(model_c, criterion = criterion, data=(means[j][i].unsqueeze(0), torch.FloatTensor([0]).long()), cuda=True)
    
  
