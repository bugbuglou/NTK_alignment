# !pip install nngeometry
import argparse
# from tasks import get_task
import time
import os
import pandas as pd
# from alignment import alignment, layer_alignment, compute_trK
import numpy as np
from nngeometry.object import PVector
from nngeometry.object.fspace import FMatDense
from nngeometry.object.pspace import PMatDense
from nngeometry.object.vector import FVector
from nngeometry.object import PMatImplicit

from nngeometry.generator import Jacobian
from nngeometry.layercollection import LayerCollection
import torch
from torch.nn.functional import one_hot
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


args = {}
args['depth'] = 10#6
args['width'] = 256 #256
args['task'] = 'mnist_fcfree' #'mnist_fc'
# args[align_train] = True
# args[align_test] = True
# args[layer_align_train] = True
# args[layer_align_test] = True
# args[save_ntk_train] = True
# args[save_ntk_test] = True
args['lr'] = 0.001
args['mom'] = 0.9
args['diff'] = 0
args['diff_type'] = 'random'
args['align_easy_diff'] = False
args['epochs'] = 300
args['no_centering'] = False
args['dir'] = '/users/hert5217/Johnny/results' # user fill in
device = 'cpu'


def extract_target_loader(baseloader, target_id, length, batch_size):
    datas = []
    targets = []

    l = target_id
    i = 0
    for d, t in iter(baseloader):
        datas.append(d.to(device))
        targets.append(t.to(device))
        i += d.size(0)
        if i >= length:
            break
    datas = torch.cat(datas)[l]
    targets = torch.cat(targets)[l]
    dataset = TensorDataset(datas.to(device), targets.to(device))

    return DataLoader(dataset, shuffle=False, batch_size=batch_size)

def alignment(model, output_fn, loader, n_output, centering=True):
    lc = LayerCollection.from_model(model)
    generator = Jacobian(layer_collection=lc,
                         model=model,
                         loader=loader,
                         function=output_fn,
                         n_output=n_output,
                         centering=centering)
    targets = torch.cat([args[1] for args in iter(loader)])
    targets = one_hot(targets).float()
    targets -= targets.mean(dim=0)
    targets = FVector(vector_repr=targets.t().contiguous())

    K_dense = FMatDense(generator)
    yTKy = K_dense.vTMv(targets)
    frobK = K_dense.frobenius_norm()

    align = yTKy / (frobK * torch.norm(targets.get_flat_representation())**2)

    return align.item(), K_dense.get_dense_tensor()

def layer_alignment(model, output_fn, loader, n_output, centering=True):
    lc = LayerCollection.from_model(model)
    alignments = []
    # denoms = []
    nums = []
    Ss = []
    targets = torch.cat([args[1] for args in iter(loader)])
    targets = one_hot(targets).float()
    targets -= targets.mean(dim=0)
    targets = FVector(vector_repr=targets.t().contiguous())

    for l in lc.layers.items():
        # print(l)
        lc_this = LayerCollection()
        lc_this.add_layer(*l)

        generator = Jacobian(layer_collection=lc_this,
                             model=model,
                             loader=loader,
                             function=output_fn,
                             n_output=n_output,
                             centering=centering)

        K_dense = FMatDense(generator)
        yTKy = K_dense.vTMv(targets)
        sd = K_dense.data.size()
        # print(sd)
        # print(targets.get_flat_representation().view(-1).shape)
        # print(K_dense.data.shape)
        T = torch.mv(K_dense.data.view(sd[0]*sd[1], sd[2]*sd[3]), targets.get_flat_representation().view(-1))
        T_frob = torch.norm(T)
        # print(T_frob**2)
        # print(torch.dot(T,T))
        # frobK = K_dense.frobenius_norm()
        frobK = K_dense.frobenius_norm()
        S = T_frob**2/(frobK**2*torch.norm(targets.get_flat_representation())**2)
        # S = S/()
        # print(torch.norm(targets.get_flat_representation())**2)
        align = yTKy / (frobK * torch.norm(targets.get_flat_representation())**2)

        alignments.append(align.item())
      
        nums.append(frobK.data/(200))
        Ss.append(S)
    # denoms.append(K_dense.data)

    return alignments, nums, Ss

def layer_alignment_matrix(model, output_fn, loader, n_output, centering=True):
    lc = LayerCollection.from_model(model)
    # alignments = []

    targets = torch.cat([args[1] for args in iter(loader)])
    # targets = one_hot(targets).float()
    # targets -= targets.mean(dim=0)
    # targets = FVector(vector_repr=targets.t().contiguous())
    means, mean_dists, mean_dists_2 ,cov_frobs, covs, ratios, ratios1, ratios2, ratios3, os = [], [], [], [], [], [], [], [], [], []
    for l in lc.layers.items():
        # print(l)
        lc_this = LayerCollection()
        lc_this.add_layer(*l)

        # generator = Jacobian(layer_collection=lc_this,
        #                      model=model,
        #                      loader=loader,
        #                      function=output_fn,
        #                      n_output=n_output,
        #                      centering=centering)

        # K = generator.get_jacobian()
        
        mean1, mean2, cov_frob, o = [], [], [], []
        for i in range(10):
            L = list(targets == i)
            # print(targets[L])
            target_loader = extract_target_loader(loader, L, length = len(L), batch_size = len(L))
            generator = Jacobian(layer_collection=lc_this,
                             model=model,
                             loader=target_loader,
                             function=output_fn,
                             n_output=n_output,
                             centering=False)
            m = generator.get_jacobian()
            # print(m.shape)
            h = m.mean(dim = 1).squeeze()
            # t = h.shape[0]
            # print(h/10)
            t = h[i,:].reshape([1,-1]) - h.mean(dim = 0).unsqueeze(0)
            mean1.append(t)
            s = torch.matmul(t, t.transpose(0,1))
            o.append(s)
            # m = m.reshape([m.size(0),-1])
            # mean.append(m.mean(dim = 0))
            mean2.append(h.reshape([1,-1]))
        for i in range(10):
            L = list(targets == i)
            target_loader = extract_target_loader(loader, L, length = len(L), batch_size = len(L))
            generator = Jacobian(layer_collection=lc_this,
                             model=model,
                             loader=target_loader,
                             function=output_fn,
                             n_output=n_output,
                             centering=True)
            cov = PMatDense(generator).frobenius_norm()
            cov_frob.append(cov)
            
        m = sum(mean1)
        os.append(sum(o))
        # t1 = torch.trace(mean_distance(mean1))
        t1 = torch.matmul(m,m.transpose(0,1))
        t2 = torch.trace(mean_distance(mean2))
        b = sum(cov_frob)
        means.append(mean1)
        mean_dists.append(t1)
        mean_dists_2.append(t2)
        cov_frobs.append(cov_frob)
        covs.append(b) #+ frobenius(mean_distance(mean).cpu())
        ratios.append(t1/(b+t2))
        ratios1.append(b/t2)
        ratios2.append(t2/t1)
        ratios3.append(t2/sum(o))

        # extract_target_loader(baseloader, target, length, batch_size)

    
    return ratios3, mean_dists, mean_dists_2, cov_frobs, covs, ratios, ratios1, ratios2

def get_loss(model, loader):
    model.eval()
    datas = []
    targets = []
    # i = 0
    # length = 2000
    for d, t in iter(loader):
        datas.append(d.to(device))
        targets.append(t.to(device))
        # i += d.size(0)
        # if i >= length:
        #     break
    datas = torch.cat(datas)#[:length]
    targets = torch.cat(targets)#[:length]
    output = model.forward(datas)
    loss = criterion(output, targets)
    l = (1 - torch.exp(-loss))/9
    return l

def SIM(model, loader):
    # model.eval()
    datas = []
    target = torch.cat([args[1] for args in iter(loader)])
    target = one_hot(target).float()
    target -= target.mean(dim=0)
    for d, t in iter(loader):
        datas.append(d.to(device))
    datas = torch.cat(datas)#[:length]

    output = model.forward(datas)
    O = output.exp()
    T = 1/(O.sum(dim = 1).unsqueeze(1))
    # O = O * T
    f = O * T
    w_0 = np.asarray(f.view(-1).cpu().detach()) - np.asarray(target.view(-1).cpu().detach())

    return np.matmul(w_0.reshape([-1,1]), w_0.reshape([1,-1]))

def similarity_w_0_y(Mat, loader):
    target = torch.cat([args[1] for args in iter(loader)])
    target = one_hot(target).float()
    target -= target.mean(dim=0)
    y_1 = np.asarray(target.view(-1).cpu().detach())
    # W_inc = SIM(model, target, y_1, loader = dataloaders['micro_train'])
    # W = W + W_inc
    y_post = np.matmul(Mat, y_1.reshape([-1,1]))
    similarity = (np.dot(y_post.reshape([-1]), y_1)/(np.linalg.norm(y_post)*np.linalg.norm(y_1)))
    
    return similarity

def two_terms(model, output_fn, loader, n_output, centering=True):
    datas = []
    target = torch.cat([args[1] for args in iter(loader)])
    # tar = target
    target = one_hot(target).float().to(device)
    # tar1 = target
    # target -= target.mean(dim=0)
    for d, t in iter(loader):
        datas.append(d.to(device))
    datas = torch.cat(datas)#[:length]

    output = model.forward(datas)
    O = output.exp()
    T = 1/(O.sum(dim = 1).unsqueeze(1))
    # O = O * T
    f = O * T 
    f -= target
    target -= target.mean(dim=0)
    targets = FVector(vector_repr=target.t().contiguous())
    w_0s = FVector(vector_repr=f.t().contiguous())
    #np.asarray(f.view(-1).cpu().detach()) - np.asarray(target.view(-1).cpu().detach())
    # w_0 = torch.FloatTensor(np.asarray(f.reshape([1,-1]).transpose(0,1).squeeze(1).cpu().detach()) - np.asarray(target.reshape([1,-1]).transpose(0,1).squeeze(1).cpu().detach())).to('cuda')
    # target -= target.mean(dim=0)
    # targets = FVector(vector_repr=target.t().contiguous())
    # target.to('cuda')
    target = targets.get_flat_representation().view(-1)
    w_0 = w_0s.get_flat_representation().view(-1)
    # del targets
    # del w_0s
    align = torch.dot(w_0, target)/(torch.norm(target)*torch.norm(w_0))
    

   
    return align

def compute_trK(align_dl, model, output_fn, n_output):
    generator = Jacobian(model, align_dl, output_fn, n_output=n_output)
    F = PMatImplicit(generator)
    return F.trace().item() * len(align_dl)
  


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class FC(nn.Module):
    def __init__(self, depth, width, bn=False, base=0):
        super(FC, self).__init__()
        self.bn = bn 
        base = base if base != 0 else 64
        self.base = base
        self.features = self._make_layers(depth, width)
        # self.classifier = nn.Linear(8 * base, 10)

    def forward(self, x):
        out = self.features(x)
        # out = out.view(out.size(0), -1)
        # out = self.classifier(out)
        return out

    def _make_layers(self, depth, width):
        layers = []
        # in_channels = 3
        for i in range(depth):
            if i==0:
                layers += [nn.Flatten(), nn.Linear(28 * 28, width), nn.ReLU()]
            elif i<depth-1:
                layers += [nn.Linear(width, width), nn.ReLU()]
            else:
                layers += [nn.Linear(width, 10)]
        # layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class VGG(nn.Module):
    def __init__(self, vgg_name, bn=False, base=0):
        super(VGG, self).__init__()
        self.bn = bn 
        base = base if base != 0 else 64
        self.base = base
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(8 * base, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                out_channels = x * self.base // 64
                layers += [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)]
                if self.bn:
                    layers += [nn.BatchNorm2d(x)]
                layers += [nn.ReLU()]
                in_channels = out_channels
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, bn=False):
        super(BasicBlock, self).__init__()
        self.bn = bn
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=not bn)
        if bn:
            self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=not bn)
        if bn:
            self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            if self.bn:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion*planes)
                )
            else:
                self.shortcut = nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        if self.bn:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
        else:
            out = F.relu(self.conv1(x))
            out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, bn=False):
        super(ResNet, self).__init__()
        self.bn = bn
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=not bn)
        if bn:
            self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.bn))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.bn:
            out = self.bn1(self.conv1(x))
        else:
            out = self.conv1(x)
        out = F.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(bn=True):
    return ResNet(BasicBlock, [2,2,2,2], bn=bn)

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])

import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, MNIST, KMNIST, FashionMNIST
# from models import VGG, ResNet18
import random
import numpy as np

default_datapath = '/tmp/data'
if 'SLURM_TMPDIR' in os.environ:
    default_datapath = os.path.join(os.environ['SLURM_TMPDIR'], 'data')

def to_tensordataset(dataset):
    d = next(iter(DataLoader(dataset,
                  batch_size=len(dataset))))
    return TensorDataset(d[0].to(device), d[1].to(device))

def extract_small_loader(baseloader, length, batch_size):
    datas = []
    targets = []
    i = 0
    for d, t in iter(baseloader):
        datas.append(d.to(device))
        targets.append(t.to(device))
        i += d.size(0)
        if i >= length:
            break
    datas = torch.cat(datas)[:length]
    targets = torch.cat(targets)[:length]
    dataset = TensorDataset(datas.to(device), targets.to(device))

    return DataLoader(dataset, shuffle=False, batch_size=batch_size)

def kaiming_init(net, tanh=False):
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            if tanh:
                nn.init.xavier_normal_(m.weight)
            else:
                nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            if m.affine:
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

def get_cifar10(args):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = CIFAR10(root=default_datapath, train=True, download=True,
                       transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)

    testset = CIFAR10(root=default_datapath, train=False, download=True,
                      transform=transform_test)
    testloader = DataLoader(testset, batch_size=1000, shuffle=False)

    return trainloader, testloader

def get_fmnist_normalization(args):
    trainset_mnist = FashionMNIST(default_datapath, train=True, download=True)
    mean_mnist = (trainset_mnist.data.float()/255).mean()
    std_mnist = (trainset_mnist.data.float()/255).std()
    if args['diff'] == 0 or args['diff_type'] == 'random':
        return mean_mnist.item(), std_mnist.item()

    # otherwise we need to include kmnist before normalization
    trainset_kmnist = KMNIST(default_datapath, train=True, download=True)
    mean_kmnist = (trainset_kmnist.data.float() / 255).mean()
    std_kmnist = (trainset_kmnist.data.float() / 255).std()

    mean_both = args.diff * mean_kmnist + (1 - args.diff) * mean_mnist
    std_both = (args.diff * std_kmnist**2 + (1 - args.diff) * std_mnist**2) ** .5
    return mean_both.item(), std_both.item()

def get_mnist_normalization(args):
    trainset_mnist = MNIST(default_datapath, train=True, download=True)
    mean_mnist = (trainset_mnist.data.float() / 255).mean()
    std_mnist = (trainset_mnist.data.float() / 255).std()
    if args['diff'] == 0 or args['diff_type'] == 'random':
        return mean_mnist.item(), std_mnist.item()

    # otherwise we need to include kmnist before normalization
    trainset_kmnist = KMNIST(default_datapath, train=True, download=True)
    mean_kmnist = (trainset_kmnist.data.float() / 255).mean()
    std_kmnist = (trainset_kmnist.data.float() / 255).std()

    mean_both = args.diff * mean_kmnist + (1 - args.diff) * mean_mnist
    std_both = (args.diff * std_kmnist**2 + (1 - args.diff) * std_mnist**2) ** .5
    return mean_both.item(), std_both.item()

def get_mnist(args):
    mean, std = get_mnist_normalization(args)
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,)),
    ])

    trainset = MNIST(root=default_datapath, train=True, download=True,
                     transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)

    testset = MNIST(root=default_datapath, train=False, download=True,
                    transform=transform_train)
    testloader = DataLoader(testset, batch_size=1000, shuffle=False)

    return trainloader, testloader


def get_fmnist(args):
    mean, std = get_fmnist_normalization(args)
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,)),
    ])

    trainset = FashionMNIST(root=default_datapath, train=True, download=True,
                     transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)

    testset = FashionMNIST(root=default_datapath, train=False, download=True,
                    transform=transform_train)
    testloader = DataLoader(testset, batch_size=1000, shuffle=False)

    return trainloader, testloader

def add_difficult_examples(dataloaders, args):
    # adds difficult examples and extract small
    # dataloaders
    if args['diff_type'] == 'random':
        trainset = dataloaders['train'].dataset
        x_easy = []
        y_easy = []
        x_diff = []
        y_diff = []
        for i in range(len(trainset.targets)):
            if random.random() < args['diff']:
                trainset.targets[i] = random.randint(0, 9)
                x_diff.append(trainset[i][0])
                y_diff.append(trainset.targets[i])
            else:
                x_easy.append(trainset[i][0])
                y_easy.append(trainset.targets[i])
        # print(x_easy)
        x_easy = torch.stack(x_easy)
        y_easy = torch.tensor(y_easy)
        x_diff = torch.stack(x_diff)
        y_diff = torch.tensor(y_diff)
    elif args['diff_type'] == 'other' and args['task'][:5] == 'mnist':
        trainset = dataloaders['train'].dataset
        trainset_kmnist = KMNIST(default_datapath, train=True, download=True,
                                 transform=trainset.transform)
        mnist_len = len(trainset)
        kmnist_len = int(args['diff'] * mnist_len)
        indices = np.arange(len(trainset_kmnist))
        np.random.shuffle(indices)
        indices = indices[:kmnist_len]

        # apply transforms by hand
        x_easy = []
        y_easy = []
        x_diff = []
        y_diff = []
        for i in range(len(trainset.targets)):
            x_easy.append(trainset[i][0])
            y_easy.append(trainset.targets[i])
        for i in indices:
            x_diff.append(trainset_kmnist[i][0])
            y_diff.append(trainset_kmnist.targets[i])
        x_easy = torch.stack(x_easy)
        y_easy = torch.tensor(y_easy)
        x_diff = torch.stack(x_diff)
        y_diff = torch.tensor(y_diff)

        x = torch.cat([x_easy, x_diff])
        y = torch.cat([y_easy, y_diff])
        trainset_both = TensorDataset(x, y)
        dataloaders['train'] = DataLoader(trainset_both, batch_size=128, shuffle=True)
    else:
        raise NotImplementedError

    indices = np.arange(len(y_easy))
    np.random.shuffle(indices)
    indices = indices[:100]
    x_easy = x_easy[indices]
    y_easy = y_easy[indices]

    indices = np.arange(len(y_diff))
    np.random.shuffle(indices)
    indices = indices[:100]
    x_diff = x_diff[indices]
    y_diff = y_diff[indices]

    dataloaders['micro_train_easy'] = DataLoader(TensorDataset(x_easy.to(device), y_easy.to(device)),
                                                 batch_size=100, shuffle=False)
    dataloaders['micro_train_diff'] = DataLoader(TensorDataset(x_diff.to(device), y_diff.to(device)),
                                                 batch_size=100, shuffle=False)


def get_task(args):
    dataloaders = dict()

    task_name, model_name = args['task'].split('_')

    if task_name == 'cifar10':
        if args['depth'] != 0:
            raise NotImplementedError
        dataloaders['train'], dataloaders['test'] = get_cifar10(args)
        if model_name == 'vgg19':
            model = VGG('VGG19', base=args['width'])
        elif model_name == 'resnet18':
            model = ResNet18()
            if args['width'] != 0:
                raise NotImplementedError
    elif task_name == 'mnist':
        dataloaders['train'], dataloaders['test'] = get_mnist(args)
        if model_name == 'fc':
            layers = [nn.Flatten(), nn.Linear(28 * 28, args['width']), nn.ReLU()] + \
                     [nn.Linear(args['width'], args['width']), nn.ReLU()] * (args['depth'] - 2) + \
                     [nn.Linear(args['width'], 10)]
            model = nn.Sequential(*layers)
        elif model_name == 'fcfree':
            model = FC(depth = args['depth'], width = args['width'])
        else:
            raise NotImplementedError

    elif task_name == 'fmnist':
        dataloaders['train'], dataloaders['test'] = get_fmnist(args)
        if model_name == 'fc':
            layers = [nn.Flatten(), nn.Linear(28 * 28, args['width']), nn.ReLU()] + \
                     [nn.Linear(args['width'], args['width']), nn.ReLU()] * (args['depth'] - 2) + \
                     [nn.Linear(args['width'], 10)]
            model = nn.Sequential(*layers)
        elif model_name == 'fcfree':
            model = FC(depth = args['depth'], width = args['width'])
        elif model_name == 'resnet':
            model = ResNet18()
            if args['width'] != 0:
                raise NotImplementedError
        else:
            raise NotImplementedError
    

    model = model.to(device)
    kaiming_init(model)

    criterion = nn.CrossEntropyLoss()

    if args['align_easy_diff']:
        add_difficult_examples(dataloaders, args)

    # if args[align_train or args.layer_align_train or args.save_ntk_train or args.complexity:
    dataloaders['micro_train'] = extract_small_loader(dataloaders['train'], 2000, 200)
    # if args.align_test or args.layer_align_test or args.save_ntk_test:
    dataloaders['micro_test'] = extract_small_loader(dataloaders['test'], 2000, 200)
    dataloaders['mini_test'] = extract_small_loader(dataloaders['test'], 1000, 1000)

    return model, dataloaders, criterion
  
# model1, dataloaders, criterion = get_task(args)
model1, dataloaders, criterion = get_task(args)
dir = args['dir']
task_dis = args['task'].split('_')[0] + '_' + str(args['depth']) + '_' + str(args['width'])
model_path = os.path.join(dir, task_dis + '_model_two_terms')
torch.save(model1, model_path)
# model1 = torch.load(model_path)

lrs = [0.001, 0.003, 0.005, 0.01, 0.02] #1e-5, 5e-5, 1e-4, 5e-4, 
# lrs = [1e-4, 5e-4]

models, optimizers,result_dirs = [], [], []
for i in range(len(lrs)):
    model_alt = FC(depth = args['depth'], width = args['width'])
    model_alt.load_state_dict(model1.state_dict())
    model_alt = model_alt.to(device)
    models.append(model_alt)
    params = [param for name, param in models[i].named_parameters()] # if 'in_features' in name or 'out_features' in name]
    # print(params)
    optimizers.append(optim.SGD(params, lrs[i], momentum=args['mom']))
    result_dir = os.path.join(dir, 'lr = ' + str(lrs[i]) + ',' + task_dis + '_two_terms')
    try:
        os.mkdir(result_dir)
    except:
        print('I will be overwriting a previous experiment')
    result_dirs.append(result_dir)

# optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=args['mom'], weight_decay=5e-4)

class RunningAverageEstimator:

    def __init__(self, gamma=.9):
        self.estimates = dict()
        self.gamma = gamma

    def update(self, key, val):
        if key in self.estimates.keys():
            self.estimates[key] = (self.gamma * self.estimates[key] +
                                   (1 - self.gamma) * val)
        else:
            self.estimates[key] = val

    def get(self, key):
        return self.estimates[key]


def stopping_criterion(log):
    if (log.loc[len(log) - 1]['train_loss'] < 1e-2
            and log.loc[len(log) - 2]['train_loss'] < 1e-2):
        return True
    return False

def do_compute_ntk(iterations):
    return iterations == 0 or iterations in 5 * (1.15 ** np.arange(300)).astype('int')
#     return iterations > 500 and iterations in (len(dataloaders['train'])) * (np.arange(args['epochs'])).astype('int') #iterations == 0 or iterations in 5 * (1.15 ** np.arange(300)).astype('int')

# Training
def train(model, optimizer, args, log, result_dir):
    def output_fn(x, t):
        return model(x)
    rae = RunningAverageEstimator()
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    iterations = 0

    # if args.complexity:
    #     w_before = PVector.from_model(model).clone().detach()

    for epoch in range(args['epochs']):
        print('\nEpoch: %d' % epoch)
        if len(log) >= 2 and stopping_criterion(log):
            print('stopping now')
            break
        for batch_idx, (inputs, targets) in enumerate(dataloaders['micro_train']):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            _, pred = outputs.max(1)
            acc = pred.eq(targets.view_as(pred)).float().mean()

            rae.update('train_loss', loss.item())
            rae.update('train_acc', acc.item())

            if do_compute_ntk(iterations):
                to_log = pd.Series()
                # to_log['time'] = time.time() - start_time
            # if args.layer_align_train:
                # to_log['layer_align_train'] = \
                #     layer_alignment(model, output_fn, dataloaders['micro_train'], 10,
                #                     centering=not args['no_centering'])
            # # if args.layer_align_test:
                # to_log['layer_align_test'] = \
                #     layer_alignment(model, output_fn, dataloaders['micro_test'], 10,
                #                     centering=not args['no_centering'])
                    
                # to_log['l'] = get_loss(model, dataloaders['micro_train'])
                # to_log['l_test'] = get_loss(model, dataloaders['micro_test'])

                # if len(log) > 0:
                #     W= SIM(model, dataloaders['micro_train']) + W
                # else:
                #     W = SIM(model, dataloaders['micro_train'])

                # to_log['corr_y_train'] = SIM(model, dataloaders['micro_train'])
                # to_log['corr_y_test'] = SIM(model, dataloaders['micro_test'])

                to_log['two_terms_train_1'] = two_terms(model, output_fn, dataloaders['micro_train'], 10, centering=True)
                to_log['two_terms_test_1'] = two_terms(model, output_fn, dataloaders['micro_test'], 10, centering=True)

                    # means, mean_dists, cov_frobs, covs
                # to_log['layer_align_train_ratio3'], to_log['layer_align_train_means'],  to_log['layer_align_train_means_2'], _, to_log['layer_align_train_covs'], to_log['layer_align_train_ratio'], to_log['layer_align_train_ratio1'], to_log['layer_align_train_ratio2'] = \
                #     layer_alignment_matrix(model, output_fn, dataloaders['micro_train'], 10,
                #                     centering=not args['no_centering'])
                # # print(len(to_log['layer_align_train_means']))
                # # print(len(to_log['layer_align_train_covs']))
                # to_log['layer_align_test_ratio3'], to_log['layer_align_test_means'],  to_log['layer_align_test_means_2'], _, to_log['layer_align_test_covs'], to_log['layer_align_test_ratio'],to_log['layer_align_test_ratio1'], to_log['layer_align_test_ratio2'] = \
                #     layer_alignment_matrix(model, output_fn, dataloaders['micro_test'], 10,
                #                     centering=not args['no_centering'])
                # l1, l2 = [], []
                # for i in range(len(to_log['layer_align_train_means'])):
                #     l1.append(to_log['layer_align_train_means'][i]/to_log['layer_align_train_covs'][i])
                #     l2.append(to_log['layer_align_test_means'][i] / to_log['layer_align_test_covs'][i])
                # to_log['layer_align_train_ratio'] = l1
                # to_log['layer_align_test_ratio'] = l2
            # if args.align_train or args.save_ntk_train:
            #     to_log['align_train'], _ = alignment(model, output_fn, dataloaders['micro_train'],
            #                                             10, centering=not args['no_centering'])
            #     # if args.save_ntk_train:
            #     # ntk_path = os.path.join(results_dir,'train_ntk_%.6d.pkl' % iterations)
            #     # torch.save(ntk, ntk_path)
            # # if args.align_test or args.save_ntk_test:
            #     to_log['align_test'], _ = alignment(model, output_fn, dataloaders['micro_test'],
            #                                           10, centering=not args['no_centering'])

                # to_log['forward_train_mean'], to_log['forward_train_cov_frob'], to_log['forward_train_ratio'] = forward_data(model, dataloaders['micro_train'])
                # to_log['forward_test_mean'], to_log['forward_test_cov_frob'], to_log['forward_test_ratio'] = forward_data(model, dataloaders['micro_test'])
                # if args.save_ntk_test:
                # ntk_path = os.path.join(results_dir,'test_ntk_%.6d.pkl' % iterations)
                # torch.save(ntk, ntk_path)
                if args['align_easy_diff']:
                    to_log['align_easy_train'], ntk = alignment(model, output_fn,
                                                                dataloaders['micro_train_easy'],
                                                                10, centering=not args['no_centering'])
                    to_log['align_diff_train'], ntk = alignment(model, output_fn,
                                                                dataloaders['micro_train_diff'],
                                                                10, centering=not args['no_centering'])
                # if args.complexity:
                #     w_after = PVector.from_model(model).clone().detach()
                #     to_log['norm_dw'] = torch.norm((w_after - w_before).get_flat_representation()).item()
                #     w_before = w_after
                #     to_log['trK'] = compute_trK(dataloaders['micro_train'], model, output_fn, 10)

                to_log['iteration'] = iterations
                to_log['epoch'] = epoch
                to_log['train_acc'], to_log['train_loss'] = rae.get('train_acc'), rae.get('train_loss')
                to_log['test_acc'], to_log['test_loss'] = test(model, dataloaders['mini_test'])


                log.loc[len(log)] = to_log
                print(log.loc[len(log) - 1])

                log.to_pickle(os.path.join(result_dir,'log2.pkl'))
                

            iterations += 1

def test(model, loader):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    model.train()
    return correct / total, test_loss / (batch_idx + 1)


# name = ''
# for k, v in sorted(args.items(), key=lambda a: a[0]):
#     if (k not in ['save_ntk_train', 'save_ntk_test']
#         and v is not False):
#         name += '%s=%s,' % (k, str(v))
# name = name[:-1]
# results_dir = os.path.join('/content/drive/MyDrive/NTK_alignment_2/', name)

# try:
#     os.mkdir(results_dir)
# except:
#     print('I will be overwriting a previous experiment')

columns = ['iteration', 'time', 'epoch',
           'train_loss', 'train_acc',
           'test_loss', 'test_acc']
# if args.layer_align_train:
columns.append('layer_align_train')
# if args.layer_align_test:
columns.append('layer_align_test')
# if args.align_train or args.save_ntk_train:
columns.append('align_train')
# if args.align_test or args.save_ntk_test:
columns.append('align_test')
if args['align_easy_diff']:
    columns.append('align_easy_train')
    columns.append('align_diff_train')
# columns.append('layer_align_train_means')
# columns.append('layer_align_test_means')
# columns.append('layer_align_train_covs')
# columns.append('layer_align_test_covs')
# columns.append('layer_align_train_ratio')
# columns.append('layer_align_test_ratio')
# columns.append('layer_align_train_ratio1')
# columns.append('layer_align_train_ratio2')
# columns.append('layer_align_test_ratio1')
# columns.append('layer_align_test_ratio2')
# columns.append('layer_align_train_means_2')
# columns.append('layer_align_test_means_2')
# columns.append('forward_train_mean')
# columns.append('forward_train_cov_frob')
# columns.append('forward_train_ratio')
# columns.append('forward_test_mean')
# columns.append('forward_test_cov_frob')
# columns.append('forward_test_ratio')
# columns.append('layer_align_train_ratio3')
# columns.append('layer_align_test_ratio3')
# columns.append('l')
# columns.append('l_test')
# # columns.append('W')
# columns.append('corr_y_train')
# columns.append('corr_y_test')
columns.append('two_terms_train_1')
columns.append('two_terms_train_2')
columns.append('two_terms_test_1')
columns.append('two_terms_test_2')

# if argcolumns.append('layer_align_train_ratio3')s.complexity:
#     columns += ['trK', 'norm_dw']

for i in range(len(lrs)):
#     print(i)
    log = pd.DataFrame(columns=columns)
    train(models[i], optimizers[i], args, log, result_dirs[i])
