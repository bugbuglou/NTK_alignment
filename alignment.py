from nngeometry.object.fspace import FMatDense
from nngeometry.object.pspace import PMatDense
from nngeometry.object.vector import FVector
from nngeometry.object import PMatImplicit

from nngeometry.generator import Jacobian
from nngeometry.layercollection import LayerCollection
import torch
from torch.nn.functional import one_hot

def frobenius(matrix):
    m = torch.FloatTensor(matrix)
    return torch.sqrt(torch.sum(torch.square(m)))

def mean_distance(mean):
    m = torch.stack(mean,dim = 0).reshape([10,-1])
    if args['no_centering'] == False:
        M = m.mean(dim = 0)
        m = m - M
    # print(m.shape)
    return torch.matmul(m, m.transpose(1,0))

def cov_frobenius(d):
    # print(d.shape[1])
    step = min(20480,d.shape[1])
    # cov_matrix = torch.zeros([d.shape[1],d.shape[1]])
    sum = 0
    for i in range(int(d.shape[1]/step)):
        f = min((i+1)*step, d.shape[1])
        for j in range(int(d.shape[1]/step)):
            t = min((j+1)*step, d.shape[1])
            h = d[:, i*step:f]
            s = d[:, j*step:t].transpose(1,0)
            # cov_matrix[j*step:(j+1)*step, i*step:(i+1)*step] = torch.matmul(s, h)/(h.size(0)-1)
            sum += torch.sum(torch.square(torch.matmul(s, h)/(h.size(0))))
            del h,s
    return torch.sqrt(sum)

def extract_target_loader(baseloader, target_id, length, batch_size):
    datas = []
    targets = []
    l = target_id
    i = 0
    for d, t in iter(baseloader):
        datas.append(d.to('cuda'))
        targets.append(t.to('cuda'))
        i += d.size(0)
        if i >= length:
            break
    datas = torch.cat(datas)[l]
    targets = torch.cat(targets)[l]
    dataset = TensorDataset(datas.to('cuda'), targets.to('cuda'))

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
        frobK = K_dense.frobenius_norm()

        align = yTKy / (frobK * torch.norm(targets.get_flat_representation())**2)

        alignments.append(align.item())

    return alignments

def layer_alignment_matrix(model, output_fn, loader, n_output, centering=True):
    lc = LayerCollection.from_model(model)
    targets = torch.cat([args[1] for args in iter(loader)])
    means, mean_dists, cov_frobs, covs, ratios = [], [], [], [], []
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
        
        mean, cov_frob = [], []
        for i in range(10):
            L = list(targets == i)
            target_loader = extract_target_loader(loader, L, length = len(L), batch_size = len(L))
            generator = Jacobian(layer_collection=lc_this,
                             model=model,
                             loader=target_loader,
                             function=output_fn,
                             n_output=n_output,
                             centering=True)
            m = generator.get_jacobian()
            m = m.reshape([m.size(0),-1])
            mean.append(m.mean(dim = 0))
            cov = PMatDense(generator).frobenius_norm()
            cov_frob.append(cov)
            
            # if j == 0:
            #     dic[i] = h[j][l,:,:,:]
            # h = K[:,l,:].transpose(0,1)
            # h = h.reshape([h.size(0),-1])
            # # dic[i] = h.reshape([h.size(0),-1])
            # mean.append(h.mean(dim = 0))
            # # print(mean[0].shape)
            # h -= h.mean(dim = 0)
            # cov_frob.append(cov_frobenius(h))
        t = torch.trace(mean_distance(mean))
        b = sum(cov_frob)
        means.append(mean)
        mean_dists.append(t)
        cov_frobs.append(cov_frob)
        covs.append(b) #+ frobenius(mean_distance(mean).cpu())
        ratios.append(t/b)

        # extract_target_loader(baseloader, target, length, batch_size)

    
    return None, mean_dists, None, covs, ratios

def compute_trK(align_dl, model, output_fn, n_output):
    generator = Jacobian(model, align_dl, output_fn, n_output=n_output)
    F = PMatImplicit(generator)
    return F.trace().item() * len(align_dl)
