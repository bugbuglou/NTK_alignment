!pip install nngeometry
from nngeometry.object import PVector
from nngeometry.object.fspace import FMatDense
from nngeometry.object.pspace import PMatDense
from nngeometry.object.vector import FVector
from nngeometry.object import PMatImplicit

from nngeometry.generator import Jacobian
from nngeometry.layercollection import LayerCollection

def output_fn(x,t):
    # define this function first whenever you have a NN model you want to calculate 
    # alignment, and input this function as an argument to alignment functions
    return model(x) 
  
def alignment(model, output_fn, loader, n_output, centering=True):
    # model: NN model
    # output_fn: defined above
    # loader: dataloader, recommended total datapoints 200-2000
    # n_output: number of output channels of the model (i.e. for cifar10 cross entropy: 10)
    # centering: if you want the kernel to be centered ot not
    
    # this function calculates alignment for entire model parameters
    # return: single value alignment
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

    return align.item()

  
def layer_alignment(model, output_fn, loader, n_output, centering=True):
    # model: NN model
    # output_fn: defined above
    # loader: dataloader, recommended total datapoints 200-2000
    # n_output: number of output channels of the model (i.e. for cifar10 cross entropy: 10)
    # centering: if you want the kernel to be centered ot not
    
    # this function calculates alignment for every layer parameters
    # return: a list of alignments for each layer
    model.eval()
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
        frobK = K_dense.frobenius_norm()
        align = yTKy / (frobK * torch.norm(targets.get_flat_representation())**2)
        
        alignments.append(align.item())
      

    model.train()
    return alignments
  
