import torchvision
import torch
from MNIST_ParallelDeepSpikeFeatureExtractor import *
import torchvision.transforms as transforms

batch_size = 1

names = 'spiking_model'
data_path = './raw/' #todo: input your data path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataset = torchvision.datasets.MNIST(root= data_path, train=True, download=True, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

test_set = torchvision.datasets.MNIST(root= data_path, train=False, download=True,  transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

if __name__ == '__main__':

    extractor = SCNN_Extractor_V1()
    extractor.train(train_loader)
    #extractor.extractor(train_loader,test_loader)

#extractor = SCNN_Extractor_V2()
#extractor.train(train_loader)
#extractor.extractor(train_loader,test_loader)
