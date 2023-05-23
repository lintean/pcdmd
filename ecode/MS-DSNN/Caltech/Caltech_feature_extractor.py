import torchvision
import torch
import _pickle as pickle
from Caltech_ChainDeepSpikeFeatureExtractor import *
import torchvision.transforms as transforms

batch_size = 1

names = 'spiking_model'
data_path = './raw/' #todo: input your data path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#在这个class里重写你的数据集
transform = transforms.Compose(               #归一化
    [
        transforms.Grayscale(),
        transforms.Resize ((40, 40)),
        transforms.ToTensor(),
    ])
class MyDataSet(torch.utils.data.Dataset):
    def __init__(self,set,transforms=None):
        with open("caltech_"+set+".pkl", 'rb') as file:
            data = pickle.loads(file.read())
        self.imgs=data[0]
        self.label = data[1]
        self.transforms = transforms

    def __getitem__(self, index):
        data = self.imgs[index]

        label = self.label[index]
        if self.transforms:
            data = self.transforms(data)
        return data,label

    def __len__(self):
        return len(self.imgs)
train_dataset =MyDataSet('train',transforms=transform)

test_dataset = MyDataSet('test',transforms=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

if __name__ == '__main__':

    extractor = SCNN_Extractor_V1()
    extractor.train(train_loader)
    #extractor.extractor(train_loader,test_loader)

# extractor = SCNN_Extractor_V2()
# extractor.train(train_loader)
# extractor.extractor(train_loader,test_loader)
