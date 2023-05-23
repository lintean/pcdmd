import torchvision
import torch
import _pickle as pkl
from TIDIGITS_ParallelDeepSpikeFeatureExtractor import *
import torchvision.transforms as transforms
# from random import randrange
# import PIL.Image as Image
def write_pkl(path,data):
  with open(path,'wb') as f:
      pkl.dump(data,f)
      f.flush()
      f.close()

def read(path):
    with open(path, 'rb') as f:
        s= pkl.load(f)
        f.close()
    return s

# data_root_path= 'E:\spiking_data\ETH-80'
# train_x = []
# train_y = []
# test_x = []
# test_y = []
# for label in os.listdir(data_root_path):
#     x = []
#     y = []
#     class_path = os.path.join(data_root_path,label)
#     for viewpoint in os.listdir(class_path):
#         if os.path.isfile(os.path.join(class_path,viewpoint)):
#             continue
#         img_list=os.listdir(os.path.join(class_path,viewpoint))
#         id = img_list.index('maps')
#         img_list.pop(id)
#         random_index = randrange (0, len (img_list))
#         x.append(Image.open(os.path.join(os.path.join(class_path,viewpoint),img_list[random_index])))
#         y.append(int(label))
#     train_x+=x[:5]
#     train_y+=y[:5]
#     test_x += x[5:]
#     test_y += y[5:]
# write_pkl('eth80_train.pkl',[train_x,train_y])
# write_pkl('eth80_test.pkl',[test_x,test_y])
class MyDataset(torch.utils.data.Dataset):  # 创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset
    def __init__(self, datas):  # 初始化一些需要传入的参数
        super(MyDataset, self).__init__()  # 对继承自父类的属性进行初始化
        x,y = datas[0],datas[1]
        imgs = []
        for i in range(len(x)):  # 迭代该列表#按行循环txt文本中的内
            imgs.append(( x[i] , y[i]))  # 把txt里的内容读入imgs列表保存，具体是words几要看txt内容而定
            # 很显然，根据我刚才截图所示txt的内容，words[0]是图片信息，words[1]是lable
        self.imgs = imgs
        # self.transform = transform


    def __getitem__(self, index):  # 这个方法是必须要有的，用于按照索引读取每个元素的具体内容
        img, label = self.imgs[index]  # fn是图片path #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
        # img = self.loader(fn)  # 按照路径读取图片
        # if self.transform is not None:
        #     img = self.transform(img)  # 数据标签转换为Tensor
        return img, label  # return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容

    def __len__(self):  # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.imgs)
[train_x,train_y]=read('tidigit_CQT_train.pkl')
train_x = torch.tensor(train_x).float().unsqueeze(1)
train_y = torch.tensor(train_y).long()

[test_x,test_y]=read('tidigit_CQT_test.pkl')
test_x = torch.tensor(test_x).float().unsqueeze(1)
test_y = torch.tensor(test_y).long()
batch_size = 1
names = 'spiking_model'
data_path = './raw/' #todo: input your data path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#在这个class里重写你的数据集

train_loader = torch.utils.data.DataLoader(MyDataset([train_x,train_y]), batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(MyDataset([test_x,test_y]), batch_size=batch_size, shuffle=False, num_workers=0)

if __name__ == '__main__':

    extractor = SCNN_Extractor_V1()
    extractor.train(train_loader)
    #extractor.extractor(train_loader,test_loader)

# extractor = SCNN_Extractor_V2()
# extractor.train(train_loader)
# extractor.extractor(train_loader,test_loader)
