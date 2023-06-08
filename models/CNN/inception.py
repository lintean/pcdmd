import torch.nn as nn
import torch
from dotmap import DotMap
from eutils.torch.container import AADModel
from eutils.util import get_gpu_with_max_memory


class Model(nn.Module):
    def __init__(self, local):
        super(Model, self).__init__()
        self.local = local
        self.conv_output_channel = 10

        self.bn = nn.BatchNorm2d(1)

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 10, (1, 64), stride=(1, 64)),
            nn.ReLU(),
            nn.MaxPool2d((16, 1))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(1, 10, (1, 8), stride=(1, 8)),
            nn.ReLU(),
            nn.MaxPool2d((16, 1)),
            nn.Conv2d(10, 10, (1, 8), stride=(1, 8)),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(1, 10, (1, 4), stride=(1, 4)),
            nn.ReLU(),
            nn.MaxPool2d((16, 1)),
            nn.Conv2d(10, 10, (1, 16), stride=(1, 16)),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(1, 10, (1, 16), stride=(1, 16)),
            nn.ReLU(),
            nn.MaxPool2d((16, 1)),
            nn.Conv2d(10, 10, (1, 4), stride=(1, 4)),
            nn.ReLU(),
        )

        self.output_fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(320, self.conv_output_channel),
            nn.Sigmoid(),
            nn.Dropout(0.5),
            nn.Linear(self.conv_output_channel, 2),
            nn.Softmax()
        )


    def forward(self, bs1, bs2, beeg, targets):
        visualization_weights = []

        # CNN
        eeg = beeg.squeeze()
        eeg = eeg[:, None, ...]
        eeg = eeg.transpose(-1, -2)
        eeg = self.bn(eeg)

        temp1 = self.conv1(eeg)
        temp2 = self.conv2(eeg)
        temp3 = self.conv3(eeg)
        temp4 = self.conv4(eeg)

        temp = torch.concat([temp1, temp2, temp3, temp4], dim=1)
        temp = temp.view(eeg.shape[0], -1)
        output = self.output_fc(temp)

        return output, targets, visualization_weights


def get_model(args: DotMap) -> AADModel:
    model = Model(args)
    aad_model = AADModel(
        model=model,
        loss=nn.CrossEntropyLoss(),
        optim=torch.optim.Adam(model.parameters(), lr=args.lr),
        sched=None,
        dev=torch.device(get_gpu_with_max_memory(args.gpu_list))
    )

    # scheduler = [torch.optim.lr_scheduler.ExponentialLR(optimzer[0], gamma=0.999), torch.optim.lr_scheduler.ExponentialLR(optimzer[1], gamma=0.999)]
    # device = torch.device('cuda:' + str(util.get_gpu_with_max_memory(device_to_use.gpu_list)))
    return aad_model

