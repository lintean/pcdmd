import torch.nn as nn
import torch
from dotmap import DotMap
from eutils.torch.container import AADModel


class Model(nn.Module):
    def __init__(self, local):
        super(Model, self).__init__()
        self.local = local
        self.conv_output_channel = 5
        self.conv_eeg_audio_number = 1
        self.output_fc_number = self.conv_eeg_audio_number * self.conv_output_channel

        self.conference_CNN = nn.Sequential(
            nn.Conv1d(local.data_meta.chan_num, self.conv_output_channel, 17),
            nn.ReLU()
        )
        self.pool1d = nn.AdaptiveAvgPool1d(1)

        self.output_fc = nn.Sequential(
            nn.Linear(self.output_fc_number, self.output_fc_number),
            nn.ReLU(),
            nn.Linear(self.output_fc_number, 2),
            nn.Sigmoid()
        )


    def forward(self, bs1, bs2, beeg, targets):
        visualization_weights = []
        data = torch.concat([bs1, beeg, bs2], dim=1)

        # CNN
        data = data.squeeze()
        data = self.conference_CNN(data)

        data = self.pool1d(data)
        data = data.squeeze()
        output = self.output_fc(data)

        return output, targets, visualization_weights


def get_model(args: DotMap) -> AADModel:
    model = Model(args)
    aad_model = AADModel(
        model=model,
        loss=nn.CrossEntropyLoss(),
        optim=torch.optim.Adam(model.parameters(), lr=args.lr),
        sched=None,
        dev=torch.device('cpu')
    )

    # scheduler = [torch.optim.lr_scheduler.ExponentialLR(optimzer[0], gamma=0.999), torch.optim.lr_scheduler.ExponentialLR(optimzer[1], gamma=0.999)]
    # device = torch.device('cuda:' + str(util.get_gpu_with_max_memory(device_to_use.gpu_list)))
    return aad_model

