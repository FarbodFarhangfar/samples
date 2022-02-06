import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F



def preprocess():
    class Mydataset(torch.utils.data.Dataset):
        def __init__(self, data, transform=None):
            self.x = data
            self.len = len(data)
            # self.transform = transform

        def __getitem__(self, index):
            sample = self.x[index]
            if self.transform:
                sample = self.transform(sample)

            return sample

        def __len__(self):
            return self.len

    class feature_extraction(nn.Module):
        def __init__(self, model1, model2, model3):
            super().__init__()
            self.model1 = model1
            self.model2 = model2
            self.model3 = model3
            self.avaragepooling = nn.AvgPool1d(3)
            self.fully_conected = nn.Linear(1000, 1000)

        def forward(self, x):
            x1 = self.model1(x)
            x2 = self.model2(x)
            x3 = self.model3(x)
            x = torch.concat((x1, x2, x3), dim=1)

            x = self.avaragepooling(x)
            x = F.softmax(self.fully_conected(x), dim=1)

            return x

    resnet = torchvision.models.resnet152(pretrained=True)
    efficientnet = torchvision.models.efficientnet_b7(pretrained=True)
    regnet = torchvision.models.regnet_y_32gf(pretrained=True)

    model = feature_extraction(resnet, efficientnet, regnet)
    model.cuda()