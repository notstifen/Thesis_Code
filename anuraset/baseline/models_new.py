from torch import nn
from torchvision import models as models
from torchvision.models import mobilenet_v3_large

from torchsummary import summary 

# keeping old code for reference (and so that nothing in the main script breaks again)
'''
class CNNetwork_2D(nn.Module):
    
    def __init__(self,multi_label=False):
        super().__init__()
            
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(6400, 42)
        if multi_label:
            self.loss_func = nn.BCEWithLogitsLoss()
        else:
            self.loss_func = nn.CrossEntropyLoss()

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        logits = self.linear(x)
        predictions = self.loss_func(logits)
        return predictions 

def model(pretrained, requires_grad):
    model = models.resnet50(progress=True, pretrained=pretrained)
    # to freeze the hidden layers
    if requires_grad == False:
        for param in model.parameters():
            param.requires_grad = False
    # to train the hidden layers
    elif requires_grad == True:
        for param in model.parameters():
            param.requires_grad = True
    # make the classification layer learnable
    # we have 42 classes in total
    model.fc = nn.Linear(2048, 42)
    return model
'''
'''
class ResNetClassifier(nn.Module):
    def __init__(self, model_type):
        super().__init__()

        if model_type=='resnet50':
            self.resnet = models.resnet50(pretrained=True)
        elif model_type=='resnet152':
            self.resnet = models.resnet152(pretrained=True)
        elif model_type=='resnet18':
            self.resnet = models.resnet18(pretrained=True)
        else:
            assert False

        self.linear = nn.Linear(in_features=1000, out_features=42)

    def forward(self, x, y=None):
        x = x.repeat(1, 3, 1, 1)    # (B, 1, F, L) -> (B, 3, F, L)

        x = self.resnet(x)
        predictions = self.linear(x)

        return predictions
'''

class ImageClassifier(nn.Module):
    def __init__(self, model_type):
        super().__init__()

        #
        if model_type == 'mobilenet_v3_small':
            self.mobilenet = models.mobilenet_v3_small(pretrained=True) #change the pretrained soom
            out_features = self.mobilenet.classifier[3].out_features  
        elif model_type == 'mobilenet_v3_large':
            self.mobilenet = models.mobilenet_v3_large(pretrained=True)
            out_features = self.mobilenet.classifier[3].out_features  
        elif model_type == 'mobilenet_v2':
            self.mobilenet = models.mobilenet_v2(pretrained=True)
            out_features = self.mobilenet.classifier[1].out_features  
        else:
            raise ValueError("Nope, not working agian")

        #
        self.linear = nn.Linear(in_features=out_features, out_features=42)

    def forward(self, x, y=None):
        x = x.repeat(1, 3, 1, 1)  

        # again and again and again etc..
        x = self.mobilenet(x)

        
        predictions = self.linear(x)

        return predictions
    
'''
class MobileNetV3LargeClassifier(nn.Module):
    def __init__(self, num_classes=42, pretrained=True):
    
        super().__init__()
       
        self.mobilenetv3_large = mobilenet_v3_large(pretrained=pretrained)
        
        self.mobilenetv3_large.classifier = nn.Linear(self.mobilenetv3_large.classifier[-1].in_features, num_classes)

    def forward(self, x):
        x = self.mobilenetv3_large(x)
        return x
'''
