from torch import nn
from torchvision import models as models
from torchvision.models import mobilenet_v3_large, mobilenet_v3_small, mobilenet_v2

from torchsummary import summary 

class ResNetClassifier(nn.Module):
    def __init__(self, model_type):
        super().__init__()

        resnet_features = 1000

        if model_type == 'resnet50':
            self.resnet = models.resnet50(pretrained=True)
            in_features = resnet_features

        elif model_type == 'resnet152':
            self.resnet = models.resnet152(pretrained=True)
            in_features = resnet_features

        elif model_type == 'resnet18':
            self.resnet = models.resnet18(pretrained=True)
            in_features = resnet_features

            
        ### check the output layer features nr.

        elif model_type == 'mobilenet_v3_large':
            self.resnet = models.mobilenet_v3_large(pretrained=True)
            in_features = 1000

        elif model_type == 'mobilenet_v3_small':
            self.resnet = models.mobilenet_v3_small(pretrained=True)
            in_features = 1000

        elif model_type == 'mobilenet_v2':
            self.resnet = models.mobilenet_v2(pretrained=True)
            in_features = 1000
        else:
            raise ValueError("Unsupported model type: {}".format(model_type))

        self.linear = nn.Linear(in_features=in_features, out_features=42)

    def forward(self, x, y=None):
        x = x.repeat(1, 3, 1, 1)    # (B, 1, F, L) -> (B, 3, F, L)

        x = self.resnet(x)
        predictions = self.linear(x)

        return predictions
    
'''
class MobileNetV3LargeClassifier(nn.Module):
    def __init__(self, num_classes=42, pretrained=True):
        super().__init__()
        # Load the pre-trained MobileNetV3 Large model
        self.mobilenetv3_large = mobilenet_v3_large(pretrained=True)
        # Modify the classifier's output to match the number of classes
        self.mobilenetv3_large.classifier = nn.Linear(self.mobilenetv3_large.classifier[-1].in_features, num_classes)

    def forward(self, x):
        x = self.mobilenetv3_large(x)
        return x
'''
