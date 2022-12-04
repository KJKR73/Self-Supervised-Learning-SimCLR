import torch
import torchvision.models as models

class MODEL_ZOO(torch.nn.Module):
    '''Custom zoo model for training simclr'''
    def __init__(self, config):
        super(MODEL_ZOO, self).__init__()
        # Define the architecure dict
        arch_dict = {
            "resnet-18" : models.resnet18(pretrained=False, num_classes=config.EMBED_SIZE),
            "resnet-34" : models.resnet34(pretrained=False, num_classes=config.EMBED_SIZE),
            "resnet-50" : models.resnet50(pretrained=False, num_classes=config.EMBED_SIZE)
        }
        
        # Make the model and collect in_features
        self.backbone = arch_dict[config.ARCH]
        prev_features = self.backbone.fc.in_features
        
        # Define the new layer
        self.backbone.fc = torch.nn.Sequential(torch.nn.Linear(prev_features, prev_features),
                                               torch.nn.ReLU(),
                                               self.backbone.fc)
        
    def forward(self, images):
        return self.backbone(images)