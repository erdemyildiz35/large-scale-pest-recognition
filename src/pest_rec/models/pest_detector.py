import torch
import torch.nn as nn
import torchvision.models as models

class PestDetector(nn.Module):
    def __init__(self, num_classes, backbone="resnet50", pretrained=True):
        super().__init__()
        self.backbone_name = backbone
        
        # Backbone seçimi
        if backbone == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif backbone == "mobilenet_v3":
            self.backbone = models.mobilenet_v3_large(pretrained=pretrained)
            feature_dim = self.backbone.classifier[3].in_features
            self.backbone.classifier = nn.Identity()
            
        # Sınıflandırıcı
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(feature_dim, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features) 