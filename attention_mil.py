import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torch.utils.checkpoint import checkpoint
class ViTAttentionMIL(nn.Module):
    def __init__(self, num_classes=1, num_frozen_blocks=10):
        super(ViTAttentionMIL, self).__init__()
        
        self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        self.vit.heads = nn.Identity() 
        feature_dim = 768 
        
        if num_frozen_blocks > 0:
            for param in self.vit.conv_proj.parameters():
                param.requires_grad = False
            self.vit.class_token.requires_grad = False
            self.vit.encoder.pos_embedding.requires_grad = False
            
            for i in range(min(num_frozen_blocks, len(self.vit.encoder.layers))):
                for param in self.vit.encoder.layers[i].parameters():
                    param.requires_grad = False
        self.attention_V = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.Sigmoid() 
        )
        self.attention_weights = nn.Linear(256, 1)
        
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, patches, chunk_size=16):
        features = []
        for i in range(0, patches.size(0), chunk_size):
            chunk = patches[i : i + chunk_size]
            if self.training:
                chunk.requires_grad_()
                def custom_forward(x):
                    return self.vit(x)
                
                feat = checkpoint(custom_forward, chunk, use_reentrant=False)
            else:
                feat = self.vit(chunk)
            features.append(feat)
        h = torch.cat(features, dim=0) 
        A_V = self.attention_V(h)  
        A_U = self.attention_U(h)  
        A = self.attention_weights(A_V * A_U) 
        A = torch.transpose(A, 1, 0)  
        A = F.softmax(A, dim=1)  
        z = torch.mm(A, h)  
        logits = self.classifier(z) 
        
        return logits, A
