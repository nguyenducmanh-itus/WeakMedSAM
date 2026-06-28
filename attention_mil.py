import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights

class AttentionMIL(nn.Module) :
    def __init__(self, num_classes=1, num_frozen_blocks=10) :
        super(AttentionMIL, self).__init__()
        self.vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        self.vit.heads = nn.Identity()
        feature_dim = 768
        
        if num_frozen_blocks > 0 :
            for param in self.vit.conv_proj.parameters() :
                param.requires_grad = False
            self.vit.class_token.requires_grad = False
            self.vit.encoder.pos_embedding.requires_grad = False
            for i in range(min(num_frozen_blocks, len(self.vit.encoder.layers))) :
                for param in self.vit.encoder.layers[i].parameters() :
                    param.requires_grad = False
        
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, 256), 
            nn.Tanh(), 
            nn.Linear(256, 1)
        )
        self.classifier = nn.Linear(feature_dim, num_classes)        

    def forward(self, patches, chunk_size = 16) :
        features = []
        for i in range(0, patches.size(0), chunk_size):
            chunk = patches[i : i + chunk_size]
            feat = self.vit(chunk) # shape: [chunk_size, 768]
            features.append(feat)
            
        # Concanate chunk
        h = torch.cat(features, dim=0)
        A = self.attention(h)  #Calculate score for each patchs
        A = torch.transpose(A, 1, 0)  # Transpose from shape [N, 1] to [1, N]
        A = F.softmax(A, dim=1)  #Translate logit to probability
        #Sum each vector of patch 
        z = torch.mm(A, h)  # [1, N] x [N, 768] = [1, 768]
        # Classify 
        logits = self.classifier(z) 
        return logits, A
