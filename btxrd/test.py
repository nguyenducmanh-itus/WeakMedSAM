import torch
from torchvision.models import resnet18, resnet50
import pickle
path_model = r"C:/Users/ADMIN/OneDrive - VNU-HCMUS/CNTT-HK8/ResNet50.pt"
resnet = resnet50(weights="DEFAULT")

state_dict = torch.load(path_model)
new_state_dict = {}
for k, values in state_dict.items() : 
    k = k.replace("backbone.0", "conv1")
    k = k.replace("backbone.1", "bn1")
    #layer 1-4
    k = k.replace("backbone.4", "layer1")
    k = k.replace("backbone.5", "layer2")
    k = k.replace("backbone.6", "layer3")
    k = k.replace("backbone.7", "layer4")

    new_state_dict[k] = values
    
resnet.load_state_dict(new_state_dict, strict=False)
torch.save(new_state_dict, 'ResNet50_medical.pt')
state_dict = torch.load('ResNet50_medical.pt')
for k, v in state_dict.items() : 
    print(k)
#state_dict = torch.load(path_model, weights_only=True)
#resnet.load_state_dict(state_dict)


