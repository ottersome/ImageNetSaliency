import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.utils.prune as prune
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torch.autograd import Variable
import os
from PIL import Image

### Constants ###
testSetLocation = './TestSet/ILSVRC/Data/DET/test'
amntOfImages = 3

#if not torch.cuda.is_available():
#    print("This will be slow")
#else:


### Get ImageNet ###
resnet18 = models.resnet18(pretrained=True)
resnet18.eval()
### Get Data ###
images = []
preprocessor = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])
# Browse for Images
imgsLoaded = 0
for subdir, dirs, files in os.walk(testSetLocation):
    for file in files:
        print(os.path.join(subdir,file))
        imgsLoaded += 1
        PILImage = Image.open(os.path.join(subdir,file))
        images.append(preprocessor(PILImage))
        if imgsLoaded >= amntOfImages:
            break

inputBatch = images[0].unsqueeze(0)
if torch.cuda.is_available():
    inputBatch = inputBatch.to('cuda')
    resnet18.to('cuda')

with torch.no_grad():
    output=resnet18(inputBatch)
score,indices = torch.max(output,1)
#result = resnet18(imgbatch)
print('Result : '+str(indices.cpu().data))
plt.imshow(np.moveaxis(np.asarray(images[0].cpu()),0,-1))
plt.show()
print('Score of index : '+str(score))