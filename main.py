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
amntOfImages = 10

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

def extractVOG(net,testSet,device):
    # Lets do the VOG thing here
    print("Evaluating VOG")
    counter = 0
    net.eval()  # Turn training
    grads = []
    input = testSet
    input.requires_grad=True
    #logits = net(images)
    preds = net(input)
    #layer_softMax = torch.nn.Softmax(dim=1)(logits)
    score, indices = torch.max(preds,1)
    # the tensor.type below casts the tensor into a LongTensor type(rather than float i think)
    #sel_nodes = layer_softMax[torch,torch.arange(len(labels),labels.type(torch.LongTensor))]# I dont understand much why this is happening
    #        layer_softMax.backward(ones)
    score.backward()
    grads.append(input.grad.cpu().data.numpy())
    counter += 1

    print("Finished VOG Evaluation")
    return grads

inputBatch = images[4].unsqueeze(0)
if torch.cuda.is_available():
    inputBatch = inputBatch.to('cuda')
    resnet18.to('cuda')

with torch.no_grad():
    output=resnet18(inputBatch)
score,indices = torch.max(output,1)
#result = resnet18(imgbatch)
print('Result : '+str(indices.cpu().data))

plt.subplot(121,title="Original Image")
plt.imshow(np.moveaxis(np.asarray(images[4].cpu()),0,-1))
print('Score of index : '+str(score))
vog = extractVOG(resnet18,inputBatch,'cuda')
plt.subplot(122,title="Salient Map of Image")
vogImage = np.moveaxis(vog[0].squeeze(),0,-1)
#vogLogTransformed =1*np.log(1 + ((vogImage+1)/2))
vogGammaTransformed = np.power(((vogImage+1)/2),2.8)
plt.imshow(vogGammaTransformed)
plt.show()

