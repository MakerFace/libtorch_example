import torch
import torchvision

torch.manual_seed(0)

# model = torchvision.models.resnet101()

model = torch.load("models/model.pt")

tsm = torch.jit.script(model)
tsm.save("models/model_jit.pt")