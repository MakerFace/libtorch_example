import torch
import torchvision

torch.manual_seed(0)

model = torchvision.models.resnet101()
in_data = torch.ones(1, 3, 224, 224)

model.eval()
output = model(in_data)

print(output[0, :5].detach().numpy())

# traced_script_module = torch.jit.script(model)
# traced_script_module.save("models/test_model.pt")
torch.save(model, "models/test_model.pt")