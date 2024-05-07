#%%
import torch
state_dict = torch.load('experiments/pretrained_models/Real_ESRGAN/RealESRGAN_x2plus.pth')
print(state_dict.keys())