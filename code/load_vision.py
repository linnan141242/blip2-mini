import torch
from transformers import CLIPVisionModel
#加载视觉模型
vision_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
for param in vision_model.parameters():
    param.requires_grad = False
print("视觉模型加载并冻结成功")