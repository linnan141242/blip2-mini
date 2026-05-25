import os
import json
from PIL import Image
import torch
from torchvision import transforms
from load_vision import vision_model  # 加载视觉模型
from load_llm import llm, tokenizer # 加载语言模型及其对应的 tokenizer
# 强制统一所有模型参数为 float32
for model in [vision_model, llm]:
    for param in model.parameters():
        param.data = param.data.float()
from qformer import QFormerWithProjection  # 导入 Q-Former 模型

# 设置运行设备：在这个案例中我们使用 CPU
device = torch.device("cpu")

# 加载图片的预处理方法
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图片大小到 224x224
    transforms.ToTensor(),  # 转换为 Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 使用 ImageNet 的均值和标准差进行归一化
])

# 定义加载图片的方法
def load_images(image_paths, transform):
    images = []
    for path in image_paths:
        image = Image.open(path).convert("RGB")  # 确保图片是 RGB 格式
        image = transform(image)
        images.append(image)
    return torch.stack(images, dim=0)  # 将所有图片堆叠为一个批次的张量

# 定义生成描述的方法
def generate_description(image_tensor, vision_model, qformer, llm, tokenizer):
    vision_model.eval()
    qformer.eval()
    llm.eval()

    with torch.no_grad():
        vision_outputs = vision_model(image_tensor.unsqueeze(0).to(device))
        visual_features = vision_outputs.last_hidden_state.float()
        query_features = qformer(visual_features).float()

        input_text = "Describe the image: "
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
        text_emb = llm.get_input_embeddings()(input_ids).float()

        inputs_embeds = torch.cat([query_features, text_emb], dim=1)
        outputs = llm.generate(
            inputs_embeds=inputs_embeds,
            max_length=50,
            do_sample=True,
            top_p=0.9,
            temperature=1.5,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_text

# 主函数
if __name__ == "__main__":
    # 加载训练好的 Q-Former 权重
    qformer = QFormerWithProjection().to(device).float()  # 加载模型到设备
    qformer.load_state_dict(torch.load("qformer.pth", map_location=device))  # 加载权重

    # 加载数据（读取 captions.json 文件）
    with open("my_mini_data/captions.json", "r", encoding="utf-8") as f:
        captions_data = json.load(f)  # 加载 JSON 文件

    # 获取前 5 张图片的路径和真实描述
    image_dir = "my_mini_data/images/"  # 图片文件夹路径
    images = []
    real_captions = []
    img_names = list(captions_data.keys())[:5]
    for img_name in img_names:
        image_path = os.path.join(image_dir, img_name)
        if os.path.exists(image_path):
            images.append(image_path)
            real_captions.append(captions_data[img_name][0])

    # 对每张图片生成描述并对比
    for i, image_path in enumerate(images):
        print(f"\n处理第 {i + 1} 张图片：{image_path}")
        # 加载图片并预处理
        image_tensor = load_images([image_path], image_transform).to(device)  # 读取并预处理图片

        # 使用模型生成描述
        generated_caption = generate_description(image_tensor[0], vision_model, qformer, llm, tokenizer)

        # 打印真实描述和生成描述
        print(f"真实描述：{real_captions[i]}")