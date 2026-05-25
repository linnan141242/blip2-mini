import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from tqdm import tqdm
from torchvision import transforms
from PIL import Image

# 导入 vision_model 和 llm、tokenizer (来自您已完成的模块)
from load_vision import vision_model
from load_llm import llm, tokenizer
from qformer import QFormerWithProjection


# ================= 数据准备 ===================

class FLICKRDataset(Dataset):
    """自定义数据集，用于加载 JSON 文件中的图片路径和描述。"""

    def __init__(self, json_file, transform):
        super().__init__()
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.img_paths = list(self.data.keys())  # 图片路径
        self.captions = list(self.data.values())  # 图片的描述列表
        self.transform = transform  # 图像预处理方法

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        captions = self.captions[index]  # 获取图片的所有描述 (可能有多个)

        # 随机选一条描述（训练时常见做法）
        caption = captions[0]

        # 拼接图像路径
        img_path = os.path.join("my_mini_data/images", img_path)
        image = Image.open(img_path).convert("RGB")

        # 应用预处理 transform
        image = self.transform(image)  # 转换为 [3, height, width] 张量
        print(f"[DEBUG] Image shape after transform: {image.shape}")  # 调试信息

        # 返回图像张量和对应的文本描述
        return image, caption


# ================= 模型组合 ===================

class MiniBLIP2(nn.Module):
    """将 vision_model, Q-Former 和 LLM 组合为一个端到端模型，并支持训练 Q-Former。"""

    def __init__(self, vision_model, qformer, llm):
        super(MiniBLIP2, self).__init__()
        self.vision_model = vision_model
        self.qformer = qformer
        self.llm = llm

        # 冻结 vision_model 和 llm
        for param in self.vision_model.parameters():
            param.requires_grad = False
        for param in self.llm.parameters():
            param.requires_grad = False

    def forward(self, images, input_ids):
        visual_features = self.vision_model(images).last_hidden_state  # [B, 50, 768]
        query_features = self.qformer(visual_features)  # [B, 32, 768]
        text_embeddings = self.llm.get_input_embeddings()(input_ids)  # [B, L, 768]
        combined_inputs = torch.cat([query_features, text_embeddings], dim=1)  # [B, 32+L, 768]

        # 手动跑 OPT 的前向传播，不传 labels
        outputs = self.llm(inputs_embeds=combined_inputs)
        logits = outputs.logits  # [B, 32+L, vocab_size]

        # 只对文本部分计算 loss，忽略视觉前缀
        text_len = input_ids.size(1)
        text_logits = logits[:, -text_len:, :]  # 取最后 text_len 个位置的 logits

        # 计算交叉熵
        loss = torch.nn.functional.cross_entropy(
            text_logits.reshape(-1, text_logits.size(-1)),
            input_ids.reshape(-1),
            ignore_index=tokenizer.pad_token_id
        )

        return type('obj', (object,), {'loss': loss})()


# ================= 训练函数定义 ===================

def train_qformer():
    # 设置超参数
    num_epochs = 3
    batch_size = 8
    learning_rate = 1e-4

    # 定义 Q-Former
    qformer = QFormerWithProjection(hidden_dim=768, output_dim=768, num_queries=32, num_heads=8)

    # 定义图像预处理 (用于 CLIPVisionModel)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整为 CLIPVisionModel 所需的输入尺寸
        transforms.ToTensor(),  # 转为张量并归一化至 [0, 1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载数据
    dataset = FLICKRDataset("my_mini_data/captions.json", transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 创建模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MiniBLIP2(vision_model, qformer, llm).to(device)
    model = model.float()
    # 优化器（仅优化 Q-Former 的参数）
    optimizer = AdamW(model.qformer.parameters(), lr=learning_rate)

    # 训练循环
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for batch in pbar:
            images, captions = batch

            # 图像和文本输入
            images = images.to(device)  # 确保图像张量的大小为 [batch_size, 3, height, width]
            print(f"[DEBUG] Images shape: {images.shape}")  # 调试信息

            input_ids = tokenizer(captions, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
            print(f"[DEBUG] Input IDs shape: {input_ids.shape}")  # 打印input_ids的维度

            # 清零梯度
            optimizer.zero_grad()

            # 前向传播
            images = images.float()
            outputs = model(images, input_ids)
            loss = outputs.loss  # LLM 的输出中包含 loss

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            # 更新进度条
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        print(f"Epoch {epoch + 1} - Loss: {epoch_loss / len(dataloader):.4f}")

    # 保存训练好的 Q-Former
    torch.save(model.qformer.state_dict(), "qformer.pth")
    print("Q-Former 训练完成并已保存至 qformer.pth")


if __name__ == "__main__":
    train_qformer()