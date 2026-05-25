\# Mini-BLIP2 图像描述生成复现实验报告



\## 1. 论文信息



\- 论文名称：BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models

\- 论文地址：https://arxiv.org/abs/2301.12597



\## 2. 任务说明



本实验复现的任务是图像描述生成 Image Captioning。



输入：图片  

输出：英文 caption



\## 3. 数据集



\- 数据集名称：Flickr8k

\- 数据集地址：https://www.kaggle.com/datasets/adityajn105/flickr8k

\- 实际使用数据量：前 200 张图片



\## 4. 模型结构



Image → Frozen Vision Encoder → Mini Q-Former → Projection Layer → Frozen Language Decoder → Caption



\### 4.1 Vision Encoder



`openai/clip-vit-base-patch32`



\### 4.2 Mini Q-Former



\- query token 数量：32

\- hidden size：768

\- Transformer 层数：1 层（只包含交叉注意力，无自注意力）

\- 是否使用 cross-attention：是



\### 4.3 Language Decoder



`facebook/opt-125m`



\## 5. 训练设置



\- 训练数据量：200 张图片

\- epoch：3

\- batch size：8

\- learning rate：1e-4

\- optimizer：AdamW

\- loss function：Cross Entropy Loss

\- 冻结的模块：Vision Encoder、Language Decoder

\- 训练的模块：Mini Q-Former



\## 6. 训练过程



| Epoch | Train Loss |

|---:|---:|

| 1 | 1.9075 |

| 2 | 0.3207 |

| 3 | 0.1513 |



Loss 下降约 92%。



\## 7. 生成结果展示



| 图片编号 | 真实 Caption | 模型生成 Caption |

|---|---|---|

| 1 | A child in a pink dress is climbing up a set of stairs in an entry way . | （空） |

| 2 | A black dog and a spotted dog are fighting | （空） |

| 3 | A little girl covered in paint sits in front of a painted rainbow with her hands in a bowl . | （空） |

| 4 | A man lays on a bench while his dog sits by him . | （空） |

| 5 | A man in an orange hat starring at something . | （空） |



生成描述为空，原因是训练数据仅 200 张、训练 3 轮，Q-Former 从随机初始化开始，尚未收敛。流程已完整跑通。



\## 8. 总结



\- 成功跑通训练，Loss 从 1.91 降至 0.15

\- 生成效果受限于数据量和训练轮数，输出为空

\- 遇到约 15 个 bug，包括 AdamW 导入、CLIPProcessor 不兼容、维度拼接、dtype 错误等，均已修复

\- 如果继续改进，可以增加训练数据量、增加训练轮数、调整学习率策略



\## 9. AI 对话过程记录



\- 录制工具：内容直接分享

\- 使用的 AI 模型：ChatGPT

\- 累计对话时长：约 5 天，多次会话



AI 在以下环节提供了帮助：

\- 解释 Q-Former 架构和 BLIP-2 原理

\- 排查训练脚本中的维度拼接、dtype 等报错

\- 提供代码修改建议



自己在以下环节独立完成或推翻了 AI 的建议：

\- 坚持 Q-Former 只保留三块核心组件（查询向量、交叉注意力、投影层）

\- 修正 AI 给出的默认参数（num\_queries=4→32, num\_heads=4→8）

\- 修正 AI 代码中的 CLIPProcessor 错误，换用 transforms

\- 多次降级/升级 transformers 库排查兼容性 bug



\## 10. Git 提交记录



\- 仓库地址：https://github.com/linnan141242/blip2-mini

\- 总 commit 数：5

见下方



