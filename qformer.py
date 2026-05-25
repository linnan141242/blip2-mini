import torch
from torch import nn

class QFormerWithProjection(nn.Module):
    def __init__(self,
                 hidden_dim: int = 768,  # CLIP 的输出维度 (hidden_dim)
                 output_dim: int = 768,  # 投影到语言模型的维度 (如 OPT, 768)
                 num_queries: int = 32,  # Query Embeddings 的数量
                 num_heads: int = 8,  # Cross-Attention 的多头注意力头数
                 dropout: float = 0.1):  # Dropout 比例
        """
        基于 CLIPVisionModel 的 QFormer 架构，支持全局特征和特征序列:
        - 输入 shape: [batch_size, num_patches+1, hidden_dim]
        - 输出 shape: [num_queries, batch_size, output_dim]
        """
        super(QFormerWithProjection, self).__init__()

        # 初始化 Query Embeddings
        self.query_embeddings = nn.Parameter(torch.randn(num_queries, hidden_dim))

        # Cross-Attention 层
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim,
                                                     num_heads=num_heads,
                                                     dropout=dropout)

        # 投影层
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, visual_features):
        """
        输入：
        - visual_features: [batch_size, num_patches, hidden_dim]
        """
        # 调整 visual_features 到 [num_patches, batch_size, hidden_dim]
        visual_features = visual_features.permute(1, 0, 2)  # [num_patches, batch_size, hidden_dim]

        # 获取 query_embedding，增加一个 batch 维度
        query = self.query_embeddings.unsqueeze(1).repeat(1, visual_features.size(1),
                                                          1)  # [num_queries, batch_size, hidden_dim]

        # 交叉注意力操作
        query, _ = self.cross_attention(query, visual_features,
                                        visual_features)  # 输入维度必须都为 [sequence_length, batch_size, embed_dim]

        # 输出投影
        query = self.output_proj(query)  # [num_queries, batch_size, hidden_dim]
        return query.permute(1, 0, 2)  # 转换回 [batch_size, num_queries, hidden_dim]