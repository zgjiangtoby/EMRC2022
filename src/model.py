import torch
import torch.nn as nn
import torch.nn.functional as F
from configs.config import Config


class QA_1(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.multihead = nn.MultiheadAttention(embed_dim=config.embed_dim,
                                               num_heads=config.heads,
                                               dropout=config.drop_rate,
                                               batch_first=True)

        self.ln = nn.LayerNorm(config.hidden_dim)
        self.proj = nn.Linear(config.embed_dim, config.hidden_dim)
        self.start = nn.Linear(config.embed_dim, 1, bias=False)
        self.end = nn.Linear(config.embed_dim, 1, bias=False)
        self.drop_out = nn.Dropout(config.drop_rate)

    def forward(self, c, q):
        c = c.squeeze()
        q = q.squeeze()
        q_att_out, _ = self.multihead(q, c, c)

        proj_q = self.proj(q_att_out)
        c_att_out, _ = self.multihead(c, q, q)
        proj_c = self.proj(c_att_out)

        concat = torch.cat((proj_c, proj_q), dim=2)

        start = self.start(concat).squeeze()
        end = self.end(concat).squeeze()

        start_evi = self.start(concat).squeeze()
        end_evi = self.start(concat).squeeze()
        # return torch.argmax(start,dim=1), torch.argmax(end,dim=1)
        return start, end, start_evi, end_evi


# fgm
class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=0.3, emb_name='robertaembeddings.word_embeddings_layer.weight'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                print('fgm attack')
                # 这里加入fgm attack来判断是否进行对抗训练了
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='robertaembeddings.word_embeddings_layer.weight'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                print('fgm restore')
                # 这里加入fgm restore判断是否恢复参数了
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
