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

        # return torch.argmax(start,dim=1), torch.argmax(end,dim=1)
        return start, end





