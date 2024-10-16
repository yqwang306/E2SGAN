import torch
import torch.nn as nn
import torch.nn.functional as F
import copy, math


class MutualAttention(nn.Module):

    def __init__(self, w, h):
        super(MutualAttention, self).__init__()
        self.w = w
        self.h = h
        self.linears = clones(nn.Linear(w, w), 2)
        self.attn = None

    def forward(self, query, key):
        nbatches = query.size(0)
        query, key = [l(x) for l, x in zip(self.linears, (query, key))]

        x1, self.attn1 = attention(query, key, query)
        x2, self.attn2 = attention(key, query, key)
        
        return torch.cat((x1, x2), dim=1)


class OneSidedAttention(nn.Module):

    def __init__(self, w, h):
        super(OneSidedAttention, self).__init__()
        self.w = w
        self.h = h
        self.linears = clones(nn.Linear(w, w), 2)
        self.attn = None
        self.query = None
        self.key = None

    def forward(self, query, key):
        nbatches = query.size(0)
        value = query  # gcn
        query, key = [l(x) for l, x in zip(self.linears, (query, key))]
        
        x1, self.attn = attention(query, key, value)
        self.query = query
        self.key = key
        
        return x1


def attention(query, key, value, mask=False):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask:
        scores[..., :128, :128] = float('-inf')
        scores[..., 128:, 128:] = float('-inf')

    p_attn = F.softmax(scores, dim=-1)
    return torch.matmul(p_attn, value), p_attn


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class GCN(nn.Module):

    def __init__(self, in_chan, out_chan):
        super(GCN, self).__init__()

        self.embedK = nn.Conv2d(in_chan, out_chan, (1, 7))
        self.embedQ = nn.Conv2d(in_chan, out_chan, (1, 7))
        self.embedV = nn.Conv2d(in_chan, out_chan, (1, 7))
        self.nonlinear = nn.LeakyReLU(0.2, True)
        self.out_chan = out_chan
        self.adj = None
        self.query = None
        self.key = None

    def forward(self, x):
        nbatches = x.size(0)
        w = x.size(-2)
        h = x.size(-1)
        keyEmbedding = self.embedK(x).reshape(nbatches, w, -1)
        queryEmbedding = self.embedQ(x).reshape(nbatches, w, -1)
        x1, self.adj = attention(queryEmbedding, keyEmbedding, x.reshape(nbatches, w, -1), mask=True)
        x1 = x1.reshape(nbatches, -1, w, h)
        x1 = self.embedV(x1 + x)  
        x1 = self.nonlinear(x1)
        self.query = queryEmbedding
        self.key = keyEmbedding

        return x1

    def get_adj(self):
        return self.adj
