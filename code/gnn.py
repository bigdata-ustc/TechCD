import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph


gcn_msg = fn.copy_u("h", "m")
gcn_reduce = fn.mean("m", "h")

class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation # 激励函数

    def forward(self, node):
        h = self.linear(node.data["h"])
        if self.activation is not None:
            h = self.activation(h)
        return {"h": h}


class GCN(nn.Module):
    def __init__(self, g, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.g = g
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

    def forward(self, feature):
        self.g.ndata["h"] = feature
        self.g.update_all(gcn_msg, gcn_reduce)
        self.g.apply_nodes(func=self.apply_mod)
        return self.g.ndata.pop("h")
