from layer import *
import numpy as np
import torch
import torch.nn as nn
from torch.nn import BCELoss
import torch.utils.data.dataset as Dataset
num=5866
dim=256


def sim(z1: torch.Tensor, z2: torch.Tensor):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())


def contrastive_loss(h1, h2, tau=0.7):
    sim_matrix = sim(h1, h2)
    f = lambda x: torch.exp(x / tau)
    matrix_t = f(sim_matrix)
    numerator = matrix_t.diag()
    denominator = torch.sum(matrix_t, dim=-1)
    loss = -torch.log(numerator / denominator).mean()
    return loss


class Model(nn.Module):
    def __init__(self, num_in_node=num, num_in_edge=num, num_hidden1=512, num_out=dim):  # 435, 757, 512, 128
        super(Model, self).__init__()
        self.node_encoders1 = node_encoder(num_in_edge, num_hidden1, 0.3)

        self.hyperedge_encoders1 = hyperedge_encoder(num_in_node, num_hidden1, 0.3)

        self.decoder1 = decoder1(act=lambda x: x)
        self.decoder2 = decoder2(act=lambda x: x)

        self._enc_mu_node = node_encoder(num_hidden1, num_out, 0.3, act=lambda x: x)
        self._enc_log_sigma_node = node_encoder(num_hidden1, num_out, 0.3, act=lambda x: x)

        self._enc_mu_hedge = node_encoder(num_hidden1, num_out, 0.3, act=lambda x: x)
        self._enc_log_sigma_hyedge = node_encoder(num_hidden1, num_out, 0.3, act=lambda x: x)

        self.hgnn_node2 = HGNN1(num_in_node, num_in_node, num_out, num_in_node, num_in_node)
        self.hgnn_hyperedge2 = HGNN1(num_in_edge, num_in_edge, num_out, num_in_edge, num_in_edge)

        self.classifierLayer = ClassifierLayer()

    def forward(self, HSL, HSLT, HF, gene1_feat, gene2_feat, data, label):
        h_feature_1 = self.hgnn_hyperedge2(gene1_feat, HSL)
        n_feature_1 = self.hgnn_node2(gene2_feat, HSLT)

        h_feature_2 = self.hgnn_hyperedge2(gene1_feat, HF)
        n_feature_2 = self.hgnn_node2(gene2_feat, HF)

        data = data.t()
        gene1 = data[0]
        gene2 = data[1]
        gene1_feature1, gene1_feature2 = h_feature_1[gene1], h_feature_2[gene1]
        gene2_feature1, gene2_feature2 = n_feature_1[gene2], n_feature_2[gene2]

        gene1_feature = torch.add(gene1_feature1, gene1_feature2)
        gene2_feature = torch.add(gene2_feature1, gene2_feature2)

        pre = self.classifierLayer(gene1_feature, gene2_feature)
        criterion = BCELoss()
        pre = pre.reshape(-1)
        ceLoss = criterion(pre, label.float())
        # ceLoss = F.cross_entropy(pre, label)

        loss_c_g1 = contrastive_loss(gene1_feature1, gene1_feature2)
        loss_c_g2 = contrastive_loss(gene2_feature1, gene2_feature2)
        loss_c = loss_c_g1 + loss_c_g2

        return ceLoss, loss_c, pre

    def predict(self, HSL, HSLT, HF, gene1_feat, gene2_feat, data):

        h_feature_1 = self.hgnn_hyperedge2(gene1_feat, HSL)
        n_feature_1 = self.hgnn_node2(gene2_feat, HSLT)

        h_feature_2 = self.hgnn_hyperedge2(gene1_feat, HF)
        n_feature_2 = self.hgnn_node2(gene2_feat, HF)

        data = data.t()
        gene1 = data[0]
        gene2 = data[1]

        gene1_feature1, gene1_feature2 = h_feature_1[gene1], h_feature_2[gene1]
        gene2_feature1, gene2_feature2 = n_feature_1[gene2], n_feature_2[gene2]

        gene1_feature = torch.add(gene1_feature1, gene1_feature2)
        gene2_feature = torch.add(gene2_feature1, gene2_feature2)

        pre = self.classifierLayer(gene1_feature, gene2_feature)
        pre = pre.reshape(-1)

        return pre,gene1_feature1,gene1_feature2


class ClassifierLayer(nn.Module):
    def __init__(self):
        super(ClassifierLayer, self).__init__()
        self.lin1 = nn.Linear(dim * 2, dim)
        self.lin2 = nn.Linear(dim, 1)

    def forward(self, gene1_f, gene2_f):
        embeds = torch.cat((gene1_f, gene2_f), 1)
        embeds = embeds.to(torch.float32)
        embeds = F.relu(self.lin1(embeds))
        embeds = F.dropout(embeds, p=0.4, training=self.training)
        embeds = self.lin2(embeds)
        ret = torch.sigmoid(embeds)

        return ret



class CVEdgeDataset(Dataset.Dataset):
    def __init__(self, edge, label):

        self.Data = edge
        self.Label = label

    def __len__(self):
        return len(self.Label)

    def __getitem__(self, index):
        data = self.Data[index]
        label = self.Label[index]
        return data, label


def calcRegLoss(model):
    ret = 0
    for W in model.parameters():
        ret +=W.norm(2).square()
    return ret