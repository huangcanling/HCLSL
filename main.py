from sklearn.model_selection import KFold
from torch import nn, optim
from torch.autograd import Variable
import argparse
from sklearn.metrics import accuracy_score,precision_recall_curve, auc,roc_curve
from model import Model,CVEdgeDataset,calcRegLoss
from hypergraph_utils import *
import os
from kl_loss import kl_loss
import torch.utils.data.dataloader as DataLoader
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
print(os.getcwd())

num=5866
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train.')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--hidden', type=int, default=128, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--lr', type=float, default=0.002, help='Initial learning rate.')        #原来
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--cv_num', type=int, default=5, help='number of fold')
parser.add_argument('--batch', type=int, default=8192, help='batch size')
parser.add_argument('--dim', type=int, default=256, help='embedding size')  # 原 128
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

SL = np.load("../0.4/slid_0.4.npy")
GO = np.load("../0.4/go_0.4.npy")
GG = np.load("../0.4/ppi2_0.4.npy")
SG = np.load("../0.4/pro2_0.4.npy")

feature_a = np.hstack((GO,GG,SG))

[row, col] = np.shape(SL)
rng = np.random.default_rng(seed=42)
pos_samples = np.where(SL ==1)
pos_samples_shuffled = rng.permutation(pos_samples, axis=1)
L = np.zeros((num, num))
L[np.tril_indices(num)] = 1.
SL_wx = SL.T + SL + L
rng = np.random.default_rng(seed=42)
neg_samples = np.where(SL_wx ==0)
neg_samples_shuffled = rng.permutation(neg_samples, axis=1)[:, :pos_samples_shuffled.shape[1]]
pos_edge = pos_samples_shuffled.T
neg_edge = neg_samples_shuffled.T
p_label = np.ones(pos_edge.shape[0])
n_label = np.zeros(pos_edge.shape[0])
pos_edge=np.insert(pos_edge,2,p_label,axis=1)
neg_edge=np.insert(neg_edge,2,n_label,axis=1)
data = np.vstack((pos_edge, neg_edge))
data = rng.permutation(data)
edge = np.array(data[:,:2], dtype='int64')
label = np.array(data[:,2:], dtype='float32').squeeze()

kf = KFold(n_splits=args.cv_num)
torch.manual_seed(42)
AUC_sum = []
AUPR_sum = []
Acc_sum = []
pre_sum,sen_sum,spe_sum,Mcc_sum=[],[],[],[]

train_idx, test_idx = [], []
for train_index, test_index in kf.split(edge):
    train_idx.append(train_index)
    test_idx.append(test_index)

for i in range(1):
    auc_best = 0
    aupr_best = 0
    AUC = 0
    acc_best = 0
    spe_best,sen_best,pre_best,Mcc_best=0,0,0,0

    edge_train, edge_test = edge[train_idx[i]], edge[test_idx[i]]
    label_train, label_test = label[train_idx[i]], label[test_idx[i]]

    trainEdges = CVEdgeDataset(edge_train, label_train)
    testEdges = CVEdgeDataset(edge_test, label_test)
    trainLoader = DataLoader.DataLoader(trainEdges, batch_size=args.batch, shuffle=True, num_workers=0)
    tstLoader = DataLoader.DataLoader(testEdges, batch_size=args.batch, shuffle=True, num_workers=0)

    HHF = construct_H_with_KNN(feature_a)
    HF = generate_G_from_H(HHF)
    HF = HF.double()

    HHSL = construct_H_with_KNN(SL)
    HSL = generate_G_from_H(HHSL)
    HSL = HSL.double()

    HHSLT = construct_H_with_KNN(SL.T)
    HSLT = generate_G_from_H(HHSL)
    HSLT = HSLT.double()

    model = Model()
    optimizer2 = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    gene1_feat = torch.eye(num)
    gene2_feat = torch.eye(num)  # 5954
    gene1_feat, gene2_feat = Variable(gene1_feat), Variable(gene2_feat)
    loss_kl = kl_loss(num, num)
    if args.cuda:
        model.cuda()
        HSL = HSL.cuda()
        HSLT = HSLT.cuda()
        HF = HF.cuda()
        gene1_feat = gene1_feat.cuda()
        gene2_feat = gene2_feat.cuda()

    for epoch in range(args.epochs):
        model.train()
        for i, item in enumerate(trainLoader):
            train_data, train_label = item
            train_data = train_data.long().cuda()
            train_label = train_label.long().cuda()
            ceLoss, loss_c, tr_pre = model.forward(HSL, HSLT, HF, gene1_feat, gene2_feat, train_data, train_label)
            tr_pre = tr_pre.detach().cpu()
            train_label = train_label.detach().cpu()
            # auc_val = roc_auc_score(train_label, tr_pre)
            fpr,tpr,thre=roc_curve(train_label, tr_pre)
            auc_val=auc(fpr,tpr)
            print('auc', auc_val)
            scores = tr_pre
            scores[scores >= 0.5] = 1
            scores[scores < 0.5] = 0
            acc_test = accuracy_score(train_label, scores)
            regLoss = calcRegLoss(model) * 1e-7
            # ceLoss = 0
            loss = ceLoss + loss_c * 1e-2 + regLoss
            print('The loss score', loss, 'and the batch', i)
            loss.backward()
            optimizer2.step()
            optimizer2.zero_grad()

        model.eval()
        for i, item in enumerate(tstLoader):
            test_data, test_label = item
            test_data = test_data.long().cuda()
            pre,gene1_feature,gene1_feature = model.predict(HSL, HSLT, HF, gene1_feat, gene2_feat, test_data)

            # 进行参数评估
            pre = pre.detach().cpu()
            test_label = test_label.detach().cpu()
            fpr, tpr, thre = roc_curve(test_label, pre)
            auc_test = auc(fpr, tpr)
            precision, recall, _ = precision_recall_curve(test_label, pre)
            aupr_test = auc(recall, precision)
            curve = {'fpr': fpr.reshape(-1, 1), 'tpr': tpr.reshape(-1, 1), 'prec': precision.reshape(-1, 1),
                     'rec': recall.reshape(-1, 1)}
            # np.save('MHGCNSL', curve)
            scores = pre
            scores[scores >= 0.5] = 1
            scores[scores < 0.5] = 0
            acc_test = accuracy_score(test_label, scores)
            tp = 0
            fp = 0
            tn = 0
            fn = 0
            for j in range(len(test_label)):
                if test_label[j] == 1:
                    if test_label[j] == scores[j]:
                        tp = tp + 1
                    else:
                        fn = fn + 1
                else:
                    if test_label[j] == scores[j]:
                        tn = tn + 1
                    else:
                        fp = fp + 1
            if tp == 0 and fp == 0:
                sensitivity = float(tp) / (tp + fn)
                specificity = float(tn) / (tn + fp)
                precision = 0
                MCC = 0
            else:
                sensitivity = float(tp) / (tp + fn)
                specificity = float(tn) / (tn + fp)
                precision = float(tp) / (tp + fp)
                MCC = float(tp * tn - fp * fn) / (np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))))

        print('Epoch: {:04d}'.format(epoch + 1),
              'loss: {:.4f}'.format(loss.data.item()),
              'auc_val: {:.4f}'.format(auc_test),
              'aupr_val: {:.4f}'.format(aupr_test),
              'acc: {:.4f}'.format(acc_test),'sen: {:.4f}'.format(sensitivity),'spe: {:.4f}'.format(specificity),'pre: {:.4f}'.format(precision),'MCC: {:.4f}'.format(MCC)
              )

        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

        if AUC < auc_test:
            AUC = auc_test
            auc_best = auc_test
            aupr_best = aupr_test
            acc_best = acc_test
            pre_best,sen_best,spe_best,Mcc_best=precision,sensitivity,specificity,MCC
            # precision1 = precision
            # f11 = max_f1_
        print('auc_best: {:.4f}'.format(auc_best),
              'aupr_best: {:.4f}'.format(aupr_best),
              'acc_best: {:.4f}'.format(acc_best),'pre_best: {:.4f}'.format(pre_best),'sen_best: {:.4f}'.format(sen_best),'spe_best: {:.4f}'.format(spe_best),'Mcc_best: {:.4f}'.format(Mcc_best))

    AUC_sum.append(auc_best)
    AUPR_sum.append(aupr_best)
    Acc_sum.append(acc_best)
    pre_sum.append(pre_best)
    sen_sum.append(sen_best)
    spe_sum.append(spe_best)
    Mcc_sum.append(Mcc_best)

print('AUC_m: {:.4f}'.format(np.mean(AUC_sum)),
      'AUPR_m: {:.4f}'.format(np.mean(AUPR_sum)),
      'Acc_m: {:.4f}'.format(np.mean(Acc_sum)),'pre_m: {:.4f}'.format(np.mean(pre_sum)),'sen_m: {:.4f}'.format(np.mean(sen_sum)),'spe_m: {:.4f}'.format(np.mean(spe_sum)),'Mcc_m: {:.4f}'.format(np.mean(Mcc_sum)))

















