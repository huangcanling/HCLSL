import pandas as pd
import numpy as np
from tqdm import *
import hdf5storage
# from KG import Kg
import os
print(os.getcwd())

# 读取文件
old_SL=pd.read_csv('../data/Human_SL (1).csv', sep=',')
old_SL=old_SL[old_SL['r.statistic_score']>0.4]   #保留置信度0.1-0.4
set_SL=pd.concat([old_SL['n1.name'],old_SL['n2.name']]).drop_duplicates()
ppi = pd.read_csv('../data/ppi.csv', sep=',', names=['gene1', 'gene2','interaction'])
db_name = pd.read_csv('../data/gname.txt',header=None)
dbfeature = pd.read_csv('../data/dbfeature.csv',sep=',',header=None)
go_link=pd.read_csv("../data/CIA_GO2.txt",sep=' ',names=['gene','go'])
data = hdf5storage.loadmat('../data/positive.mat')
seq = data['positive1']

print('求全部数据集的交集')
old_SL = old_SL[['n1.name', 'n2.name']]
set_gene=pd.concat([old_SL['n1.name'],old_SL['n2.name']]).drop_duplicates()
set_ppi=pd.concat([ppi['gene1'],ppi['gene2']]).drop_duplicates()
gog_set=go_link['gene'].drop_duplicates()
db_set=db_name.iloc[:,0].drop_duplicates()
set_ppi.name = 'gene'
gog_set.name = 'gene'
db_set.name = 'gene'
set_gene.name = 'gene'
intersected_df1 = pd.merge(set_ppi, gog_set, how='inner')
intersected_df2 = pd.merge(intersected_df1, db_set, how='inner')
intersected_df3 = pd.merge(intersected_df2, set_gene, how='inner')

print('去除不在交集中的关联对')
SL = old_SL[(old_SL['n1.name'].isin(intersected_df3['gene'])) & (old_SL['n2.name'].isin(intersected_df3['gene']))]
set_gene2=pd.concat([SL['n1.name'],SL['n2.name']]).drop_duplicates().reset_index(drop=True)
go_new = go_link[(go_link['gene'].isin(set_gene2))]
db_new = db_name[(db_name.iloc[:,0].isin(set_gene2))]

print('去除ppi中已知的SL')
SL_list = [tuple(r) for r in SL[['n1.name', 'n2.name']].to_numpy()]
SL_list_reverse = [tuple(r) for r in SL[['n2.name', 'n1.name']].to_numpy()]
ppi_list = [tuple(r) for r in ppi[['gene1', 'gene2']].to_numpy()]
ppi = pd.DataFrame(list(set(ppi_list) - set(SL_list) - set(SL_list_reverse)))
ppi.columns = ['gene1', 'gene2']
# Kg(go_new,ppi,SL,set_gene)

print('处理蛋白序列特征')
# ------------氨基酸性质
dbfeature_new=dbfeature.iloc[db_new.index]
dbfeature_new.index = db_new.iloc[:,0]
dbfeature1 = dbfeature_new.loc[list(set_gene2)].reset_index(drop=True)
# -------------needle 序列相似性
seq_new=pd.DataFrame(seq).iloc[db_new.index]
seq_new.index = db_new.iloc[:,0]
seq_new1=pd.DataFrame(seq_new).iloc[:,db_new.index]
seq_new1.columns = db_new.iloc[:,0]
dbfeature2 = seq_new1.loc[list(set_gene2)].reset_index(drop=True)
dbfeature2 = dbfeature2.loc[:,list(set_gene2)].reset_index(drop=True)

# gog_set2=go_new['gene'].drop_duplicates()
# db_set2=db_new.iloc[:,0].drop_duplicates()
# ppi_new1 = ppi[(ppi['gene1'].isin(set_gene2))]
# ppi_new2 = ppi[(ppi['gene2'].isin(set_gene2))]
# ppi_new=pd.concat([ppi_new1,ppi_new2]).drop_duplicates()

print('处理go特征,构建邻接矩阵')
go_set=go_new['go'].drop_duplicates()
go_map = dict(zip(go_set, range(go_set.shape[0])))
gene_map = dict(zip(set_gene2, range(set_gene2.shape[0])))
go_new = [[gene_map[str(row[0])], go_map[str(row[1])]] for index, row in
           go_new.iterrows()]

# 保存基因映射
g_mapping = {
    "name": gene_map.keys(),
    "id": gene_map.values(),
}
# pd.DataFrame(g_mapping).to_csv("../data/gene_name.txt", index=False, sep='\t')
# np.save("../data/gene_name_0.8.npy",gene_map)

GOfuture=np.zeros((set_gene2.shape[0],go_set.shape[0]))
for row in tqdm(go_new):
    GOfuture[int(row[0]), int(row[1])] = 1


print('处理PPI特征数据')

# ---------------邻接矩阵
ppi_map = dict(zip(set_ppi, range(set_ppi.shape[0])))
ppi_new = [[ppi_map[str(row[0])], ppi_map[str(row[1])]] for index, row in
           ppi.iterrows()]
ppi_feature2=np.zeros((set_ppi.shape[0],set_ppi.shape[0]))
for row in tqdm(ppi_new):
    # ppi_feature2[int(row[0]), int(row[1])] = 1
    ppi_feature2[int(row[1]), int(row[0])] = 1
ppi_feature2=pd.DataFrame(ppi_feature2)
ppi_feature2.index = set_ppi
ppi_feature2_new = ppi_feature2.loc[list(set_gene2)].reset_index(drop=True)


print('构建SL邻接矩阵')
SL = [[gene_map[str(row[0])], gene_map[str(row[1])]] for index, row in
           SL.iterrows()]
sl=np.zeros((set_gene2.shape[0],set_gene2.shape[0]))
for row in tqdm(SL):
    sl[int(row[0]), int(row[1])] = 1
    # sl[int(row[1]), int(row[0])] = 1


np.save('../0.4/slid_0.4',sl)
np.save('../0.4/ppi2_0.4',np.array(ppi_feature2_new))
np.save('../0.4/go_0.4',GOfuture)
np.save('../0.4/pro2_0.4',np.array(dbfeature2))


