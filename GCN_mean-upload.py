
import os
import numpy as np
import pandas as pd
from tqdm import tqdm,trange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from sklearn.metrics import precision_score, recall_score, roc_auc_score,roc_curve, f1_score,accuracy_score,auc
import networkx as nx
from torch_geometric.utils.convert import to_networkx
import matplotlib.pyplot as plt

EPOCH= 201
step=2
r=0.6
    
df_e= pd.read_csv(r'D:\CERT_database\r4.2\user_feature_label\1-data-test-undirected_edge.csv')
df_data= pd.read_csv(r'D:\CERT_database\r4.2\user_feature_label\1-data-test-combine.csv')
edge= df_e.to_numpy().T

edge_index = torch.from_numpy(edge)                         

# Input for x, [node_number,feature_dim]
cols= ['f1','f2','f3','f4','f5']
x = df_data[cols].to_numpy(dtype=np.float32)
x = torch.from_numpy(x)  # 

# prepare lable y
y = df_data['label'].to_numpy()
y = torch.from_numpy(y)                               


#wayne' debug:
#print(x)
#print(y)
    
# pack x,y into pyg special data class
data = Data(x=x,
            edge_index=edge_index,
            y=y)
# seperate dataset make sure train and test are the same with CNN
data.train_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
data.train_mask[:data.num_nodes - 572] = 1                  
data.val_mask = None                                         # 0valid
data.test_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
data.test_mask[data.num_nodes - 572:] = 1                    # 1510test
data.num_classes = 2
#degbug:
# print(data.keys)
# print(data.test_mask[:])
data

print("graph information:")
print("edges : {}/2={}".format(data.num_edges, int(data.num_edges/2)))
print("nodes : {}".format(data.num_nodes))
print("node feature dim : {}".format(data.num_node_features))
print("training nodes : {}".format(data.train_mask.sum().item()))
print("testing nodes : {}".format(data.test_mask.sum().item()))
print("output dimension : {}".format(data.num_classes))


class Net(torch.nn.Module):
    def __init__(self, feat_dim, num_class):
        super(Net, self).__init__()
        self.conv1 = GCNConv(feat_dim, 16)
        self.conv2 = GCNConv(16, num_class)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

#start training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(feat_dim=data.num_node_features, num_class=2).to(device)         # Initialize model
data = data.to(device)                                                       
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4) # Initialize optimizer and training params

## try to draw graph, takes long time, meanless for visual
# G = to_networkx(data,to_undirected=(True),remove_self_loops=(True))
# nx.draw_networkx(G,with_labels=(False),node_size=3)
# plt.savefig(r'D:\CERT_database\r4.2\user_feature_label\graph.png')
# plt.show()

# create empty list to store result for recorded epoch
L_pos=np.empty([0,572])
L_acc=[]
L_pre=[]
L_rec=[]
L_f1=[]



for epoch in range(EPOCH):
    model.train()
    optimizer.zero_grad()
    # Get output
    out = model(data)
# wayne's debug:
    #print(out)    
    
    # Get loss
    loss = F.nll_loss(out[data.train_mask.bool()], data.y[data.train_mask.bool()].long())
    
    # Backward
    loss.backward()
    optimizer.step()
    # when training after a certain epoch, record result every other step
    if (epoch % step ==0)and (epoch > r*EPOCH):
        model.eval()
        _, pred = model(data).max(dim=1)
        #debug:
        test_out= model(data)[data.num_nodes - 572:]
                    # test_output, last_layer = cnn(test_x)
        pred_y = torch.masked_select(pred, data.test_mask.bool()).tolist()# wayne' note: return max one's index in dimension 1
        true_y = data.y[data.test_mask.bool()].tolist()
        l_out= np.array(test_out[:,1].tolist())
        l_out= l_out[np.newaxis]
        L_pos= np.append(L_pos,l_out,axis=0)
        L_acc =np.append(L_acc,accuracy_score(true_y, pred_y)) 
        L_pre=np.append(L_pre,precision_score(true_y, pred_y))
        L_rec=np.append(L_rec,precision_score(true_y, pred_y))
        L_f1=np.append(L_f1,f1_score(true_y, pred_y))
        #print(L_acc,L_pre,L_rec,L_f1)

true_y_gcn= data.y[data.test_mask.bool()].tolist()                       
pos_score_gcn= np.mean(L_pos,axis=0)
accuracy_gcn= np.mean(L_acc)
precision_gcn= np.mean(L_pre)
recall_gcn= np.mean(L_rec)
f1_gcn= np.mean(L_f1)

     

fpr_g, tpr_g, threshold_g = roc_curve(true_y_gcn,pos_score_gcn)
roc_auc_g = auc(fpr_g,tpr_g)   

#print('roc_auc:', roc_auc)
lw = 1.5
#plt.subplot(5,3,2)
# plt.plot(fpr, tpr, color='grey',
#          lw=lw, label='CNN (AUC = %0.4f)' % roc_auc)  
plt.plot(fpr_g, tpr_g, color='red',
         lw=lw, label='GCN (AUC = %0.4f)' % roc_auc_g)  

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random', x=0.6,y=0.4)
plt.legend(loc="lower right")

plt.show()

print(accuracy_gcn,precision_gcn ,recall_gcn ,f1_gcn)

#torch.save(gcn,'gcn_cert4.2.pkl')
print('finish training')        
        #
        # correct = float(pred[data.test_mask.bool()].eq(data.y[data.test_mask.bool()]).sum().item())
        # acc = correct / data.test_mask.sum().item()
        # print('Accuracy: {:.4f}'.format(acc))

## use gcn's output embeding as new feature for each node        
# model.eval()
# out=model(data).tolist()
# df=pd.DataFrame()
# df['g1']=[i[0] for i in out]
# df['g2']=[i[1] for i in out]
# df.to_csv(r'D:\CERT_database\r4.2\user_feature_label\1-data-test-gfeature-gcn.csv',index=False)

