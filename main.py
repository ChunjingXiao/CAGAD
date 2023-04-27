import random
import dgl
import torch
import torch.nn.functional as F
import numpy
import argparse
import time
import logging
from GCN import GCN
from GraphSAGE import GraphSAGE
from dataset import Dataset
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, precision_score, confusion_matrix,precision_recall_curve,auc
from BWGNN import *
from sklearn.model_selection import train_test_split
from ddpm.feature_train import ddpmFeatures
from util import saveOneStep

def getChangeNode(H_ALL,C,change_ratio1,change_ratio2):
    np.random.shuffle(C)
    C1 = C[:int(change_ratio1*len(C))]
    # H_ALL.remove(C1)
    H_ALL = H_ALL[:int(change_ratio2*len(H_ALL))]
    return C1,H_ALL

def train(model, g, args):
    # logging.basicConfig(filename=f"./log/logger_"+args.dataset+".log",filemode="a",format="%(asctime)s-%(name)s-%(levelname)s-%(message)s",level=logging.INFO)
    # logger=logging.getLogger('BWGNN')
    features = g.ndata['feature'].clone()
    homo = args.homo
    labels = g.ndata['label']
    abchr = args.abchr
    nchr = args.nchr
    neighborchr = args.neighborchr
    dataset_name = args.dataset
    normal_idx=np.where(labels!=1)[0]
    abnormal_idx=np.where(labels==1)[0]
    index = list(range(len(labels)//3))
    if dataset_name == 'amazon':
        index = list(range(3305, len(labels)//3))
    random_state = 2
    idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels[index], stratify=labels[index],
                                                            train_size=args.train_ratio,
                                                            random_state=random_state, shuffle=True)
    idx_rest,idx_change,y_rest,y_change = train_test_split(idx_rest, y_rest, stratify=y_rest,
                                                            test_size=0.5,
                                                            random_state=random_state, shuffle=True)
    idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest,
                                                            test_size=0.67,
                                                            random_state=random_state, shuffle=True)
    # t = list(abnormal_idx)
    # temp = t[:len(t)//3]
    # saveOneStep(graph,temp,labels)
    abC = []
    AB = list(abnormal_idx) +idx_train
    for i in AB:
        if i in idx_train and i in list(abnormal_idx):
            if i not in abC:
                abC.append(i)
        else:
            continue
    nC = []
    AB = list(normal_idx) +idx_train
    for i in AB:
        if i in idx_train and i in list(normal_idx):
            if i not in nC:
                nC.append(i)
        else:
            continue
    '''
    obtain replaced nodes
    '''

    abC1,abH_ALL = getChangeNode(idx_change,abC,abchr,neighborchr)
    nC1,nH_ALL = getChangeNode(idx_change,idx_test+idx_valid,nchr,neighborchr)
    ab_features = features[abC]
    
    non_features = features[nC][:len(abC)]
    # put ab_features to generate new gen_features
    gen_features = ddpmFeatures(ab_features,non_features)
    # gen_features = ab_features
    # features of anomalies
    abC2 = [ i+len(labels)//3 for i in abC1]
    nC2 = [ i+2*len(labels)//3 for i in nC1]
    for i in abH_ALL:
        t = random.randint(0,len(abC)-1)
        features[i+len(labels)//3] =gen_features[t]
    for i in nH_ALL:
        t = random.randint(0,len(abC)-1)
        features[i+2*len(labels)//3] =gen_features[t]

    C1 = nC1
    C2 = nC2
    train_mask = torch.zeros([len(labels)]).bool()
    val_mask = torch.zeros([len(labels)]).bool()
    test_mask = torch.zeros([len(labels)]).bool()
    train_mask[idx_train+abC2] = 1
    val_mask[idx_valid] = 1
    test_mask[idx_test] = 1
    print('train/dev/test samples: ', train_mask.sum().item(), val_mask.sum().item(), test_mask.sum().item())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_f1, final_tf1, final_trec, final_tpre, final_tmf1, final_tauc,finanl_tauc_pr = 0., 0., 0., 0., 0., 0.,0.

    weight = (1-labels[train_mask]).sum().item() / labels[train_mask].sum().item()
    print('cross entropy weight: ', weight)
    time_start = time.time()

    for e in range(args.epoch):
        model.train()
        logits = model(g,features)
        logits[C1] = logits[C2]
        
        loss = F.cross_entropy(logits[train_mask], labels[train_mask], weight=torch.tensor([1., weight]))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.eval()
        probs = logits.softmax(1)
        f1, thres = get_best_f1(labels[val_mask], probs[val_mask])

        preds = numpy.zeros_like(labels)
        preds[probs[:, 1] > thres] = 1
        trec = recall_score(labels[test_mask], preds[test_mask])
        tpre = precision_score(labels[test_mask], preds[test_mask])
        tmf1 = f1_score(labels[test_mask], preds[test_mask], average='macro')
        tauc = roc_auc_score(labels[test_mask], probs[test_mask][:, 1].detach().numpy())
        precision,recall,_ = precision_recall_curve(labels[test_mask], probs[test_mask][:, 1].detach().numpy())
        tauc_pr = auc(recall,precision)
        if best_f1 < f1:
            best_f1 = f1
            final_trec = trec
            final_tpred = preds
            final_tpre = tpre
            final_tmf1 = tmf1
            final_tauc = tauc
            finanl_tauc_pr = tauc_pr
        print('Epoch {}, loss: {:.4f}, val mf1: {:.4f}, (best {:.4f})'.format(e, loss, f1, best_f1))

    time_end = time.time()
    
    print('time cost: ', time_end - time_start, 's')
    print('Test: REC {:.2f} PRE {:.2f} MF1 {:.2f} AUC {:.2f} AUC_PR {:.2f}'.format(final_trec*100,
                                                                     final_tpre*100, final_tmf1*100, final_tauc*100,finanl_tauc_pr*100))
    
    # logger.info('dataset {:s} {:.2f}'.format(args.dataset,args.train_ratio))
    # logger.info('random_state {:d}  Test: REC {:.2f} PRE {:.2f} MF1 {:.2f} AUC {:.2f} AUC_PR {:.2f}'.format(random_state,final_trec*100,
                                                                    #  final_tpre*100, final_tmf1*100, final_tauc*100,finanl_tauc_pr*100))
    # logger.info('abchr {:.1f}, nchr {:.1f},neighborchr {:.1f}\n'.format(abchr,nchr,neighborchr))
    return final_tmf1, final_tauc


# threshold adjusting for best macro f1
def get_best_f1(labels, probs):
    best_f1, best_thre = 0, 0
    for thres in np.linspace(0.05, 0.95, 19):
        preds = np.zeros_like(labels)
        preds[probs[:,1] > thres] = 1
        mf1 = f1_score(labels, preds, average='macro')
        if mf1 > best_f1:
            best_f1 = mf1
            best_thre = thres
    return best_f1, best_thre


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='BWGNN')
    parser.add_argument("--dataset", type=str, default="yelp",
                        help="Dataset for this model (yelp/amazon/tfinance/pubmed)")
    parser.add_argument("--train_ratio", type=float, default=0.01, help="Training ratio")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--abchr", type=float, default=1, help="Abnormal data exchange ratio")
    parser.add_argument("--nchr", type=float, default=0, help="Normal data exchange ratio")
    parser.add_argument("--model", type=str, default="BWGNN",
                        help="(BWGNN/GCN/GraphSAGE)")
    parser.add_argument("--neighborchr", type=float, default=0.4, help="neighbor data exchange ratio")
    parser.add_argument("--hid_dim", type=int, default=64, help="Hidden layer dimension")
    parser.add_argument("--order", type=int, default=2, help="Order C in Beta Wavelet")
    parser.add_argument("--homo", type=int, default=0, help="1 for BWGNN(Homo) and 0 for BWGNN(Hetero)")
    parser.add_argument("--epoch", type=int, default=100, help="The max number of epochs")
    parser.add_argument("--run", type=int, default=1, help="Running times")

    args = parser.parse_args()
    print(args)
    dataset_name = args.dataset
    homo = args.homo
    order = args.order
    h_feats = args.hid_dim
    graph = Dataset(dataset_name, homo).graph
    if (homo):
        c = list(graph.edges())
        h =c[0]+graph.num_nodes()
        k=c[1]+graph.num_nodes()
        h2 =c[0]+2*graph.num_nodes()
        k2=c[1]+2*graph.num_nodes()
        c[0] = torch.cat((c[0],h,h2))
        c[1] = torch.cat((c[1],k,k2))
        c = tuple(c)
        graph2= dgl.graph(c)
    else:
        c1 = list(graph.edges(etype='net_rsr'))
        h =c1[0]+graph.num_nodes()
        k=c1[1]+graph.num_nodes()
        h2 =c1[0]+2*graph.num_nodes()
        k2=c1[1]+2*graph.num_nodes()
        c1[0] = torch.cat((c1[0],h,h2))
        c1[1] = torch.cat((c1[1],k,k2))
        c1 = tuple(c1)

        c2 = list(graph.edges(etype='net_rtr'))
        h =c2[0]+graph.num_nodes()
        k=c2[1]+graph.num_nodes()
        h2 =c2[0]+2*graph.num_nodes()
        k2=c2[1]+2*graph.num_nodes()
        c2[0] = torch.cat((c2[0],h,h2))
        c2[1] = torch.cat((c2[1],k,k2))
        c2 = tuple(c2)

        c3 = list(graph.edges(etype='net_rur'))
        h =c3[0]+graph.num_nodes()
        k=c3[1]+graph.num_nodes()
        h2 =c3[0]+2*graph.num_nodes()
        k2=c3[1]+2*graph.num_nodes()
        c3[0] = torch.cat((c3[0],h,h2))
        c3[1] = torch.cat((c3[1],k,k2))
        c3 = tuple(c3)

        print(graph)
        graph_data = {
            ('review', 'net_rsr', 'review'):c1,
            ('review', 'net_rtr', 'review'):c2,
            ('review', 'net_rur', 'review'):c3
        }
        graph2 = dgl.heterograph(graph_data)

    graph2.ndata['feature'] = torch.cat((graph.ndata['feature'],graph.ndata['feature'],graph.ndata['feature']))
    graph2.ndata['label'] = torch.cat((graph.ndata['label'],graph.ndata['label'],graph.ndata['label']))

    f = graph2.ndata['label'] 
    graph = graph2
    in_feats = graph.ndata['feature'].shape[1]

    num_classes = 2

    if args.run == 1:

        if homo:
            if args.model =='BWGNN':
                model = BWGNN(in_feats, h_feats, num_classes, graph, d=order)
            elif args.model =='GCN': 
                model = GCN(None,in_feats,h_feats,num_classes,3,F.relu,0.5)
            elif args.model =='GraphSAGE':  
                model = GraphSAGE(None,in_feats,h_feats,num_classes,3,F.relu,0.5,aggregator_type='pool')
        else:
            model = BWGNN_Hetero(in_feats, h_feats, num_classes, graph, d=order)
        train(model, graph, args)

    else:
        final_mf1s, final_aucs = [], []
        for tt in range(args.run):
            if homo:
                if args.model =='BWGNN':
                    model = BWGNN(in_feats, h_feats, num_classes, graph, d=order)
                elif args.model =='GCN': 
                    model = GCN(None,in_feats,h_feats,num_classes,3,F.relu,0.5)
                elif args.model =='GraphSAGE':  
                    model = GraphSAGE(None,in_feats,h_feats,num_classes,3,F.relu,0.5,aggregator_type='pool')
            else:
                model = BWGNN_Hetero(in_feats, h_feats, num_classes, graph, d=order)
            mf1, auc1 = train(model, graph, args)
            final_mf1s.append(mf1)
            final_aucs.append(auc1)
        final_mf1s = np.array(final_mf1s)
        final_aucs = np.array(final_aucs)
        print('MF1-mean: {:.2f}, MF1-std: {:.2f}, AUC-mean: {:.2f}, AUC-std: {:.2f}'.format(100 * np.mean(final_mf1s),
                                                                                            100 * np.std(final_mf1s),
                                                               100 * np.mean(final_aucs), 100 * np.std(final_aucs)))
