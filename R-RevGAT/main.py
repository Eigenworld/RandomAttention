import math
import os
import random
import time

import dgl
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
from loss import loss_kd_only
from model_rev import R_RevGAT

from collections import namedtuple

epsilon = 1 - math.log(2)
dataset = "ogbn-arxiv"
n_node_feats, n_classes = 0, 0

#args start--------------------------------------------------------------------
path = '/home/lixue/XRT/ogbn-arxiv/X.all.xrt-emb.npy'# path for giant xrt embed
argparser = namedtuple('argparser', [
    'n_runs',
    'n_epochs',
    'gpu',
    'use_labels',
    'n_label_iters',
    'mask_rate',
    'no_attn_dst',
    'use_norm',
    'lr',
    'n_layers',
    'n_heads',
    'n_hidden',
    'dropout',
    'input_drop',
    'attn_drop',
    'edge_drop',
    'wd',
    'log_every',
    'save_pred',
    'save',
    'backbone',
    'group',
    'kd_dir',
    'mode',
    'alpha',
    'temp',
  
    'data_root_dir',
    'pretrain_path',])

args = argparser(
    n_runs=1,
    n_epochs=1000,#==========
    gpu=0,
    use_labels=True,
    n_label_iters=1,
    mask_rate=0.5,
    no_attn_dst = True,
    use_norm=True,
    lr=0.002,
    n_layers=2,
    n_heads=3,
    n_hidden=256,
    dropout=0.75,
    input_drop=0.25,
    attn_drop=0.,
    edge_drop=0.3,
    wd=0,
    log_every=20,#==========
    save_pred=False,
    save='kd',
    backbone='rev',
    group=2,#===============================
    kd_dir='./kd',
    mode='teacher',#=====teacher->student
    alpha=0.95,
    temp=0.7,
 
    data_root_dir='/home/lixue/Ipy5/OverSmoothing/dgl_dataset',#======set data root dir
    pretrain_path=path,)
# args over--------------------------------------------------------------------------

if not args.use_labels and args.n_label_iters > 0:
        raise ValueError("'--use-labels' must be enabled when n_label_iters > 0")
device = torch.device(f"cuda:{args.gpu}")

def load_data(dataset,args):
    global n_node_feats, n_classes

    if args.data_root_dir == 'default':
        data = DglNodePropPredDataset(name=dataset)
    else:
        data = DglNodePropPredDataset(name=dataset,root=args.data_root_dir)

    evaluator = Evaluator(name=dataset)

    splitted_idx = data.get_idx_split()
    train_idx, val_idx, test_idx = splitted_idx["train"], splitted_idx["valid"], splitted_idx["test"]
    graph, labels = data[0]
    
    # Replace node features here
    if args.pretrain_path != 'None':
        #graph.ndata["feat"] = torch.tensor(np.load(args.pretrain_path)).float()
        xrt = torch.tensor(np.load(args.pretrain_path)).float()
        graph.ndata["feat"] = torch.cat([xrt,graph.ndata["feat"]], dim=-1)
        print("Pretrained node feature loaded! Path: {}".format(args.pretrain_path))
        
    
    n_node_feats = graph.ndata["feat"].shape[1]
    n_classes = (labels.max() + 1).item()

    return graph, labels, train_idx, val_idx, test_idx, evaluator

def preprocess(graph):
    global n_node_feats

    # make bidirected
    feat = graph.ndata["feat"]
    graph = dgl.to_bidirected(graph)
    graph.ndata["feat"] = feat

    # add self-loop
    print(f"Total edges before adding self-loop {graph.number_of_edges()}")
    graph = graph.remove_self_loop().add_self_loop()
    print(f"Total edges after adding self-loop {graph.number_of_edges()}")

    graph.create_formats_()

    return graph

def gen_model(args):
    if args.use_labels:
        n_node_feats_ = n_node_feats + n_classes
    else:
        n_node_feats_ = n_node_feats

    if args.backbone == "rev":
        model = R_RevGAT(
                      n_node_feats_,
                      n_classes,
                      n_hidden=args.n_hidden,
                      n_layers=args.n_layers,
                      n_heads=args.n_heads,
                      activation=F.relu,
                      dropout=args.dropout,
                      input_drop=args.input_drop,
                      attn_drop=args.attn_drop,
                      edge_drop=args.edge_drop,
                      use_attn_dst=not args.no_attn_dst,
                      use_symmetric_norm=args.use_norm)
    else:
        raise Exception("Unknown backnone")

    return model


def custom_loss_function(x, labels):
    y = F.cross_entropy(x, labels[:, 0], reduction="none")
    y = torch.log(epsilon + y) - math.log(epsilon)
    return torch.mean(y)


def add_labels(feat, labels, idx):
    onehot = torch.zeros([feat.shape[0], n_classes], device=device)
    onehot[idx, labels[idx, 0]] = 1
    return torch.cat([feat, onehot], dim=-1)


def adjust_learning_rate(optimizer, lr, epoch):
    if epoch <= 50:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr * epoch / 50


def train(args, model, graph, labels, train_idx, val_idx, test_idx, optimizer,
          evaluator, mode='teacher', teacher_output=None):
    model.train()
    if mode == 'student':
        assert teacher_output != None

    alpha = args.alpha
    temp = args.temp

    feat = graph.ndata["feat"]

    if args.use_labels:
        mask = torch.rand(train_idx.shape) < args.mask_rate

        train_labels_idx = train_idx[mask]
        train_pred_idx = train_idx[~mask]

        feat = add_labels(feat, labels, train_labels_idx)
    else:
        mask = torch.rand(train_idx.shape) < args.mask_rate

        train_pred_idx = train_idx[mask]

    optimizer.zero_grad()

    if args.n_label_iters > 0:
        with torch.no_grad():
            pred = model(graph, feat)
    else:
        pred = model(graph, feat)

    if args.n_label_iters > 0:
        unlabel_idx = torch.cat([train_pred_idx, val_idx, test_idx])
        for _ in range(args.n_label_iters):
            pred = pred.detach()
            torch.cuda.empty_cache()
            feat[unlabel_idx, -n_classes:] = F.softmax(pred[unlabel_idx], dim=-1)
            pred = model(graph, feat)
    if mode == 'teacher':
        loss = custom_loss_function(pred[train_pred_idx],
                                    labels[train_pred_idx])
    elif mode == 'student':
        loss_gt = custom_loss_function(pred[train_pred_idx],
                                       labels[train_pred_idx])
        loss_kd = loss_kd_only(pred, teacher_output, temp)
        loss = loss_gt * (1 - alpha) + loss_kd * alpha
    else:
        raise Exception('unkown mode')

    loss.backward()
    optimizer.step()

    return evaluator(pred[train_idx], labels[train_idx]), loss.item()


@torch.no_grad()
def evaluate(args, model, graph, labels, train_idx, val_idx, test_idx, evaluator):
    model.eval()

    feat = graph.ndata["feat"]

    if args.use_labels:
        feat = add_labels(feat, labels, train_idx)

    pred = model(graph, feat)

    if args.n_label_iters > 0:
        unlabel_idx = torch.cat([val_idx, test_idx])
        for _ in range(args.n_label_iters):
            feat[unlabel_idx, -n_classes:] = F.softmax(pred[unlabel_idx], dim=-1)
            pred = model(graph, feat)

    train_loss = custom_loss_function(pred[train_idx], labels[train_idx])
    val_loss = custom_loss_function(pred[val_idx], labels[val_idx])
    test_loss = custom_loss_function(pred[test_idx], labels[test_idx])

    return (
        evaluator(pred[train_idx], labels[train_idx]),
        evaluator(pred[val_idx], labels[val_idx]),
        evaluator(pred[test_idx], labels[test_idx]),
        train_loss,
        val_loss,
        test_loss,
        pred,
    )

def save_pred(pred, run_num, kd_dir):
    if not os.path.exists(kd_dir):
        os.makedirs(kd_dir)
    fname = os.path.join(kd_dir, 'best_pred_run{}.pt'.format(run_num))
    torch.save(pred.cpu(), fname)


def run(args, graph, labels, train_idx, val_idx, test_idx,
        evaluator, n_running):

    evaluator_wrapper = lambda pred, labels: evaluator.eval(
        {"y_pred": pred.argmax(dim=-1, keepdim=True), "y_true": labels}
    )["acc"]
    
    
    # kd mode
    mode = args.mode

    # define model and optimizer
    model = gen_model(args).to(device)
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.wd)
    
    # training loop
    total_time = 0
    best_val_acc, final_test_acc, best_val_loss = 0, 0, float("inf")
    final_pred = None

    accs, train_accs, val_accs, test_accs = [], [], [], []
    losses, train_losses, val_losses, test_losses = [], [], [], []
    bad_counter = 0
    for epoch in range(1, args.n_epochs + 1):
        tic = time.time()
        if mode == 'student':
            teacher_output = torch.load('./{}/best_pred_run{}.pt'.format(
              args.kd_dir,
              n_running)).cpu().cuda()
        else:
            teacher_output = None

        adjust_learning_rate(optimizer, args.lr, epoch)
        
        acc, loss = train(args, model, graph, labels, train_idx,
                          val_idx, test_idx, optimizer, evaluator_wrapper,
                          mode=mode, teacher_output=teacher_output)

        train_acc, val_acc, test_acc, train_loss, val_loss, test_loss, pred = evaluate(
            args, model, graph, labels, train_idx, val_idx, test_idx, evaluator_wrapper
        )

        toc = time.time()
        total_time += toc - tic

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            final_test_acc = test_acc
            final_pred = pred
            bad_counter = 0
            
            if mode == 'teacher':
                save_pred(final_pred, n_running, args.kd_dir)
                
            save_info = {'optimizer':optimizer.state_dict(),
                        'model':model.state_dict(),
                         'att0':model.convs[0].attn_l,
                         'att1':model.convs[1].attn_l
                        }
        else:
            bad_counter += 1
        if epoch ==300:
            bad_counter=0
        if epoch>300 and bad_counter >= 150:
                break

        if epoch == args.n_epochs or epoch % args.log_every == 0:
            print(
                f"Run: {n_running}/{args.n_runs}, Epoch: {epoch}/{args.n_epochs}, Average epoch time: {total_time / epoch:.2f}\n"
                f"Loss: {loss:.4f}, Acc: {acc:.4f}\n"
                f"Train/Val/Test loss: {train_loss:.4f}/{val_loss:.4f}/{test_loss:.4f}\n"
                f"Train/Val/Test/Best val/Final test acc: {train_acc:.4f}/{val_acc:.4f}/{test_acc:.4f}/{best_val_acc:.4f}/{final_test_acc:.4f}"
            )

        for l, e in zip(
            [accs, train_accs, val_accs, test_accs, losses, train_losses, val_losses, test_losses],
            [acc, train_acc, val_acc, test_acc, loss, train_loss, val_loss, test_loss],
        ):
            l.append(e)

    print("*" * 50)
    print(f"Best val acc: {best_val_acc}, Final test acc: {final_test_acc}")
    print("*" * 50)

  

    if args.savn_node_feats + n_classese_pred:
        os.makedirs("./output", exist_ok=True)
        torch.save(F.softmax(final_pred, dim=1), f"./output/{n_running}.pt")

    return best_val_acc, final_test_acc


def main():
    # load data & preprocess
    graph, labels, train_idx, val_idx, test_idx, evaluator = load_data(dataset,args)
    graph = preprocess(graph)
    graph, labels, train_idx, val_idx, test_idx = map(lambda x: x.to(device), (graph, labels, train_idx, val_idx, test_idx))


    # run
    val_accs, test_accs = [], []

    for i in range(args.n_runs):
        val_acc, test_acc = run(args, graph, labels, train_idx, val_idx,test_idx, evaluator, i + 1)
        val_accs.append(val_acc)
        test_accs.append(test_acc)

    print(f"Runned {args.n_runs} times")
    print("Val Accs:", val_accs)
    print("Test Accs:", test_accs)
    print(f"Average val accuracy: {np.mean(val_accs)} ± {np.std(val_accs)}")
    print(f"Average test accuracy: {np.mean(test_accs)} ± {np.std(test_accs)}")


if __name__ == "__main__":
    main()
