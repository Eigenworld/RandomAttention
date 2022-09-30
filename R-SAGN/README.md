# Irregular Message Passing Networks

This repository is an implementation of Generalization Test of Random Attentions in "Irregular Message Passing Networks"(Knowledge-Based Systems). The proposed concepts in this paper challenge the widely used attention mechanisms in the research community. Our main observation is that we do not need to learn the graph attention! Just replace it  with a $L_2$ normalized random vector! Previous theories about Graph Attention Mechnism may be wrong!

**[Two Models are tested]:**

**RevGAT**: RevGAT generalizes reversible residual connections to grouped reversible residual connections for GAT.

**SAGN:** SAGN keeps the message passing linear and hypothesizes that the intermediate representations are informative. Diagonal attentions are used to integrate these multi-hop node features into the final prediction.

**[Two Datasets]:** **Ogbn-arxiv**, **Ogbn-products**

**[Our Models]:**

**R-RevGAT**: R-RevGAT is simply constructed by replacing the learnable attentions in RevGAT with normalized random attentions.

**R-SAGN**: Diagonal attentions in SAGN are replaced with normalized random attentions.

**[Model differences]:**

1. Attention in R-RevGAT, RevGAT or GAT is constrained by the graph topology.
2. Attention in R-SAGN is used as a feature selection tool. (enlarges the application of random attentions.)

**[Ruminations on Graph Attention Mechnism]:**

1. The success of the attention mechanism has long been attributed to its adaptive information selection. Somewhat shockingly, however, this is not the case for graph attention. We could incorporate the node feature information into the edge weights in a random way, without considering which neighbor node is more important.
2.  The effectiveness of message passing is not uniquely tied to the well-designed propagation weight. Some “bad” edge weights could achieve a similar effect.

**[GIANT-XRT+R-RevGAT+KD on ogbn-arxiv]:** 

**Notation:** 

1. set your **data_root_dir** (for ogbn-arxiv) and **pretrain_path** (for giant xrt embedding) in **main.py**.
2. KD: set args.mode='teacher' -> args.mode='student' in **main.py**.

**command:** python main.py

-------

-----

**[GIANT-XRT+R-SAGN+SCR+C&S on ogbn-products]:**

**command:** 

1. python pre_processing.py --num_hops 3 --dataset ogbn-products --giant_path " your path"
2. python main.py --method R_SAGN --stages 300 --train-num-epochs 0 --input-drop 0.2 --att-drop 0.4 --pre-process --residual --dataset ogbn-products --num-runs 10 --eval 10 --batch_size 200000 --patience 300 --tem 0.5 --lam 0.5 --ema --mean_teacher --ema_decay 0.0 --lr 0.001 --adap --gap 20 --warm_up 100 --top 0.85 --down 0.8 --kl --kl_lam 0.2 --hidden 256 --zero-inits --dropout 0.5 --num-heads 1  --label-drop 0.5  --mlp-layer 1 --num_hops 3 --label_num_hops 9 --giant
3. python post_processing.py --file_name '.pt file in your output file' --correction_alpha 0.4003159464410826 --smoothing_alpha 0.49902336390254404

**Notation:** you may encounter the following warnings, and leave it alone.

R-SCR-main/ogbn-products/utils.py:357: UserWarning: This overload of add_ is deprecated:
add_(Number alpha, Tensor other) Consider using one of the following signatures instead:add_(Tensor other, *, Number alpha) (Triggered internally at  /opt/conda/conda-bld/pytorch_1646755849709/work/torch/csrc/utils/python_arg_parser.cpp:1055.) mean_param.data.mul_(alpha).add_(1 - alpha, param.data)

# Results

**[GIANT-XRT+R-RevGAT on ogbn-arxiv]:** Number of params: 1500712

| run  | val    | test   |
| ---- | ------ | ------ |
| 1    | 0.7708 | 0.7603 |
| 2    | 0.7701 | 0.7637 |
| 3    | 0.7702 | 0.7557 |
| 4    | 0.7682 | 0.7608 |
| 5    | 0.7702 | 0.7622 |
| 6    | 0.7712 | 0.7600 |
| 7    | 0.7702 | 0.7622 |
| 8    | 0.7676 | 0.7608 |
| 9    | 0.7686 | 0.7583 |
| 10   | 0.7687 | 0.7597 |

```
Val: 0.7696 +- 0.0011
Test:0.7604 +- 0.0021
```

\[GIANT-XRT+R-RevGAT**+KD on ogbn-arxiv\]:** 

| run  | val    | test   |
| ---- | ------ | ------ |
| 1    | 0.7693 | 0.7648 |
| 2    | 0.7687 | 0.7639 |
| 3    | 0.7678 | 0.7631 |
| 4    | 0.7690 | 0.7636 |
| 5    | 0.7705 | 0.7627 |
| 6    | 0.7689 | 0.7638 |
| 7    | 0.7695 | 0.7639 |
| 8    | 0.7708 | 0.7636 |
| 9    | 0.7697 | 0.7628 |
| 10   | 0.7675 | 0.7631 |

```
Val: 0.7692 +- 0.0010
Test:0.7635 +- 0.0006
```

---

**[GIANT-XRT+R-SAGN+SCR+C&S on ogbn-products]: **Number of params: 1154142

| run  | val    | test   |
| ---- | ------ | ------ |
| 1    | 0.9364 | 0.8657 |
| 2    | 0.9359 | 0.8657 |
| 3    | 0.9361 | 0.8666 |
| 4    | 0.9360 | 0.8665 |
| 5    | 0.9357 | 0.8664 |
| 6    | 0.9361 | 0.8659 |
| 7    | 0.9361 | 0.8663 |
| 8    | 0.9360 | 0.8666 |
| 9    | 0.9355 | 0.8668 |
| 10   | 0.9358 | 0.8660 |

```
Val: 0.9360 +- 0.0002
Test: 0.8662 +- 0.0004
```

​                                                                   



\[GIANT-XRT+R-SAGN+SCR**+C&S on ogbn-products\]:**

| run  | val    | test   |
| ---- | ------ | ------ |
| 1    | 0.9366 | 0.8677 |
| 2    | 0.9367 | 0.8677 |
| 3    | 0.9361 | 0.8684 |
| 4    | 0.9360 | 0.8688 |
| 5    | 0.9359 | 0.8688 |
| 6    | 0.9365 | 0.8680 |
| 7    | 0.9362 | 0.8685 |
| 8    | 0.9367 | 0.8689 |
| 9    | 0.9364 | 0.8681 |
| 10   | 0.9365 | 0.8692 |

```
Val: 0.9365 +- 0.0003
Test: 0.8684 +- 0.0005
```











