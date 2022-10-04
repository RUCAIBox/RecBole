## Time and memory cost of knowledge-based recommendation models 

### Datasets information:

| Dataset             |   #User |  #Item | #Interaction | Sparsity | #Entity | #Relation | #Triple |
| ------------------- | ------: | -----: | -----------: | -------: | ------: | --------: | ------: |
| MovieLens-1m        |   6,034 |  3,096 |      832,104 | 95.5458% |  10,234 |        54 | 206,844 |
| Amazon-Books (2018) | 160,383 |  4,000 |      344,601 | 99.9463% |  10,302 |        22 | 152,882 |
| Lastfm-track        |  45,987 | 38,439 |    3,118,764 | 99.8236% |  94,104 |        12 | 774,508 |

### Device information

```
OS:                   Linux
Python Version:       3.8.10
PyTorch Version:      1.8.1
cudatoolkit Version:  10.1
GPU:                  TITAN V（12GB）
Machine Specs:        14 CPU machine, 256GB RAM
```

### 1) MovieLens-1m dataset:

#### Time and memory cost on MovieLens-1m dataset:

| Method    | Training Time (sec/epoch) | Evaluation Time (sec/epoch) | Training GPU Memory (GB) | Evaluation GPU Memory (GB) |
| --------- | ------------------------- | --------------------------- | ------------------------ | -------------------------- |
| CFKG      | 3.03                      | 0.30                        | 0.05                     | 18.15                      |
| CKE       | 9.32                      | 0.22                        | 0.29                     | 0.58                       |
| KGAT      | 9.39                      | 0.18                        | 0.46                     | 1.74                       |
| KGCN      | 5.61                      | 2.07                        | 0.07                     | 0.19                       |
| KGIN      | 53.89                     | 0.48                        | 0.35                     | 0.42                       |
| KGNNLS    | 10.63                     | 2.2                         | 0.22                     | 0.22                       |
| KTUP      | 6.57                      | 2.31                        | 0.16                     | 0.16                       |
| MCCLK     | 667.11                    | 2.87                        | 3.55                     | 3.55                       |
| MKR       | 18.07                     | 19.30                       | 0.58                     | 0.58                       |
| RippleNet | 81.82                     | 0.69                        | 6.67                     | 34.12                      |

#### Config file of MovieLens-1m dataset:

```
# dataset config
field_separator: "\t"
seq_separator: " "
USER_ID_FIELD: user_id
ITEM_ID_FIELD: item_id
RATING_FIELD: rating
HEAD_ENTITY_ID_FIELD: head_id
TAIL_ENTITY_ID_FIELD: tail_id
RELATION_ID_FIELD: relation_id
ENTITY_ID_FIELD: entity_id
NEG_PREFIX: neg_
LABEL_FIELD: label
load_col:
    inter: [user_id, item_id, rating]
    kg: [head_id, relation_id, tail_id]
    link: [item_id, entity_id]

# data filtering for interactions
val_interval:
    rating: "[3,inf)"    
unused_col: 
    inter: [rating]

user_inter_num_interval: "[10,inf)"
item_inter_num_interval: "[10,inf)"

# data preprocessing for knowledge graph triples
kg_reverse_r: True
entity_kg_num_interval: "[5,inf)"
relation_kg_num_interval: "[5,inf)"

# training and evaluation
epochs: 500
train_batch_size: 4096
eval_batch_size: 40960000
valid_metric: NDCG@10
train_neg_sample_args: 
    distribution: uniform
    sample_num: 1
    dynamic: False

# model
embedding_size: 64
```

Other parameters (including model parameters) are default value. 

### 2）Amazon-Books (2018) dataset:

#### Time and memory cost on Amazon-Books (2018) dataset:

| Method    | Training Time (sec/epoch) | Evaluation Time (sec/epoch) | Training GPU Memory (GB) | Evaluation GPU Memory (GB) |
| --------- | ------------------------- | --------------------------- | ------------------------ | -------------------------- |
| CFKG      | 66.53                     | 1.92                        | 16.79                    | 16.79                      |
| CKE       | 64.80                     | 2.09                        | 1.1                      | 1.1                        |
| KGAT      | 2.42                      | 0.56                        | 1.95                     | 1.95                       |
| KGCN      | 1.43                      | 6.74                        | 0.36                     | 0.36                       |
| KGIN      | 18.85                     | 2.24                        | 1.05                     | 1.67                       |
| KGNNLS    | 1.89                      | 6.92                        | 0.36                     | 0.36                       |
| KTUP      | 1.79                      | 1.33                        | 36.6                     | 36.6                       |
| MCCLK     | 243.87                    | 7.08                        | 4.06                     | 4.37                       |
| MKR       | 3.27                      | 15.02                       | 0.82                     | 0.82                       |
| RippleNet | 6.36                      | 0.81                        | 25.54                    | 25.54                      |

#### Config file of Amazon-Books (2018) dataset:

```
# dataset config
field_separator: "\t"
seq_separator: " "
USER_ID_FIELD: user_id
ITEM_ID_FIELD: item_id
RATING_FIELD: rating
HEAD_ENTITY_ID_FIELD: head_id
TAIL_ENTITY_ID_FIELD: tail_id
RELATION_ID_FIELD: relation_id
ENTITY_ID_FIELD: entity_id
NEG_PREFIX: neg_
LABEL_FIELD: label
load_col:
    inter: [user_id, item_id, rating]
    kg: [head_id, relation_id, tail_id]
    link: [item_id, entity_id]

# data filtering for interactions
val_interval:
    rating: "[3,inf)"    
unused_col: 
    inter: [rating]

user_inter_num_interval: "[10,inf)"
item_inter_num_interval: "[10,inf)"

# data preprocessing for knowledge graph triples
kg_reverse_r: True
entity_kg_num_interval: "[5,inf)"
relation_kg_num_interval: "[5,inf)"

# training and evaluation
epochs: 500
train_batch_size: 4096
eval_batch_size: 40960000
valid_metric: NDCG@10
train_neg_sample_args: 
    distribution: uniform
    sample_num: 1
    dynamic: False

# model
embedding_size: 64
```

Other parameters (including model parameters) are default value. 

### 3）Lastfm-track dataset:

#### Time and memory cost on Lastfm-track dataset:

| Method    | Training Time (sec/epoch) | Evaluation Time (sec/epoch) | Training GPU Memory (GB) | Evaluation GPU Memory (GB) |
| --------- | ------------------------- | --------------------------- | ------------------------ | -------------------------- |
| CFKG      | 48.34                     | 583.24                      | 0.25                     | 0.25                       |
| CKE       | 68.14                     | 1.78                        | 0.48                     | 1.1                        |
| KGAT      | 161.36                    | 2.01                        | 5.89                     | 5.89                       |
| KGCN      | 378.91                    | 2186.54                     | 0.38                     | 0.38                       |
| KGIN      | 660.62                    | 2.54                        | 1.53                     | 2.14                       |
| KGNNLS    | 66.58                     | 264.76                      | 0.92                     | 0.92                       |
| KTUP      | 120.53                    | 457.34                      | 0.32                     | 0.32                       |
| MCCLK     | -                         | -                           | -                        | -                          |
| MKR       | 30.93                     | 550.61                      | 0.52                     | 1.7                        |
| RippleNet | 275.47                    | 131.51                      | 6.82                     | 6.82                       |

#### Config file of Lastfm-track  dataset:

```
# dataset config
field_separator: "\t"
seq_separator: " "
USER_ID_FIELD: user_id
ITEM_ID_FIELD: item_id
RATING_FIELD: rating
HEAD_ENTITY_ID_FIELD: head_id
TAIL_ENTITY_ID_FIELD: tail_id
RELATION_ID_FIELD: relation_id
ENTITY_ID_FIELD: entity_id
NEG_PREFIX: neg_
LABEL_FIELD: label
load_col:
    inter: [user_id, item_id, rating]
    kg: [head_id, relation_id, tail_id]
    link: [item_id, entity_id]

# data filtering for interactions
val_interval:
    rating: "[3,inf)"    
unused_col: 
    inter: [rating]

user_inter_num_interval: "[10,inf)"
item_inter_num_interval: "[10,inf)"

# data preprocessing for knowledge graph triples
kg_reverse_r: True
entity_kg_num_interval: "[5,inf)"
relation_kg_num_interval: "[5,inf)"

# training and evaluation
epochs: 500
train_batch_size: 4096
eval_batch_size: 40960000
valid_metric: NDCG@10
train_neg_sample_args: 
    distribution: uniform
    sample_num: 1
    dynamic: False

# model
embedding_size: 64
```

Other parameters (including model parameters) are default value. 

