## Time and memory cost of sequential recommendation models 

### Datasets information:

| Dataset    | #User   | #Item   | #Interaction | Sparsity |
| ---------- | ------: | ------: | -----------: | -------: |
| ml-1m      |  6,034  |  3,124  |     834,449   |  0.9557 |
| Amazon-Books | 40,550   | 31,094   |     1,181,294   |  0.9991  |
| Yelp2022   | 72,488  | 43,749  |   2,043,402  |  0.9994 |

### Device information

```
OS:                   Linux
Python Version:       3.8.10
PyTorch Version:      1.8.1
cudatoolkit Version:  10.1
GPU:                  TITAN V（12GB）
Machine Specs:        14 CPU machine, 256GB RAM
```

### 1) ml-1m dataset:

#### Time and memory cost on ml-1m dataset:

| Method    | Training Time (sec/epoch) | Evaluation Time (sec/epoch) | Training GPU Memory (GB) | Evaluation GPU Memory (GB) |
| --------- | ------------------------- | --------------------------- | ------------------------ | -------------------------- |
| BERT4Rec  | 77.75                     | 1.35                        | 4.81                     | 5.10                       |
| Caser     | 569.90                    | 0.46                        | 0.34                     | 0.42                       |
| CORE      | 44.04                     | 0.39                        | 4.03                     | 5.18                       |
| FDSA      | 89.30                     | 0.51                        | 7.47                     | 8.33                       |
| FOSSIL    | 6.31                      | 0.30                        | 0.57                     | 0.71                       |
| FPMC      | 6.08                      | 0.29                        | 0.04                     | 0.25                       |
| GCSAN     | 693.87                    | 5.72                        | 4.66                     | 6.25                       |
| GRU4Rec   | 18.43                     | 0.42                        | 1.31                     | 1.31                       |
| GRU4RecF  | 39.09                     | 0.45                        | 2.36                     | 2.36                       |
| HGN       | 10.71                     | 0.41                        | 0.38                     | 0.74                       |
| HRM       | 14.82                     | 0.73                        | 0.19                     | 0.26                       |
| LightSANs | 76.18                     | 0.57                        | 3.73                     | 4.89                       |
| NARM      | 22.73                     | 0.43                        | 1.38                     | 1.38                       |
| NextItNet | 819.55                    | 3.43                        | 3.41                     | 4.10                       |
| NPE       | 4.79                      | 0.40                        | 0.18                     | 0.25                       |
| RepeatNet | 848.43                    | 6.51                        | 5.98                     | 9.50                       |
| S3Rec     | 27.98                     | 0.33                        | 4.01                     | 5.16                       |
| SASRec    | 69.37                     | 0.57                        | 4.01                     | 5.16                       |
| SASRecF   | 80.04                     | 0.60                        | 4.68                     | 5.54                       |
| SHAN      | 14.73                     | 0.70                        | 0.29                     | 0.51                       |
| SINE      | 15.25                     | 0.31                        | 1.37                     | 2.08                       |
| SRGNN     | 655.62                    | 5.73                        | 0.99                     | 1.93                       |
| STAMP     | 10.18                     | 0.41                        | 0.43                     | 1.00                       |
| TransRec  | 10.49                     | 0.19                        | 0.03                     | 9.23                       |

#### Config file of ml-1m dataset:

```
# dataset config
field_separator: "\t"
seq_separator: " "
USER_ID_FIELD: user_id
ITEM_ID_FIELD: item_id
RATING_FIELD: rating
TIME_FIELD: timestamp
NEG_PREFIX: neg_
ITEM_LIST_LENGTH_FIELD: item_length
LIST_SUFFIX: _list
MAX_ITEM_LIST_LENGTH: 50
POSITION_FIELD: position_id
load_col:
    inter: [user_id, item_id, rating, timestamp]

# data filtering for interactions
val_interval:
    rating: "[3,inf)"    
unused_col: 
    inter: [rating]

user_inter_num_interval: "[10,inf)"
item_inter_num_interval: "[10,inf)"

# training and evaluation
epochs: 500
train_batch_size: 4096
eval_batch_size: 40960000
valid_metric: NDCG@10
eval_args:
    split: {'LS': 'valid_and_test'}
    mode: full
    order: TO

# disable negative sampling
train_neg_sample_args: ~

# model
embedding_size: 64
```

Other parameters (including model parameters) are default value. 

**NOTE :** 

1) For FPMC and TransRec model,  `neg_sampling`  should be  `{'uniform': 1}` .

2) For SASRecF, GRU4RecF and FDSA,   `load_col` should as below:

```
load_col:
  inter: [user_id, item_id, rating, timestamp]
  item: [item_id, genre]
```

3) For KSR and GRU4RecKG, you should prepare pretrained knowledge graph embedding.

### 2）Amazon-Books dataset:

#### Time and memory cost on Amazon-books dataset:

| Method    | Training Time (sec/epoch) | Evaluation Time (sec/epoch) | Training GPU Memory (GB) | Evaluation GPU Memory (GB) |
| --------- | ------------------------- | --------------------------- | ------------------------ | -------------------------- |
| BERT4Rec  | 254.92                    | 4.94                        | 4.74                     | 4.74                       |
| Caser     | 3253.50                   | 1.34                        | 0.56                     | 2.46                       |
| CORE      | 131.23                    | 1.85                        | 1.37                     | 5.00                       |
| FDSA      | 148.74                    | 1.76                        | 2.16                     | 4.84                       |
| FOSSIL    | 19.80                     | 0.40                        | 0.47                     | 2.37                       |
| FPMC      | 15.44                     | 0.47                        | 0.22                     | 3.54                       |
| GCSAN     | 528.84                    | 19.14                       | 1.48                     | 4.46                       |
| GRU4Rec   | 24.51                     | 0.41                        | 0.66                     | 2.56                       |
| GRU4RecF  | 37.67                     | 0.50                        | 0.88                     | 2.98                       |
| HGN       | 39.80                     | 0.71                        | 0.63                     | 2.54                       |
| HRM       | 22.00                     | 0.67                        | 0.55                     | 2.45                       |
| LightSANs | 89.64                     | 0.60                        | 1.29                     | 3.97                       |
| NARM      | 24.91                     | 0.49                        | 0.76                     | 2.76                       |
| NextItNet | 298.14                    | 4.44                        | 1.31                     | 3.31                       |
| NPE       | 13.13                     | 0.35                        | 0.57                     | 2.47                       |
| S3Rec     | 1.37                      | 4.05                        | 68.71                    | 0.59                       |
| SASRec    | 78.19                     | 1.88                        | 1.36                     | 4.04                       |
| SASRecF   | 81.50                     | 1.09                        | 1.42                     | 4.34                       |
| SHAN      | 37.32                     | 0.83                        | 0.58                     | 2.47                       |
| SINE      | 42.13                     | 0.50                        | 0.80                     | 3.18                       |
| SRGNN     | 478.05                    | 16.83                       | 0.76                     | 3.20                       |
| STAMP     | 21.46                     | 0.38                        | 0.52                     | 2.53                       |
| TransRec  | 44.64                     | 10.79                       | 0.12                     | 7.75                       |

#### Config file of Amazon-books dataset:

```
# dataset config
field_separator: "\t"
seq_separator: " "
USER_ID_FIELD: user_id
ITEM_ID_FIELD: item_id
RATING_FIELD: rating
TIME_FIELD: timestamp
NEG_PREFIX: neg_
ITEM_LIST_LENGTH_FIELD: item_length
LIST_SUFFIX: _list
MAX_ITEM_LIST_LENGTH: 50
POSITION_FIELD: position_id
load_col:
    inter: [user_id, item_id, rating, timestamp]

# data filtering for interactions
val_interval:
    rating: "[3,inf)"    
unused_col: 
    inter: [rating]

user_inter_num_interval: "[10,inf)"
item_inter_num_interval: "[10,inf)"

# training and evaluation
epochs: 500
train_batch_size: 4096
eval_batch_size: 40960000
valid_metric: NDCG@10
eval_args:
    split: {'LS': 'valid_and_test'}
    mode: full
    order: TO

# disable negative sampling
train_neg_sample_args: ~

# model
embedding_size: 64
```

Other parameters (including model parameters) are default value. 

**NOTE :** 

1) For FPMC and TransRec model,  `neg_sampling`  should be  `{'uniform': 1}` .

2) For SASRecF, GRU4RecF and FDSA,   `load_col` should as below:

```
load_col:
   inter: [session_id, item_id, timestamp]
   item: [item_id, item_category]
```

3) For KSR and GRU4RecKG, you should prepare pretrained knowledge graph embedding.

### 3）Yelp2022 dataset:

#### Time and memory cost on Yelp2022 dataset:

| Method    | Training Time (sec/epoch) | Evaluation Time (sec/epoch) | Training GPU Memory (GB) | Evaluation GPU Memory (GB) |
| --------- | ------------------------- | --------------------------- | ------------------------ | -------------------------- |
| BERT4Rec  | 270.80                    | 20.12                       | 6.31                     | 6.31                       |
| CORE      | 103.14                    | 1.25                        | 6.12                     | 6.12                       |
| FDSA      | 203.31                    | 1.50                        | 2.37                     | 2.70                       |
| FOSSIL    | 26.14                     | 0.79                        | 2.41                     | 3.74                       |
| FPMC      | 13.05                     | 0.72                        | 5.01                     | 5.01                       |
| GCSAN     | 1396.14                   | 57.13                       | 4.38                     | 4.72                       |
| GRU4Rec   | 22.95                     | 0.95                        | 3.84                     | 5.18                       |
| GRU4RecF  | 46.41                     | 0.98                        | 1.09                     | 1.42                       |
| HGN       | 20.12                     | 0.72                        | 2.54                     | 3.88                       |
| HRM       | 34.11                     | 1.41                        | 2.28                     | 3.61                       |
| LightSANs | 152.20                    | 1.55                        | 5.83                     | 7.16                       |
| NARM      | 41.23                     | 0.98                        | 3.54                     | 4.88                       |
| NPE       | 18.48                     | 0.90                        | 2.29                     | 3.62                       |
| SASRec    | 62.29                     | 1.02                        | 6.10                     | 7.44                       |
| SASRecF   | 99.60                     | 1.12                        | 1.62                     | 1.95                       |
| SINE      | 27.23                     | 0.74                        | 3.35                     | 4.69                       |
| SRGNN     | 847.11                    | 20.12                       | 20.17                    | 21.33                      |
| STAMP     | 19.08                     | 0.68                        | 2.60                     | 3.94                       |
| TransRec  | 57.48                     | 8.41                        | 0.18                     | 10.91                      |

#### Config file of Yelp2022 dataset:

```
# dataset config
field_separator: "\t"
seq_separator: " "
USER_ID_FIELD: user_id
ITEM_ID_FIELD: item_id
RATING_FIELD: rating
TIME_FIELD: timestamp
NEG_PREFIX: neg_
ITEM_LIST_LENGTH_FIELD: item_length
LIST_SUFFIX: _list
MAX_ITEM_LIST_LENGTH: 50
POSITION_FIELD: position_id
load_col:
    inter: [user_id, item_id, rating, timestamp]

# data filtering for interactions
val_interval:
    rating: "[3,inf)"    
unused_col: 
    inter: [rating]

user_inter_num_interval: "[10,inf)"
item_inter_num_interval: "[10,inf)"

# training and evaluation
epochs: 500
train_batch_size: 4096
eval_batch_size: 40960000
valid_metric: NDCG@10
eval_args:
    split: {'LS': 'valid_and_test'}
    mode: full
    order: TO

# disable negative sampling
train_neg_sample_args: ~

# model
embedding_size: 64
```

Other parameters (including model parameters) are default value. 

**NOTE :** 

1) For FPMC and TransRec model,  `neg_sampling`  should be  `{'uniform': 1}` .

2) For SASRecF, GRU4RecF and FDSA,   `load_col` should as below:

```
load_col:
  inter: [user_id, business_id, stars, date]
  item: [business_id, categories]
```

3) For KSR and GRU4RecKG, you should prepare pretrained knowledge graph embedding.
