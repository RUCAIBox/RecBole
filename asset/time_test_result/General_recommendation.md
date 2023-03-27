## Time  and memory cost of general recommendation models 

### Datasets information:

| Dataset             |  #User |  #Item | #Interaction | Sparsity |
| ------------------- | -----: | -----: | -----------: | -------: |
| MovieLens-1m        |  6,034 |  3,124 |      834,449 | 95.5733% |
| Amazon-Books (2018) | 40,550 | 31,094 |    1,181,294 | 99.9063% |
| Yelp2022            | 72,488 | 43,749 |    2,043,402 | 99.9356% |

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

| Method      | Training Time (sec/epoch) | Evaluation Time (sec/epoch) | Training GPU Memory (GB) | Evaluation GPU Memory (GB) |
| ----------- | ------------------------- | --------------------------- | ------------------------ | -------------------------- |
| ADMMSLIM    | 3.81                      | 6.16                        | 0                        | 0                          |
| BPR         | 1.73                      | 0.27                        | 0.31                     | 0.31                       |
| CDAE        | 0.03                      | 0.19                        | 1.16                     | 1.16                       |
| ConvNCF     | 11.11                     | 10.61                       | 0.72                     | 3.82                       |
| DGCF        | 99.88                     | 0.75                        | 4.14                     | 4.28                       |
| DMF         | 3.86                      | 0.99                        | 0.97                     | 0.97                       |
| EASE        | 1.65                      | 3.19                        | 0                        | 0                          |
| ENMF        | 0.15                      | 0.33                        | 6.26                     | 6.26                       |
| FISM        | 24.33                     | 3.12                        | 3.27                     | 3.27                       |
| GCMC        | 5.84                      | 0.24                        | 0.67                     | 0.67                       |
| ItemKNN     | 1.56                      | 7.24                        | 0                        | 0.3                        |
| LightGCN    | 4.10                      | 0.22                        | 0.44                     | 0.44                       |
| LINE        | 2.55                      | 0.24                        | 0.34                     | 0.34                       |
| MacridVAE   | 0.13                      | 0.26                        | 2.45                     | 2.45                       |
| MultiDAE    | 0.05                      | 0.23                        | 0.94                     | 0.94                       |
| MultiVAE    | 0.04                      | 0.23                        | 0.94                     | 0.94                       |
| NAIS        | 50.41                     | 12.86                       | 9.13                     | 9.13                       |
| NCE-PLRec   | 1.75                      | 0.25                        | 0.41                     | 0.41                       |
| NCL         | 6.51                      | 0.24                        | 0.61                     | 0.61                       |
| NeuMF       | 3.30                      | 1.49                        | 0.15                     | 0.15                       |
| NGCF        | 5.76                      | 0.28                        | 0.47                     | 0.47                       |
| NNCF        | 6.40                      | 6.69                        | 1.08                     | 1.08                       |
| Pop         | 1.51                      | 0.22                        | 0                        | 0.38                       |
| RaCT        | 0.04                      | 0.17                        | 0.74                     | 1.02                       |
| RecVAE      | 0.20                      | 0.23                        | 1.11                     | 1.11                       |
| SGL         | 14.93                     | 0.26                        | 0.67                     | 0.67                       |
| SimpleX     | 2.60                      | 0.19                        | 0.6                      | 0.6                        |
| SLIMElastic | 1.43                      | 1.81                        | 0                        | 0                          |
| SpectralCF  | 6.72                      | 0.33                        | 0.46                     | 0.46                       |

#### Config file of MovieLens-1m dataset:

```
# dataset config
field_separator: "\t"
seq_separator: " "
USER_ID_FIELD: user_id
ITEM_ID_FIELD: item_id
RATING_FIELD: rating
NEG_PREFIX: neg_
LABEL_FIELD: label
load_col:
    inter: [user_id, item_id, rating]

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

| Method      | Training Time (sec/epoch) | Evaluation Time (sec/epoch) | Training GPU Memory (GB) | Evaluation GPU Memory (GB) |
| ----------- | ------------------------- | --------------------------- | ------------------------ | -------------------------- |
| ADMMSLIM    | -                         | -                           | -                        | -                          |
| BPR         | 3.93                      | 2.80                        | 0.74                     | 0.74                       |
| CDAE        | 0.84                      | 2.58                        | 4.71                     | 4.71                       |
| ConvNCF     | -                         | -                           | -                        | -                          |
| DGCF        | 221.81                    | 3.65                        | 6.05                     | 6.05                       |
| DMF         | 10.29                     | 1.53                        | 2.77                     | 2.77                       |
| EASE        | 2.99                      | 72.01                       | 0                        | 0                          |
| ENMF        | 1.22                      | 2.89                        | 10.5                     | 10.5                       |
| FISM        | 25.23                     | 34.85                       | 8.79                     | 8.79                       |
| GCMC        | 59.54                     | 6.47                        | 2.81                     | 2.81                       |
| ItemKNN     | 3.62                      | 52.00                       | 0                        | 0.62                       |
| LightGCN    | 13.54                     | 3.21                        | 1.01                     | 1.01                       |
| LINE        | 6.06                      | 3.07                        | 0.86                     | 0.86                       |
| MacridVAE   | -                         | -                           | -                        | -                          |
| MultiDAE    | 1.38                      | 2.01                        | 3.5                      | 3.5                        |
| MultiVAE    | 1.37                      | 2.79                        | 3.51                     | 3.51                       |
| NAIS        | -                         | -                           | -                        | -                          |
| NCE-PLRec   | 1.89                      | 1.54                        | 1.03                     | 1.03                       |
| NCL         | 64.89                     | 756.20                      | 2.72                     | 2.72                       |
| NeuMF       | -                         | -                           | -                        | -                          |
| NGCF        | 22.63                     | 3.28                        | 1.34                     | 1.34                       |
| NNCF        | -                         | -                           | -                        | -                          |
| Pop         | 3.98                      | 2.63                        | 0.77                     | 0.77                       |
| RaCT        | 1.05                      | 2.41                        | 6.47                     | 6.47                       |
| RecVAE      | 6.00                      | 2.77                        | 3.59                     | 3.59                       |
| SGL         | 64.89                     | 2.49                        | 2.93                     | 2.93                       |
| SimpleX     | 5.90                      | 2.36                        | 0.85                     | 0.85                       |
| SLIMElastic | 2.37                      | 20.34                       | 0                        | 0                          |
| SpectralCF  | 25.61                     | 2.59                        | 1.2                      | 1.2                        |

#### Config file of Amazon-Books (2018) dataset:

```
# dataset config
field_separator: "\t"
seq_separator: " "
USER_ID_FIELD: user_id
ITEM_ID_FIELD: item_id
RATING_FIELD: rating
NEG_PREFIX: neg_
LABEL_FIELD: label
load_col:
    inter: [user_id, item_id, rating]

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
train_neg_sample_args: 
    distribution: uniform
    sample_num: 1
    dynamic: False

# model
embedding_size: 64
```

Other parameters (including model parameters) are default value. 

### 3) Yelp2022 dataset:

#### Time and memory cost on Yelp2022 dataset:

| Method      | Training Time (sec/epoch) | Evaluation Time (sec/epoch) | Training GPU Memory (GB) | Evaluation GPU Memory (GB) |
| ----------- | ------------------------- | --------------------------- | ------------------------ | -------------------------- |
| ADMMSLIM    | 137.43                    | 3561.60                     | 0                        | 0                          |
| BPR         | 6.20                      | 5.05                        | 0.17                     | 0.79                       |
| CDAE        | 1.95                      | 4.43                        | 6.99                     | 6.99                       |
| ConvNCF     | -                         | -                           | -                        | -                          |
| DGCF        | -                         | -                           | -                        | -                          |
| DMF         | 33.40                     | 5.09                        | 4.45                     | 4.45                       |
| EASE        | -                         | -                           | -                        | -                          |
| ENMF        | 2.68                      | 21.13                       | 7.92                     | 7.92                       |
| FISM        | -                         | -                           | -                        | -                          |
| GCMC        | 157.97                    | 18.47                       | 2.57                     | 3.49                       |
| ItemKNN     | 5.52                      | 79.59                       | 0                        | 0.62                       |
| LightGCN    | 34.03                     | 4.54                        | 0.59                     | 1.21                       |
| LINE        | 9.84                      | 3.56                        | 0.32                     | 0.93                       |
| MacridVAE   | -                         | -                           | -                        | -                          |
| MultiDAE    | 3.13                      | 4.83                        | 5.81                     | 5.81                       |
| MultiVAE    | 3.05                      | 5.07                        | 5.82                     | 5.82                       |
| NAIS        | -                         | -                           | -                        | -                          |
| NCE-PLRec   | 5.32                      | 11.38                       | 0.39                     | 1.18                       |
| NCL         | 82.13                     | 3.59                        | 5.16                     | 5.16                       |
| NeuMF       | 11.34                     | 141.59                      | 0.32                     | 10.98                      |
| NGCF        | 59.71                     | 3.70                        | 1.22                     | 1.84                       |
| NNCF        | -                         | -                           | -                        | -                          |
| Pop         | 0.01                      | 0.08                        | 0                        | 0.77                       |
| RaCT        | -                         | -                           | -                        | -                          |
| RecVAE      | 12.71                     | 4.79                        | 5.87                     | 5.88                       |
| SGL         | 169.00                    | 3.51                        | 5.01                     | 5.01                       |
| SimpleX     | 9.04                      | 3.80                        | 1.17                     | 1.78                       |
| SLIMElastic | 8.38                      | 75.45                       | 0                        | 0                          |
| SpectralCF  | 68.78                     | 4.00                        | 1.08                     | 1.69                       |

#### Config file of Yelp2022 dataset:

```
# dataset config
field_separator: "\t"
seq_separator: " "
USER_ID_FIELD: user_id
ITEM_ID_FIELD: item_id
RATING_FIELD: rating
NEG_PREFIX: neg_
LABEL_FIELD: label
load_col:
    inter: [user_id, item_id, rating]

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
train_neg_sample_args: 
    distribution: uniform
    sample_num: 1
    dynamic: False

# model
embedding_size: 64
```

Other parameters (including model parameters) are default value. 









