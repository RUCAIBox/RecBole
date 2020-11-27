## Time  and memory cost of general recommendation models 

### Datasets information:

| Dataset | #User   | #Item  | #Interaction | Sparsity |
| ------- | -------: | ------: | ------------: | --------: |
| ml-1m   | 6,041   | 3,707  | 1,000,209    | 0.9553   |
| Netflix | 80,476  | 16,821 | 1,977,844    | 0.9985   |
| Yelp    | 102,046 | 98,408 | 2,903,648    | 0.9997   |

### Device information

```
OS:                   Linux
Python Version:       3.8.3
PyTorch Version:      1.7.0
cudatoolkit Version:  10.1
GPU:                  TITAN RTX（24GB）
Machine Specs:        32 CPU machine, 64GB RAM
```

### 1) ml-1m dataset:

#### Time and memory cost on ml-1m dataset:

| Method     | Training Time (sec/epoch) | Evaluation Time (sec/epoch) | GPU Memory (GB) |
| ---------- | ------------------------: | --------------------------: | --------------: |
| Popularity |                      2.11 |                        8.08 |            0.82 |
| ItemKNN    |                      2.00 |                       11.76 |            0.82 |
| BPRMF      |                      1.93 |                        7.43 |            0.91 |
| NeuMF      |                      4.94 |                       13.12 |            0.94 |
| DMF        |                      4.47 |                       12.63 |            1.52 |
| NAIS       |                     59.27 |                       24.41 |           21.83 |
| NGCF       |                     12.09 |                        7.12 |            1.20 |
| GCMC       |                      9.04 |                       54.15 |            1.32 |
| LightGCN   |                      7.83 |                        7.47 |            1.15 |
| DGCF       |                    181.66 |                        8.06 |            6.59 |
| ConvNCF    |                      8.46 |                       19.60 |            1.31 |
| FISM       |                     19.30 |                       10.92 |            6.94 |
| SpectralCF |                     13.87 |                        6.97 |            1.19 |

#### Config file of ml-1m dataset:

```
# dataset config
field_separator: "\t"
seq_separator: " "
USER_ID_FIELD: user_id
ITEM_ID_FIELD: item_id
RATING_FIELD: rating
TIME_FIELD: timestamp
LABEL_FIELD: label
NEG_PREFIX: neg_
load_col:
  inter: [user_id, item_id, rating, timestamp]
min_user_inter_num: 0
min_item_inter_num: 0


# training and evaluation
epochs: 500
train_batch_size: 2048
eval_batch_size: 2048
valid_metric: MRR@10
```

### 2）Netflix dataset:

#### Time and memory cost on Netflix dataset:

| Method     | Training Time (sec/epoch) | Evaluation Time (sec/epoch) | GPU Memory (GB) |
| ---------- | ----------------: | -----------------: | -----------: |
| Popularity | 3.98              | 58.86             | 0.86     |
| ItemKNN    | 5.42              | 69.64             | 0.86      |
| BPRMF      | 4.42              | 52.81             | 1.08    |
| NeuMF      | 11.33             | 238.92            | 1.26     |
| DMF        | 20.62             | 68.89             | 7.12     |
| NAIS       | -                 | -                 | -           |
| NGCF       | 52.50             | 51.60             | 2.00     |
| GCMC       | 93.15             |                     1810.43 | 3.17     |
| LightGCN   | 30.21             | 47.12             | 1.58     |
| DGCF       |                    750.74 |                       47.23 |           12.52 |
| ConvNCF    | 17.02             | 402.65            | 1.44     |
| FISM       | 86.52             | 83.26             | 20.54   |
| SpectralCF | 59.92             | 46.94             | 1.88     |

#### Config file of Netflix dataset:

```
# dataset config
field_separator: "\t"
seq_separator: " "
USER_ID_FIELD: user_id
ITEM_ID_FIELD: item_id
RATING_FIELD: rating
TIME_FIELD: timestamp
LABEL_FIELD: label
NEG_PREFIX: neg_
load_col:
  inter: [user_id, item_id, rating, timestamp]
min_user_inter_num: 3
min_item_inter_num: 0
lowest_val:
  timestamp: 1133366400
  rating: 3
drop_filter_field : True

# training and evaluation
epochs: 500
train_batch_size: 2048
eval_batch_size: 2048
valid_metric: MRR@10
```

### 3) Yelp dataset:

#### Time and memory cost on Yelp dataset:

| Method     | Training Time (sec/epoch) | Evaluate Time (sec/epoch) | GPU Memory (GB) |
| ---------- | -------------------------: | -------------------------: | ---------------: |
| Popularity | 5.69                      | 134.23                    | 0.89            |
| ItemKNN    | 8.44                      | 194.24                    | 0.90            |
| BPRMF      | 6.31                      | 120.03                    | 1.29            |
| NeuMF      | 17.38                     | 2069.53                   | 1.67            |
| DMF        | 43.96                     | 173.13                    | 9.22            |
| NAIS       | -                         | -                         | -               |
| NGCF       | 122.90                    | 129.59                    | 3.28            |
| GCMC       | 299.36                    | 9833.24                   | 5.96            |
| LightGCN   | 67.91                     | 116.16                    | 2.02            |
| DGCF       | 1542.00                   | 119.00                    | 17.17           |
| ConvNCF    | 87.56                     | 11155.31                  | 1.62            |
| FISM       | -                         | -                         | -               |
| SpectralCF | 138.99                    | 133.37                    | 3.10            |

#### Config file of Yelp dataset:

```
# dataset config
field_separator: "\t"
seq_separator: " "
USER_ID_FIELD: user_id
ITEM_ID_FIELD: business_id
RATING_FIELD: stars
TIME_FIELD: date
LABEL_FIELD: label
NEG_PREFIX: neg_
load_col:
  inter: [user_id, business_id, stars]
min_user_inter_num: 10
min_item_inter_num: 4
lowest_val:
  stars: 3
drop_filter_field: True

# training and evaluation
epochs: 500
train_batch_size: 2048
eval_batch_size: 2048
valid_metric: MRR@10
```






