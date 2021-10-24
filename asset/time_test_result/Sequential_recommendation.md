## Time and memory cost of sequential recommendation models 

### Datasets information:

| Dataset    | #User   | #Item   | #Interaction | Sparsity |
| ---------- | ------: | ------: | -----------: | -------: |
| ml-1m      |  6,040  |  3,629  |     836,478  |  0.9618  |
| DIGINETICA | 72,014  | 29,454  |     580,490  |  0.9997  |
| Yelp       | 45,478  | 30,709  |   1,777,765  |  0.9987  |

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

| Method           | Training Time (sec/epoch) | Evaluate Time (sec/epoch)   | GPU Memory (GB)          |
| ---------------- | ------------------------: | --------------------------: | -----------------------: |
| Improved GRU-Rec |                     6.15  |                       0.06  |                    1.46  |
| SASRec           |                    24.86  |                       0.09  |                    4.11  |
| NARM             |                     7.34  |                       0.07  |                    1.51  |
| FPMC             |                     6.07  |                       0.05  |                    0.30  |
| STAMP            |                     3.10  |                       0.10  |                    0.65  |
| Caser            |                   549.40  |                       0.21  |                    1.08  |
| NextItNet        |                   203.85  |                       0.71  |                    3.68  |
| TransRec         |                     5.81  |                       0.12  |                    7.21  |
| S3Rec            |                         - |                           - |       CUDA out of memory |
| GRU4RecF         |                    13.54  |                       0.09  |                    2.18  |
| SASRecF          |                    28.16  |                       0.11  |                    5.08  |
| BERT4Rec         |                    56.57  |                       0.32  |                    6.20  |
| FDSA             |                    50.18  |                       0.14  |                    8.12  |
| SRGNN            |                   631.08  |                       4.58  |                    1.12  |
| GCSAN            |                   671.66  |                       4.99  |                    2.96  |
| KSR              |                    61.18  |                       0.21  |                    6.94  |
| GRU4RecKG        |                    11.36  |                       0.07  |                    2.22  |
| LightSANs        |                    28.21  |                       0.12  |                    3.85  |

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
val_interval:
  rating: "[3,inf)"

# training and evaluation
epochs: 500
train_batch_size: 4096
eval_batch_size: 2048
valid_metric: recall@10
eval_args:
  split: {'LS': 'valid_and_test'}
  mode: full
  order: TO
neg_sampling: ~

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

### 2）DIGINETICA dataset:

#### Time and memory cost on DIGINETICA dataset:

| Method           | Training Time (sec/epoch) | Evaluate Time (sec/epoch)   | GPU Memory (GB)          |
| ---------------- | ------------------------: | --------------------------: | -----------------------: |
| Improved GRU-Rec |                     2.22  |                       0.74  |                    2.70  |
| SASRec           |                     6.88  |                       0.88  |                    3.53  |
| NARM             |                     2.42  |                       0.74  |                    2.37  |
| FPMC             |                     2.44  |                       0.72  |                    2.31  |
| STAMP            |                     2.01  |                       0.71  |                    2.44  |
| Caser            |                    17.51  |                       0.86  |                    2.49  |
| NextItNet        |                    34.00  |                       2.98  |                    3.71  |
| TransRec         |                         - |                           - |       CUDA out of memory |
| S3Rec            |                   160.15  |                           - |                    6.20  |
| GRU4RecF         |                     3.00  |                       0.84  |                    3.02  |
| SASRecF          |                     6.67  |                       0.93  |                    3.61  |
| BERT4Rec         |                    12.97  |                       3.43  |                   10.26  |
| FDSA             |                    11.17  |                       1.10  |                    4.71  |
| SRGNN            |                    66.67  |                      14.81  |                    2.66  |
| GCSAN            |                    69.03  |                      14.29  |                    3.13  |
| KSR              |                         - |                           - |                        - |
| GRU4RecKG        |                         - |                           - |                        - |
| LightSANs        |                     7.12  |                       0.88  |                    3.48  |

#### Config file of DIGINETICA dataset:

```
# dataset config
field_separator: "\t"
seq_separator: " "
USER_ID_FIELD: session_id
ITEM_ID_FIELD: item_id
TIME_FIELD: timestamp
NEG_PREFIX: neg_
ITEM_LIST_LENGTH_FIELD: item_length
LIST_SUFFIX: _list
MAX_ITEM_LIST_LENGTH: 20
POSITION_FIELD: position_id
load_col:
  inter: [session_id, item_id, timestamp]
user_inter_num_interval: "[5,inf)"
item_inter_num_interval: "[5,inf)"

# training and evaluation
epochs: 500
train_batch_size: 4096
eval_batch_size: 2048
valid_metric: MRR@10
eval_args:
  split: {'LS':"valid_and_test"}
  mode: full
  order: TO
neg_sampling: ~

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

### 3）Yelp dataset:

#### Time and memory cost on Yelp dataset:

| Method           | Training Time (sec/epoch) | Evaluation Time (sec/epoch) | GPU Memory (GB)          |
| ---------------- | ------------------------: | --------------------------: | -----------------------: |
| Improved GRU-Rec |                    16.52  |                       0.66  |                    3.34  |
| SASRec           |                    54.92  |                       0.90  |                    5.82  |
| NARM             |                    19.35  |                       0.79  |                    3.26  |
| FPMC             |                    11.44  |                       0.58  |                    0.76  |
| STAMP            |                     9.97  |                       0.59  |                    2.76  |
| Caser            |                  1105.96  |                       2.02  |                    3.21  |
| NextItNet        |                   413.43  |                       5.57  |                    5.73  |
| TransRec         |                         - |                           - |       CUDA out of memory |
| S3Rec            |                         - |                           - |       CUDA out of memory |
| GRU4RecF         |                    62,41  |                       1.32  |                    7.36  |
| SASRecF          |                    86.20  |                       1.42  |                    6.12  |
| BERT4Rec         |                         - |                           - |       CUDA out of memory |
| FDSA             |                   133.52  |                       1.87  |                    9.19  |
| SRGNN            |                  1165.06  |                      30.35  |                    3.41  |
| GCSAN            |                  1112.46  |                      29.84  |                    4.67  |
| KSR              |                         - |                           - |                        - |
| GRU4RecKG        |                         - |                           - |                        - |
| LightSANs        |                    58.46  |                       0.83  |                    5.08  |

#### Config file of Yelp dataset:

```
# dataset config
field_separator: "\t"
seq_separator: " "
USER_ID_FIELD: user_id
ITEM_ID_FIELD: business_id
RATING_FIELD: stars
TIME_FIELD: date
NEG_PREFIX: neg_
ITEM_LIST_LENGTH_FIELD: item_length
LIST_SUFFIX: _list
MAX_ITEM_LIST_LENGTH: 50
POSITION_FIELD: position_id
load_col:
  inter: [user_id, business_id, stars, date]

user_inter_num_interval: "[15,inf)"
item_inter_num_interval: "[15,inf)"
val_interval:
  stars: "[3,inf)"

# training and evaluation
epochs: 500
train_batch_size: 4096
eval_batch_size: 512
valid_metric: MRR@10
eval_args:
  split: {'LS':"valid_and_test"}
  order: TO
neg_sampling: ~

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