## Time  and memory cost of sequential recommendation models 

### Datasets information:

| Dataset    | #User   | #Item  | #Interaction | Sparsity |
| ---------- | ------- | ------ | ------------ | -------- |
| ml-1m      | 6,041   | 3,707  | 1,000,209    | 0.9553   |
| DIGINETICA | 59,425  | 42,116 | 547,416      | 0.9998   |
| Yelp       | 102,046 | 98,408 | 2,903,648    | 0.9997   |

### 1) ml-1m dataset:

#### Time and memory cost on ml-1m dataset:

| Method           | Training Time (s) | Evaluate Time (s) | Memory (MB) |
| ---------------- | ----------------- | ----------------- | ----------- |
| Improved GRU-Rec | 7.78              | 0.11              | 1305        |
| SASRec           | 17.78             | 0.12              | 1889        |
| NARM             | 8.29              | 0.11              | 1323        |
| FPMC             | 7.51              | 0.11              | 1209        |
| STAMP            | 7.32              | 0.11              | 1229        |
| Caser            | 44.85             | 0.12              | 1165        |
| TransRec         | 10.08             | 0.16              | 8373        |
| GRU4RecF         | 10.2              | 0.15              | 1847        |
| SASRecF          | 18.84             | 0.17              | 1819        |
| BERT4Rec         | 36.09             | 0.34              | 2017        |
| FDSA             | 31.86             | 0.19              | 2375        |
| SRGNN            | 327.38            | 2.19              | 1243        |
| GCSAN            | 335.27            | 0.022             | 1623        |

#### Config file of ml-1m dataset:

```
# dataset config
field_separator: "\t"
seq_separator: " "
USER_ID_FIELD: user_id
ITEM_ID_FIELD: item_id
TIME_FIELD: timestamp
NEG_PREFIX: neg_
ITEM_LIST_LENGTH_FIELD: item_length
LIST_SUFFIX: _list
MAX_ITEM_LIST_LENGTH: 20
POSITION_FIELD: position_id
load_col:
  inter: [user_id, item_id, timestamp]
min_user_inter_num: 0
min_item_inter_num: 0

# training and evaluation
epochs: 500
train_batch_size: 2048
eval_batch_size: 2048
valid_metric: MRR@10
eval_setting: TO_LS,full
training_neg_sample_num: 0
```

**NOTE :** 

1) For FPMC and TransRec model,  `training_neg_sample_num`  should be  `1` . 

2) For  SASRecF, GRU4RecF and FDSA,   `load_col` should as below:

```
load_col:
  inter: [user_id, item_id, timestamp]
  item: [item_id, genre]
```

### 2）DIGINETICA dataset:

#### Time and memory cost on DIGINETICA dataset:

| Method           | Training Time (s) | Evaluate Time (s) | Memory (MB) |
| ---------------- | ----------------- | ----------------- | ----------- |
| Improved GRU-Rec | 4.1               | 1.05              | 4121        |
| SASRec           | 8.36              | 1.21              | 4537        |
| NARM             | 4.3               | 1.08              | 4185        |
| FPMC             | 2.98              | 1.08              | 4181        |
| STAMP            | 4.27              | 1.04              | 3973        |
| Caser            | 17.15             | 1.18              | 4033        |
| GRU4RecF         | 4.79              | 1.17              | 4949        |
| SASRecF          | 8.66              | 1.29              | 5237        |
| BERT4Rec         | 16.8              | 3.54              | 8157        |
| FDSA             | 13.44             | 1.47              | 5799        |
| SRGNN            | 88.59             | 15.37             | 4105        |
| GCSAN            | 96.69             | 17.11             | 4355        |

#### Config file of DIGINETICA dataset:

```
# dataset config
field_separator: "\t"
seq_separator: " "
USER_ID_FIELD: user_id
ITEM_ID_FIELD: item_id
TIME_FIELD: timestamp
NEG_PREFIX: neg_
ITEM_LIST_LENGTH_FIELD: item_length
LIST_SUFFIX: _list
MAX_ITEM_LIST_LENGTH: 20
POSITION_FIELD: position_id
load_col:
  inter: [user_id, item_id, timestamp]
min_user_inter_num: 0
min_item_inter_num: 0

# training and evaluation
epochs: 500
train_batch_size: 2048
eval_batch_size: 2048
valid_metric: MRR@10
eval_setting: TO_LS,full
training_neg_sample_num: 1
```

### 3) Yelp dataset:

#### Time and memory cost on Yelp dataset:



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











