## Training and testing time of general recommendation models 

### Datasets information:

| Dataset | #User   | #Item  | #Interaction | Sparsity |
| ------- | ------- | ------ | ------------ | -------- |
| ml-1m   | 6,041   | 3,707  | 1,000,209    | 0.9553   |
| Netflix | 80,476  | 16,821 | 1,977,844    | 0.9985   |
| Yelp    | 102,046 | 98,408 | 2,903,648    | 0.9997   |

### 1) ml-1m dataset:

#### Time and memory cost on ml-1m dataset:

| Method     | Training Time (s) | Evaluate Time (s) | Memory (MB) |
| ---------- | ----------------- | ----------------- | ----------- |
| Popularity | 2.11              | 8.08              | 843         |
| ItemKNN    | 2                 | 11.76             | 843         |
| BPRMF      | 1.93              | 7.43              | 931         |
| NeuMF      | 4.94              | 13.12             | 965         |
| DMF        | 4.47              | 12.63             | 1555        |
| NAIS       | 59.27             | 24.41             | 22351       |
| NGCF       | 12.09             | 7.12              | 1231        |
| GCMC       | 9.04              | 54.15             | 1353        |
| LightGCN   | 7.83              | 7.47              | 1177        |
| DGCF       | 181.66            | 8.06              | 6745        |
| ConvNCF    | 8.46              | 19.6              | 1341        |
| FISM       | 19.3              | 10.92             | 7109        |
| SpectralCF | 13.87             | 6.97              | 1219        |

#### Config file of ml-1m:

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



### Time and memory cost on Netflix dataset:



### Time and memory cost on Yelp dataset:









