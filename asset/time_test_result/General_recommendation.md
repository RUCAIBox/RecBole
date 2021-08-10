## Time  and memory cost of general recommendation models 

### Datasets information:

| Dataset | #User   | #Item   | #Interaction | Sparsity |
| ------- | ------: | ------: | -----------: | -------: |
| ml-1m   |  6,040  |  3,629  |   836,478    |  0.9618  |
| Netflix | 40,227  |  8,727  |  1,752,648   |  0.9950  |
| Yelp    | 45,478  | 30,709  |  1,777,765   |  0.9987  |

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

| Method     | Training Time (sec/epoch) | Evaluation Time (sec/epoch) | GPU Memory (GB) |
| ---------- | ------------------------: | --------------------------: | --------------: |
| Popularity |                     0.62  |                       0.41  |           0.00  |
| ItemKNN    |                     0.65  |                       4.87  |           0.00  |
| BPRMF      |                     0.89  |                       0.71  |           0.03  |
| NeuMF      |                     3.63  |                       0.83  |           0.33  |
| DMF        |                     3.70  |                       1.34  |           0.87  |
| NAIS       |                    44.94  |                      13.73  |           8.12  |
| NGCF       |                     6.19  |                       0.40  |           0.19  |
| GCMC       |                     4.46  |                       1.74  |           0.26  |
| LightGCN   |                     3.76  |                       0.76  |           0.16  |
| DGCF       |                    63.83  |                       0.57  |           4.15  |
| ConvNCF    |                     8.43  |                      10.04  |           8.58  |
| FISM       |                    17.54  |                       3.46  |           3.35  |
| SpectralCF |                     8.02  |                       0.43  |           0.18  |

#### Config file of ml-1m dataset:

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
val_interval:
    rating: "[3,inf)"    
unused_col: 
    inter: [rating]

# training and evaluation
epochs: 500
train_batch_size: 4096
eval_batch_size: 102400
valid_metric: MRR@10

# model
embedding_size: 64
```

Other parameters (including model parameters) are default value. 

### 2）Netflix dataset:

#### Time and memory cost on Netflix dataset:

| Method     | Training Time (sec/epoch) | Evaluation Time (sec/epoch) | GPU Memory (GB)      |
| ---------- | ------------------------: | --------------------------: | -------------------: |
| Popularity |                     1.55  |                       6.62  |                0.00  |
| ItemKNN    |                     2.48  |                      25.71  |                0.00  |
| BPRMF      |                     1.92  |                       5.82  |                0.09  |
| NeuMF      |                     7.54  |                      12.61  |                0.40  |
| DMF        |                    10.66  |                       8.14  |                3.54  |
| NAIS       |                         - |                           - |   CUDA out of memory |
| NGCF       |                    18.26  |                       5.70  |                0.58  |
| GCMC       |                    22.07  |                      86.32  |                1.17  |
| LightGCN   |                    10.85  |                       6.31  |                0.41  |
| DGCF       |                   269.08  |                       5.39  |                8.80  |
| ConvNCF    |                    15.66  |                     168.54  |                8.29  |
| FISM       |                    57.58  |                      23.10  |                8.51  |
| SpectralCF |                    20.67  |                       5.52  |                0.51  |

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

user_inter_num_interval: "[10,inf)"
item_inter_num_interval: "[10,inf)"
val_interval:
  rating: "[3,inf)"
  timestamp: "[1133366400, inf)"


# training and evaluation
epochs: 500
train_batch_size: 4096
eval_batch_size: 102400
valid_metric: MRR@10

# model
embedding_size: 64 
```

Other parameters (including model parameters) are default value. 

### 3) Yelp dataset:

#### Time and memory cost on Yelp dataset:

| Method     | Training Time (sec/epoch) | Evaluate Time (sec/epoch) | GPU Memory (GB)      |
| ---------- | ------------------------: | ------------------------: | -------------------: |
| Popularity |                     1.71  |                     6.45  |                0.02  |
| ItemKNN    |                     5.67  |                    37.37  |                0.02  |
| BPRMF      |                     2.86  |                     5.96  |                0.13  |
| NeuMF      |                     7.75  |                    32.75  |                1.27  |
| DMF        |                    12.82  |                     9.27  |                2.90  |
| NAIS       |                         - |                         - |   CUDA out of memory |
| NGCF       |                    23.17  |                     5.62  |                0.79  |
| GCMC       |                    32.20  |                   110.34  |                1.65  |
| LightGCN   |                    13.06  |                     5.85  |                0.47  |
| DGCF       |                   270.31  |                     5.92  |                8.62  |
| ConvNCF    |                         - |                         - |   CUDA out of memory |
| FISM       |                         - |                         - |   CUDA out of memory |
| SpectralCF |                    24.44  |                     5.73  |                0.62  |

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

user_inter_num_interval: "[15,inf)"
item_inter_num_interval: "[15,inf)"
val_interval:
  stars: "[3,inf)"


# training and evaluation
epochs: 500
train_batch_size: 4096
eval_batch_size: 102400
valid_metric: MRR@10

# model
embedding_size: 64
```

Other parameters (including model parameters) are default value. 









