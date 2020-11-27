## Time and memory cost of context-aware recommendation models 

### Datasets information:

| Dataset | #Interaction | #Feature Field                                               | #Feature |
| ------- | ------------: | ------------------------------------------------------------ | --------: |
| ml-1m   | 1,000,209    | item: [ release_year, genre]  user: [ age, gender, occupation] | 134      |
| Criteo  | 2,292,530    | inter:[I1-I13, C1-C26]                                       | 2572192  |
| Avazu   | 4,218,938    | inter: [ C1, banner_pos, site_id,  site_domain, site_category,  app_id, app_domain, app_category,  device_id, device_ip,  device_model, device_type,  device_conn_type, C14, C15,  C16, C17, C18, C19, C20, C21] | 1326631  |

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

| Method    | Training Time (sec/epoch) | Evaluation Time (sec/epoch) | GPU Memory (GB) |
| --------- | -----------------: | -----------------: | -----------: |
| LR        | 18.34             | 2.18              | 0.82        |
| DIN       | 20.37             | 2.26              | 1.16        |
| DSSM      | 21.93             | 2.24              | 0.95        |
| FM        | 19.33             | 2.34              | 0.83        |
| DeepFM    | 20.42             | 2.27              | 0.91        |
| Wide&Deep | 26.13             | 2.95              | 0.89        |
| NFM       | 23.36             | 2.26              | 0.89        |
| AFM       | 20.08             | 2.26              | 0.92        |
| AutoInt   | 22.41             | 2.34              | 0.94        |
| DCN       | 28.33             | 2.97              | 0.93        |
| FNN(DNN)  | 19.51             | 2.21              | 0.91        |
| PNN       | 22.29             | 2.23              | 0.91        |
| FFM       | 22.98             | 2.47              | 0.87        |
| FwFM      | 23.38             | 2.50              | 0.85        |
| xDeepFM   | 24.40             | 2.30              | 1.06        |

#### Config file of ml-1m dataset:

```
# dataset config
field_separator: "\t"
seq_separator: " "
USER_ID_FIELD: user_id
ITEM_ID_FIELD: item_id
LABEL_FIELD: label
threshold:
  rating: 4.0
drop_filter_field : True
load_col:
  inter: [user_id, item_id, rating]
  item: [item_id, release_year, genre]
  user: [user_id, age, gender, occupation]

# training and evaluation
epochs: 500
train_batch_size: 2048
eval_batch_size: 2048
eval_setting: RO_RS
group_by_user: False
valid_metric: AUC
metrics: ['AUC', 'LogLoss']
```

### 2）Criteo dataset:

#### Time and memory cost on Criteo dataset:

| Method    | Training Time (sec/epoch) | Evaluation Time (sec/epoch) | GPU Memory (GB) |
| --------- | -------------------------: | ---------------------------: | ---------------: |
| LR        | 7.65                      | 0.61                        | 1.11            |
| DIN       | -                         | -                           | -               |
| DSSM      | -                         | -                           | -               |
| FM        | 9.77                      | 0.73                        | 1.45            |
| DeepFM    | 13.64                     | 0.83                        | 1.72            |
| Wide&Deep | 13.58                     | 0.80                        | 1.72            |
| NFM       | 13.36                     | 0.75                        | 1.72            |
| AFM       | 19.40                     | 1.02                        | 2.34            |
| AutoInt   | 19.40                     | 0.98                        | 2.06            |
| DCN       | 16.25                     | 0.78                        | 1.67            |
| FNN(DNN)  | 10.03                     | 0.64                        | 1.63            |
| PNN       | 12.92                     | 0.72                        | 1.85            |
| FFM       | -                         | -                           | -               |
| FwFM      | 1175.24                   | 8.90                        | 2.12            |
| xDeepFM   | 32.27                     | 1.34                        | 2.25            |

#### Config file of Criteo dataset:

```
# dataset config
field_separator: "\t"
seq_separator: " "
USER_ID_FIELD: ~
ITEM_ID_FIELD: ~
LABEL_FIELD: label

load_col: 
    inter: '*'

highest_val:
    index: 2292530

fill_nan: True
normalize_all: True
min_item_inter_num: 0
min_user_inter_num: 0

drop_filter_field : True


# training and evaluation
epochs: 500
train_batch_size: 2048
eval_batch_size: 2048
eval_setting: RO_RS
group_by_user: False
valid_metric: AUC
metrics: ['AUC', 'LogLoss']
```

### 3）Avazu dataset:

#### Time and memory cost on Avazu dataset:

| Method    | Training Time (sec/epoch) | Evaluation Time (sec/epoch) | GPU Memory (GB) |
| --------- | -------------------------: | ---------------------------: | ---------------: |
| LR        | 9.30                      | 0.76                        | 1.42            |
| DIN       | -                         | -                           | -               |
| DSSM      | -                         | -                           | -               |
| FM        | 25.68                     | 0.94                        | 2.60            |
| DeepFM    | 28.41                     | 1.19                        | 2.66            |
| Wide&Deep | 27.58                     | 0.97                        | 2.66            |
| NFM       | 30.46                     | 1.06                        | 2.66            |
| AFM       | 31.03                     | 1.06                        | 2.69            |
| AutoInt   | 38.11                     | 1.41                        | 2.84            |
| DCN       | 30.78                     | 0.96                        | 2.64            |
| FNN(DNN)  | 23.53                     | 0.84                        | 2.60            |
| PNN       | 25.86                     | 0.90                        | 2.68            |
| FFM       | -                         | -                           | -               |
| FwFM      | 336.75                    | 7.49                        | 2.63            |
| xDeepFM   | 54.88                     | 1.45                        | 2.89            |

#### Config file of Avazu dataset:

```
# dataset config
field_separator: "\t"
seq_separator: " "
USER_ID_FIELD: ~
ITEM_ID_FIELD: ~
LABEL_FIELD: label
fill_nan: True
normalize_all: True

load_col:
    inter: '*'
    
lowest_val:
  timestamp: 14102931
drop_filter_field : False

# training and evaluation
epochs: 500
train_batch_size: 2048
eval_batch_size: 2048
eval_setting: RO_RS
group_by_user: False
valid_metric: AUC
metrics: ['AUC', 'LogLoss']
```









