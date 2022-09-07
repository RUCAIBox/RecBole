# Context-aware Recommendation

- **Dataset**: [ml-1m](../../md/ml-1m_contxt.md)

- **Model**: [DIN](https://recbole.io/docs/user_guide/model/context/din.html)

- **Hyper-parameter searching** (hyper.test):

  ```yaml
    learning_rate choice [1e-4,5e-4,1e-3,3e-3,5e-3,6e-3,1e-2]
    dropout_prob choice[0.0,0.1,0.2,0.3]
    mlp_hidden_size choice ['[64,64,64]','[128,128,128]','[256,256,256]','[512,512,512]']
    pooling_mode choice ['mean','max','sum']
  ```

- **Best parameters**:

  ```
    learning_rate: 0.01
    mlp_hidden_size: [128,128,128]
    pooling_mode: 'mean'
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
    learning_rate:0.001, mlp_hidden_size:[64,64,64], pooling_mode:sum
    Valid result:
    auc : 0.9342    logloss : 0.3072
    Test result:
    auc : 0.9257    logloss : 0.3095

    learning_rate:0.0001, mlp_hidden_size:[64,64,64], pooling_mode:max
    Valid result:
    auc : 0.9292    logloss : 0.3023
    Test result:
    auc : 0.9188    logloss : 0.3064

    learning_rate:0.0005, mlp_hidden_size:[128,128,128], pooling_mode:mean
    Valid result:
    auc : 0.9291    logloss : 0.3076
    Test result:
    auc : 0.9203    logloss : 0.3102

    learning_rate:0.005, mlp_hidden_size:[128,128,128], pooling_mode:max
    Valid result:
    auc : 0.9372    logloss : 0.2891
    Test result:
    auc : 0.9273    logloss : 0.2923

    learning_rate:0.005, mlp_hidden_size:[512,512,512], pooling_mode:sum
    Valid result:
    auc : 0.9372    logloss : 0.2926
    Test result:
    auc : 0.9286    logloss : 0.2953

    learning_rate:0.003, mlp_hidden_size:[64,64,64], pooling_mode:sum
    Valid result:
    auc : 0.9355    logloss : 0.2668
    Test result:
    auc : 0.9272    logloss : 0.2686

    learning_rate:0.003, mlp_hidden_size:[64,64,64], pooling_mode:mean
    Valid result:
    auc : 0.9358    logloss : 0.2818
    Test result:
    auc : 0.9265    logloss : 0.2854

    learning_rate:0.006, mlp_hidden_size:[64,64,64], pooling_mode:max
    Valid result:
    auc : 0.9369    logloss : 0.2678
    Test result:
    auc : 0.9282    logloss : 0.2701

    learning_rate:0.01, mlp_hidden_size:[128,128,128], pooling_mode:mean
    Valid result:
    auc : 0.9377    logloss : 0.2851
    Test result:
    auc : 0.9299    logloss : 0.2879

    learning_rate:0.003, mlp_hidden_size:[256,256,256], pooling_mode:max
    Valid result:
    auc : 0.9374    logloss : 0.2813
    Test result:
    auc : 0.929    logloss : 0.2852

    learning_rate:0.0005, mlp_hidden_size:[512,512,512], pooling_mode:max
    Valid result:
    auc : 0.9325    logloss : 0.2828
    Test result:
    auc : 0.9231    logloss : 0.2864

    learning_rate:0.0001, mlp_hidden_size:[64,64,64], pooling_mode:sum
    Valid result:
    auc : 0.9295    logloss : 0.2952
    Test result:
    auc : 0.9206    logloss : 0.2966

    learning_rate:0.0001, mlp_hidden_size:[512,512,512], pooling_mode:sum
    Valid result:
    auc : 0.9288    logloss : 0.2928
    Test result:
    auc : 0.9196    logloss : 0.2958

    learning_rate:0.003, mlp_hidden_size:[128,128,128], pooling_mode:mean
    Valid result:
    auc : 0.9358    logloss : 0.283
    Test result:
    auc : 0.9269    logloss : 0.2859

    learning_rate:0.0001, mlp_hidden_size:[512,512,512], pooling_mode:mean
    Valid result:
    auc : 0.9274    logloss : 0.2924
    Test result:
    auc : 0.9181    logloss : 0.2966

    learning_rate:0.003, mlp_hidden_size:[128,128,128], pooling_mode:sum
    Valid result:
    auc : 0.9355    logloss : 0.2855
    Test result:
    auc : 0.927    logloss : 0.2877

    learning_rate:0.0001, mlp_hidden_size:[256,256,256], pooling_mode:sum
    Valid result:
    auc : 0.9288    logloss : 0.2901
    Test result:
    auc : 0.9196    logloss : 0.2934

    learning_rate:0.0005, mlp_hidden_size:[256,256,256], pooling_mode:mean
    Valid result:
    auc : 0.9306    logloss : 0.2999
    Test result:
    auc : 0.9208    logloss : 0.3025

    learning_rate:0.005, mlp_hidden_size:[64,64,64], pooling_mode:sum
    Valid result:
    auc : 0.9366    logloss : 0.2683
    Test result:
    auc : 0.9267    logloss : 0.2718
  ```

- **Logging Result**:

  ```yaml
    23%|██▎       | 19/84 [12:14:36<41:53:07, 2319.81s/trial, best loss: -0.9377]
    best params:  {'learning_rate': 0.01, 'mlp_hidden_size': '[128,128,128]', 'pooling_mode': 'mean'}
    best result: 
    {'model': 'DIN', 'best_valid_score': 0.9377, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('auc', 0.9377), ('logloss', 0.2851)]), 'test_result': OrderedDict([('auc', 0.9299), ('logloss', 0.2879)])}
  ```