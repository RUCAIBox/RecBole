# Context-aware Recommendation

- **Dataset**: [Avazu](../../md/avazu.md)

- **Model**: [NFM](https://recbole.io/docs/user_guide/model/context/nfm.html)

- **Time cost**: 172.10s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
    learning_rate choice [5e-5,8e-5,1e-4,5e-4,1e-3]
    dropout_prob choice [0.1,0.2,0.3]
    mlp_hidden_size choice ['[20,20,20]','[40,40,40]','[50,50,50]']
  ```

- **Best parameters**:

  ```
    dropout_prob: 0.1
    learning_rate: 0.001
    mlp_hidden_size: [50,50,50]
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
    dropout_prob:0.2, learning_rate:0.0005, mlp_hidden_size:[40,40,40]
    Valid result:
    auc : 0.7864    logloss : 0.3696
    Test result:
    auc : 0.7838    logloss : 0.37

    dropout_prob:0.3, learning_rate:0.001, mlp_hidden_size:[40,40,40]
    Valid result:
    auc : 0.7856    logloss : 0.3698
    Test result:
    auc : 0.7823    logloss : 0.3707

    dropout_prob:0.2, learning_rate:0.0005, mlp_hidden_size:[20,20,20]
    Valid result:
    auc : 0.787    logloss : 0.3711
    Test result:
    auc : 0.7838    logloss : 0.3717

    dropout_prob:0.3, learning_rate:5e-05, mlp_hidden_size:[40,40,40]
    Valid result:
    auc : 0.7797    logloss : 0.3841
    Test result:
    auc : 0.7767    logloss : 0.3844

    dropout_prob:0.1, learning_rate:0.0005, mlp_hidden_size:[50,50,50]
    Valid result:
    auc : 0.7911    logloss : 0.3667
    Test result:
    auc : 0.7883    logloss : 0.3674

    dropout_prob:0.2, learning_rate:8e-05, mlp_hidden_size:[40,40,40]
    Valid result:
    auc : 0.7729    logloss : 0.3785
    Test result:
    auc : 0.771    logloss : 0.3788

    dropout_prob:0.1, learning_rate:0.0001, mlp_hidden_size:[20,20,20]
    Valid result:
    auc : 0.7767    logloss : 0.4542
    Test result:
    auc : 0.7733    logloss : 0.4551

    dropout_prob:0.2, learning_rate:5e-05, mlp_hidden_size:[40,40,40]
    Valid result:
    auc : 0.7825    logloss : 0.3772
    Test result:
    auc : 0.78    logloss : 0.3775

    dropout_prob:0.3, learning_rate:0.0005, mlp_hidden_size:[40,40,40]
    Valid result:
    auc : 0.7834    logloss : 0.371
    Test result:
    auc : 0.78    logloss : 0.3719

    dropout_prob:0.2, learning_rate:0.001, mlp_hidden_size:[50,50,50]
    Valid result:
    auc : 0.7908    logloss : 0.3679
    Test result:
    auc : 0.7881    logloss : 0.3686

    dropout_prob:0.2, learning_rate:8e-05, mlp_hidden_size:[50,50,50]
    Valid result:
    auc : 0.7704    logloss : 0.4207
    Test result:
    auc : 0.767    logloss : 0.4217

    dropout_prob:0.1, learning_rate:0.0005, mlp_hidden_size:[20,20,20]
    Valid result:
    auc : 0.7901    logloss : 0.369
    Test result:
    auc : 0.7874    logloss : 0.3694

    dropout_prob:0.1, learning_rate:5e-05, mlp_hidden_size:[20,20,20]
    Valid result:
    auc : 0.7814    logloss : 0.4511
    Test result:
    auc : 0.7779    logloss : 0.4522

    dropout_prob:0.1, learning_rate:0.001, mlp_hidden_size:[40,40,40]
    Valid result:
    auc : 0.7913    logloss : 0.3663
    Test result:
    auc : 0.7886    logloss : 0.3668

    dropout_prob:0.3, learning_rate:0.0005, mlp_hidden_size:[50,50,50]
    Valid result:
    auc : 0.7862    logloss : 0.3702
    Test result:
    auc : 0.7832    logloss : 0.3711

    dropout_prob:0.3, learning_rate:0.0001, mlp_hidden_size:[20,20,20]
    Valid result:
    auc : 0.7805    logloss : 0.3917
    Test result:
    auc : 0.7772    logloss : 0.3922

    dropout_prob:0.3, learning_rate:8e-05, mlp_hidden_size:[40,40,40]
    Valid result:
    auc : 0.7811    logloss : 0.3745
    Test result:
    auc : 0.7784    logloss : 0.375

    dropout_prob:0.2, learning_rate:5e-05, mlp_hidden_size:[50,50,50]
    Valid result:
    auc : 0.7742    logloss : 0.4049
    Test result:
    auc : 0.7709    logloss : 0.4062

    dropout_prob:0.2, learning_rate:0.0001, mlp_hidden_size:[20,20,20]
    Valid result:
    auc : 0.775    logloss : 0.3836
    Test result:
    auc : 0.7739    logloss : 0.3837

    dropout_prob:0.3, learning_rate:8e-05, mlp_hidden_size:[20,20,20]
    Valid result:
    auc : 0.7823    logloss : 0.4076
    Test result:
    auc : 0.7791    logloss : 0.408

    dropout_prob:0.1, learning_rate:0.001, mlp_hidden_size:[50,50,50]
    Valid result:
    auc : 0.7924    logloss : 0.3658
    Test result:
    auc : 0.7898    logloss : 0.3663

    dropout_prob:0.1, learning_rate:5e-05, mlp_hidden_size:[50,50,50]
    Valid result:
    auc : 0.7646    logloss : 0.4631
    Test result:
    auc : 0.7616    logloss : 0.464

    dropout_prob:0.3, learning_rate:5e-05, mlp_hidden_size:[50,50,50]
    Valid result:
    auc : 0.7778    logloss : 0.3993
    Test result:
    auc : 0.7745    logloss : 0.4004

    dropout_prob:0.3, learning_rate:0.001, mlp_hidden_size:[50,50,50]
    Valid result:
    auc : 0.7892    logloss : 0.3678
    Test result:
    auc : 0.7863    logloss : 0.3685

    dropout_prob:0.1, learning_rate:0.001, mlp_hidden_size:[20,20,20]
    Valid result:
    auc : 0.7921    logloss : 0.3664
    Test result:
    auc : 0.7893    logloss : 0.3668

    dropout_prob:0.1, learning_rate:0.0005, mlp_hidden_size:[40,40,40]
    Valid result:
    auc : 0.7894    logloss : 0.3677
    Test result:
    auc : 0.7865    logloss : 0.3683

    dropout_prob:0.3, learning_rate:8e-05, mlp_hidden_size:[50,50,50]
    Valid result:
    auc : 0.7756    logloss : 0.3832
    Test result:
    auc : 0.7719    logloss : 0.3847

    dropout_prob:0.2, learning_rate:0.0005, mlp_hidden_size:[50,50,50]
    Valid result:
    auc : 0.7882    logloss : 0.3692
    Test result:
    auc : 0.7853    logloss : 0.37

    dropout_prob:0.1, learning_rate:5e-05, mlp_hidden_size:[40,40,40]
    Valid result:
    auc : 0.773    logloss : 0.382
    Test result:
    auc : 0.7708    logloss : 0.3823

    dropout_prob:0.2, learning_rate:5e-05, mlp_hidden_size:[20,20,20]
    Valid result:
    auc : 0.7824    logloss : 0.4549
    Test result:
    auc : 0.779    logloss : 0.4558

    dropout_prob:0.2, learning_rate:0.001, mlp_hidden_size:[40,40,40]
    Valid result:
    auc : 0.7883    logloss : 0.3683
    Test result:
    auc : 0.7854    logloss : 0.369
  ```

- **Logging Result**:

  ```yaml
    69%|██████▉   | 31/45 [1:30:16<40:46, 174.71s/trial, best loss: -0.7924]
    best params:  {'dropout_prob': 0.1, 'learning_rate': 0.001, 'mlp_hidden_size': '[50,50,50]'}
    best result: 
    {'model': 'NFM', 'best_valid_score': 0.7924, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('auc', 0.7924), ('logloss', 0.3658)]), 'test_result': OrderedDict([('auc', 0.7898), ('logloss', 0.3663)])}
  ```