# General Recommendation

- **Dataset**: [MovieLens-1m](../../md/ml-1m_general.md)

- **Model**: [ConvNCF](https://recbole.io/docs/user_guide/model/general/convncf.html)

- **Time cost**: 2717.97s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [5e-3, 1e-2, 2e-2] 
  cnn_channels choice ['[1, 32, 32, 32, 32, 32, 32]', '[1, 64, 32, 32, 32, 32]''] 
  cnn_kernels choice ['[2, 2, 2, 2, 2, 2]', '[4, 2, 2, 2, 2]'] 
  cnn_strides choice ['[2, 2, 2, 2, 2, 2]', '[4, 2, 2, 2, 2]'] 
  dropout_prob choice [0.0,0.1,0.2,0.3] 
  reg_weights choice ['[0.0, 0.0]', '[0.1, 0.1]']
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 0.02  
  reg_weights: [0.1,0.1]  
  cnn_channels: [1,64,32,32,32,32]  
  cnn_kernels: [4,2,2,2,2]  
  cnn_strides: [4,2,2,2,2]  
  dropout_prob: 0.3
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  cnn_channels:'[1,32,32,32,32,32,32]', cnn_kernels:'[2,2,2,2,2,2]', cnn_strides:'[2,2,2,2,2,2]', dropout_prob:0.2, learning_rate:0.02, reg_weights:'[0.0,0.0]'
  Valid result:
  recall@10 : 0.1166    mrr@10 : 0.2961    ndcg@10 : 0.1547    hit@10 : 0.6101    precision@10 : 0.1196
  Test result:
  recall@10 : 0.1249    mrr@10 : 0.3407    ndcg@10 : 0.1819    hit@10 : 0.6271    precision@10 : 0.1406

  cnn_channels:'[1,64,32,32,32,32]', cnn_kernels:'[4,2,2,2,2]', cnn_strides:'[4,2,2,2,2]', dropout_prob:0.3, learning_rate:0.02, reg_weights:'[0.1,0.1]'
  Valid result:
  recall@10 : 0.1249    mrr@10 : 0.3105    ndcg@10 : 0.1649    hit@10 : 0.6275    precision@10 : 0.1272
  Test result:
  recall@10 : 0.1343    mrr@10 : 0.3653    ndcg@10 : 0.1969    hit@10 : 0.6473    precision@10 : 0.151

  cnn_channels:'[1,64,32,32,32,32]', cnn_kernels:'[4,2,2,2,2]', cnn_strides:'[4,2,2,2,2]', dropout_prob:0.3, learning_rate:0.01, reg_weights:'[0.0,0.0]'
  Valid result:
  recall@10 : 0.1127    mrr@10 : 0.2952    ndcg@10 : 0.1523    hit@10 : 0.5972    precision@10 : 0.1178
  Test result:
  recall@10 : 0.1234    mrr@10 : 0.3464    ndcg@10 : 0.1835    hit@10 : 0.6206    precision@10 : 0.1411
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 24/24 [18:07:11<00:00, 2717.97s/trial, best loss: -0.1649]
  best params:  {'cnn_channels': '[1,64,32,32,32,32]', 'cnn_kernels': '[4,2,2,2,2]', 'cnn_strides': '[4,2,2,2,2]', 'dropout_prob': 0.3, 'learning_rate': 0.02, 'reg_weights': '[0.1,0.1]'}
  best result: 
  {'model': 'ConvNCF', 'best_valid_score': 0.1649, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.1249), ('mrr@10', 0.3105), ('ndcg@10', 0.1649), ('hit@10', 0.6275), ('precision@10', 0.1272)]), 'test_result': OrderedDict([('recall@10', 0.1343), ('mrr@10', 0.3653), ('ndcg@10', 0.1969), ('hit@10', 0.6473), ('precision@10', 0.151)])}
  ```