# General Recommendation

- **Dataset**: [MovieLens-1m](../../md/ml-1m_general.md)

- **Model**: [CDAE](https://recbole.io/docs/user_guide/model/general/cdae.html)

- **Time cost**: 3233.25s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [1e-2,1e-3,5e-3,5e-4]
  loss_type choice ['BCE','MSE']
  corruption_ratio choice [0.5,0.3,0.1]
  reg_weight_1 choice [0.0,0.01]
  reg_weight_2 choice [0.0,0.01]
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 0.005
  corruption_ratio: 0.5
  reg_weight_1: 0.01
  reg_weight_2: 0.0
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  corruption_ratio:0.5, learning_rate:0.01, loss_type:'MSE', reg_weight_1:0.01, reg_weight_2:0.0
  Valid result:
  recall@10 : 0.0941    mrr@10 : 0.2867    ndcg@10 : 0.135    hit@10 : 0.5332    precision@10 : 0.0951
  Test result:
  recall@10 : 0.0968    mrr@10 : 0.3317    ndcg@10 : 0.15     hit@10 : 0.5496    precision@10 : 0.1008

  corruption_ratio:0.3, learning_rate:0.005, loss_type:'MSE', reg_weight_1:0.01, reg_weight_2:0.0
  Valid result:
  recall@10 : 0.1037    mrr@10 : 0.3116    ndcg@10 : 0.148     hit@10 : 0.5624    precision@10 : 0.1036
  Test result:
  recall@10 : 0.1106    mrr@10 : 0.3653    ndcg@10 : 0.1703    hit@10 : 0.5874    precision@10 : 0.1146
  
  corruption_ratio:0.1, learning_rate:0.01, loss_type:'BCE', reg_weight_1:0.0, reg_weight_2:0.01
  Valid result:
  recall@10 : 0.1713    mrr@10 : 0.3729    ndcg@10 : 0.208     hit@10 : 0.7217    precision@10 : 0.151
  Test result:
  recall@10 : 0.1892    mrr@10 : 0.434     ndcg@10 : 0.2481    hit@10 : 0.7416    precision@10 : 0.1795
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 96/96 [86:13:11<00:00, 3233.25s/trial, best loss: -0.2131]
  best params:  {'corruption_ratio': 0.3, 'learning_rate': 0.01, 'loss_type': 'BCE', 'reg_weight_1': 0.01, 'reg_weight_2': 0.0}
  best result: 
  {'model': 'CDAE', 'best_valid_score': 0.2131, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.1781), ('mrr@10', 0.3806), ('ndcg@10', 0.2131), ('hit@10', 0.733), ('precision@10', 0.1527)]), 'test_result': OrderedDict([('recall@10', 0.1936), ('mrr@10', 0.4359), ('ndcg@10', 0.25), ('hit@10', 0.7515), ('precision@10', 0.1806)])}
  ```