# General Recommendation

- **Dataset**: [Amazon-Books](../../md/amazon-books_general.md)

- **Model**: [CDAE](https://recbole.io/docs/user_guide/model/general/cdae.html)

- **Time cost**: 1329.64s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [1e-2,1e-3,5e-3,5e-4]
  corruption_ratio choice [0.5,0.1]
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
  corruption_ratio:0.5, learning_rate:0.005, reg_weight_1:0.0, reg_weight_2:0.01
  Valid result:
  recall@10 : 0.1853    mrr@10 : 0.1437    ndcg@10 : 0.1283    hit@10 : 0.2916    precision@10 : 0.0363
  Test result:
  recall@10 : 0.1915    mrr@10 : 0.1571    ndcg@10 : 0.1377    hit@10 : 0.298     precision@10 : 0.0382

  corruption_ratio:0.5, learning_rate:0.005, reg_weight_1:0.01, reg_weight_2:0.0
  Valid result:
  recall@10 : 0.1893    mrr@10 : 0.1446    ndcg@10 : 0.1303    hit@10 : 0.295     precision@10 : 0.0365
  Test result:
  recall@10 : 0.1932    mrr@10 : 0.1571    ndcg@10 : 0.139     hit@10 : 0.2984    precision@10 : 0.038
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 32/32 [11:49:51<00:00, 1329.64s/trial, best loss: -0.1303]
  best params:  {'corruption_ratio': 0.5, 'learning_rate': 0.005, 'reg_weight_1': 0.01, 'reg_weight_2': 0.0}
  best result: 
  {'model': 'CDAE', 'best_valid_score': 0.1303, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.1893), ('mrr@10', 0.1446), ('ndcg@10', 0.1303), ('hit@10', 0.295), ('precision@10', 0.0365)]), 'test_result': OrderedDict([('recall@10', 0.1932), ('mrr@10', 0.1571), ('ndcg@10', 0.139), ('hit@10', 0.2984), ('precision@10', 0.038)])}
  ```
