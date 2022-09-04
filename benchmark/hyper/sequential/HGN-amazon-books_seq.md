# Sequential Recommendation

- **Dataset**: [Amazon-Books](../../md/amazon-books_seq.md)

- **Model**: [HGN](https://recbole.io/docs/user_guide/model/sequential/hgn.html)

- **Time cost**: 4766.50s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [0.0003, 0.0005, 0.001, 0.003, 0.005]
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 0.001
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  learning_rate:0.005
  Valid result:
  recall@10 : 0.1636    mrr@10 : 0.0786    ndcg@10 : 0.0985    hit@10 : 0.1636    precision@10 : 0.0164
  Test result:
  recall@10 : 0.0983    mrr@10 : 0.0457    ndcg@10 : 0.058    hit@10 : 0.0983    precision@10 : 0.0098

  learning_rate:0.003
  Valid result:
  recall@10 : 0.1753    mrr@10 : 0.0791    ndcg@10 : 0.1016    hit@10 : 0.1753    precision@10 : 0.0175
  Test result:
  recall@10 : 0.1116    mrr@10 : 0.0485    ndcg@10 : 0.0632    hit@10 : 0.1116    precision@10 : 0.0112

  learning_rate:0.0003
  Valid result:
  recall@10 : 0.1731    mrr@10 : 0.0816    ndcg@10 : 0.103    hit@10 : 0.1731    precision@10 : 0.0173
  Test result:
  recall@10 : 0.1143    mrr@10 : 0.0505    ndcg@10 : 0.0653    hit@10 : 0.1143    precision@10 : 0.0114

  learning_rate:0.0005
  Valid result:
  recall@10 : 0.1786    mrr@10 : 0.0807    ndcg@10 : 0.1036    hit@10 : 0.1786    precision@10 : 0.0179
  Test result:
  recall@10 : 0.1182    mrr@10 : 0.0506    ndcg@10 : 0.0664    hit@10 : 0.1182    precision@10 : 0.0118

  learning_rate:0.001
  Valid result:
  recall@10 : 0.1813    mrr@10 : 0.0809    ndcg@10 : 0.1044    hit@10 : 0.1813    precision@10 : 0.0181
  Test result:
  recall@10 : 0.118    mrr@10 : 0.0508    ndcg@10 : 0.0665    hit@10 : 0.118    precision@10 : 0.0118
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 5/5 [6:37:12<00:00, 4766.50s/trial, best loss: -0.1044]
  best params:  {'learning_rate': 0.001}
  best result: 
  {'model': 'HGN', 'best_valid_score': 0.1044, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.1813), ('mrr@10', 0.0809), ('ndcg@10', 0.1044), ('hit@10', 0.1813), ('precision@10', 0.0181)]), 'test_result': OrderedDict([('recall@10', 0.118), ('mrr@10', 0.0508), ('ndcg@10', 0.0665), ('hit@10', 0.118), ('precision@10', 0.0118)])}
  ```
