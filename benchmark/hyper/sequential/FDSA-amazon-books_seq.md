# Sequential Recommendation

- **Dataset**: [Amazon-Books](../../md/amazon-books_seq.md)

- **Model**: [FDSA](https://recbole.io/docs/user_guide/model/sequential/fdsa.html)

- **Time cost**: 22970.74s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate in [0.0003, 0.0005, 0.001, 0.003, 0.005]
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 0.0005
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  learning_rate:0.001
  Valid result:
  recall@10 : 0.1967    mrr@10 : 0.084    ndcg@10 : 0.1104    hit@10 : 0.1967    precision@10 : 0.0197
  Test result:
  recall@10 : 0.1535    mrr@10 : 0.064    ndcg@10 : 0.0849    hit@10 : 0.1535    precision@10 : 0.0153

  learning_rate:0.0003
  Valid result:
  recall@10 : 0.2062    mrr@10 : 0.0885    ndcg@10 : 0.116    hit@10 : 0.2062    precision@10 : 0.0206
  Test result:
  recall@10 : 0.1654    mrr@10 : 0.0701    ndcg@10 : 0.0924    hit@10 : 0.1654    precision@10 : 0.0165

  learning_rate:0.005
  Valid result:
  recall@10 : 0.1685    mrr@10 : 0.0693    ndcg@10 : 0.0924    hit@10 : 0.1685    precision@10 : 0.0168
  Test result:
  recall@10 : 0.1318    mrr@10 : 0.0534    ndcg@10 : 0.0717    hit@10 : 0.1318    precision@10 : 0.0132

  learning_rate:0.003
  Valid result:
  recall@10 : 0.1885    mrr@10 : 0.0792    ndcg@10 : 0.1048    hit@10 : 0.1885    precision@10 : 0.0189
  Test result:
  recall@10 : 0.149    mrr@10 : 0.062    ndcg@10 : 0.0822    hit@10 : 0.149    precision@10 : 0.0149

  learning_rate:0.0005
  Valid result:
  recall@10 : 0.2074    mrr@10 : 0.0892    ndcg@10 : 0.1168    hit@10 : 0.2074    precision@10 : 0.0207
  Test result:
  recall@10 : 0.1643    mrr@10 : 0.0696    ndcg@10 : 0.0917    hit@10 : 0.1643    precision@10 : 0.0164
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 5/5 [31:54:13<00:00, 22970.74s/trial, best loss: -0.1168]
  best params:  {'learning_rate': 0.0005}
  best result: 
  {'model': 'FDSA', 'best_valid_score': 0.1168, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.2074), ('mrr@10', 0.0892), ('ndcg@10', 0.1168), ('hit@10', 0.2074), ('precision@10', 0.0207)]), 'test_result': OrderedDict([('recall@10', 0.1643), ('mrr@10', 0.0696), ('ndcg@10', 0.0917), ('hit@10', 0.1643), ('precision@10', 0.0164)])}
  ```
