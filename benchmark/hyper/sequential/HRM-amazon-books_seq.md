# Sequential Recommendation

- **Dataset**: [Amazon-Books](../../md/amazon-books_seq.md)

- **Model**: [HRM](https://recbole.io/docs/user_guide/model/sequential/hrm.html)

- **Time cost**: 3616.11s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [0.0005, 0.001, 0.003]
  high_order choice [1, 2, 3]
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 0.0005
  high_order: 1
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  high_order:1, learning_rate:0.003
  Valid result:
  recall@10 : 0.176    mrr@10 : 0.0523    ndcg@10 : 0.0813    hit@10 : 0.176    precision@10 : 0.0176
  Test result:
  recall@10 : 0.1205    mrr@10 : 0.039    ndcg@10 : 0.0579    hit@10 : 0.1205    precision@10 : 0.012

  high_order:3, learning_rate:0.0005
  Valid result:
  recall@10 : 0.1731    mrr@10 : 0.051    ndcg@10 : 0.0795    hit@10 : 0.1731    precision@10 : 0.0173
  Test result:
  recall@10 : 0.1231    mrr@10 : 0.0374    ndcg@10 : 0.0572    hit@10 : 0.1231    precision@10 : 0.0123

  high_order:1, learning_rate:0.001
  Valid result:
  recall@10 : 0.1752    mrr@10 : 0.0554    ndcg@10 : 0.0836    hit@10 : 0.1752    precision@10 : 0.0175
  Test result:
  recall@10 : 0.1242    mrr@10 : 0.0411    ndcg@10 : 0.0604    hit@10 : 0.1242    precision@10 : 0.0124

  high_order:2, learning_rate:0.0005
  Valid result:
  recall@10 : 0.1797    mrr@10 : 0.0547    ndcg@10 : 0.0839    hit@10 : 0.1797    precision@10 : 0.018
  Test result:
  recall@10 : 0.1304    mrr@10 : 0.041    ndcg@10 : 0.0618    hit@10 : 0.1304    precision@10 : 0.013

  high_order:2, learning_rate:0.001
  Valid result:
  recall@10 : 0.1765    mrr@10 : 0.053    ndcg@10 : 0.0819    hit@10 : 0.1765    precision@10 : 0.0176
  Test result:
  recall@10 : 0.1267    mrr@10 : 0.0396    ndcg@10 : 0.0599    hit@10 : 0.1267    precision@10 : 0.0127

  high_order:1, learning_rate:0.0005
  Valid result:
  recall@10 : 0.1823    mrr@10 : 0.0583    ndcg@10 : 0.0875    hit@10 : 0.1823    precision@10 : 0.0182
  Test result:
  recall@10 : 0.132    mrr@10 : 0.0441    ndcg@10 : 0.0646    hit@10 : 0.132    precision@10 : 0.0132

  high_order:2, learning_rate:0.003
  Valid result:
  recall@10 : 0.1703    mrr@10 : 0.0507    ndcg@10 : 0.0786    hit@10 : 0.1703    precision@10 : 0.017
  Test result:
  recall@10 : 0.1197    mrr@10 : 0.0367    ndcg@10 : 0.0559    hit@10 : 0.1197    precision@10 : 0.012

  high_order:3, learning_rate:0.003
  Valid result:
  recall@10 : 0.1634    mrr@10 : 0.0471    ndcg@10 : 0.0741    hit@10 : 0.1634    precision@10 : 0.0163
  Test result:
  recall@10 : 0.1127    mrr@10 : 0.0331    ndcg@10 : 0.0514    hit@10 : 0.1127    precision@10 : 0.0113

  high_order:3, learning_rate:0.001
  Valid result:
  recall@10 : 0.1702    mrr@10 : 0.0499    ndcg@10 : 0.0779    hit@10 : 0.1702    precision@10 : 0.017
  Test result:
  recall@10 : 0.1214    mrr@10 : 0.0357    ndcg@10 : 0.0555    hit@10 : 0.1214    precision@10 : 0.0121
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 9/9 [9:02:24<00:00, 3616.11s/trial, best loss: -0.0875]
  best params:  {'high_order': 1, 'learning_rate': 0.0005}
  best result: 
  {'model': 'HRM', 'best_valid_score': 0.0875, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.1823), ('mrr@10', 0.0583), ('ndcg@10', 0.0875), ('hit@10', 0.1823), ('precision@10', 0.0182)]), 'test_result': OrderedDict([('recall@10', 0.132), ('mrr@10', 0.0441), ('ndcg@10', 0.0646), ('hit@10', 0.132), ('precision@10', 0.0132)])}
  ```
