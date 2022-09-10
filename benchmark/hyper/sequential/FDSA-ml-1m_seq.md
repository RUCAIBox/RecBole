# Sequential Recommendation

- **Dataset**: [MovieLens-1m](../../md/ml-1m_seq.md)

- **Model**: [FDSA](https://recbole.io/docs/user_guide/model/sequential/fdsa.html)

- **Time cost**: 9211.53s/trial

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
  learning_rate:0.0005
  Valid result:
  recall@10 : 0.2418    mrr@10 : 0.0918    ndcg@10 : 0.1267    hit@10 : 0.2418    precision@10 : 0.0242
  Test result:
  recall@10 : 0.2428    mrr@10 : 0.0952    ndcg@10 : 0.1294    hit@10 : 0.2428    precision@10 : 0.0243

  learning_rate:0.005
  Valid result:
  recall@10 : 0.219    mrr@10 : 0.085    ndcg@10 : 0.116    hit@10 : 0.219    precision@10 : 0.0219
  Test result:
  recall@10 : 0.2132    mrr@10 : 0.0808    ndcg@10 : 0.1116    hit@10 : 0.2132    precision@10 : 0.0213

  learning_rate:0.0003
  Valid result:
  recall@10 : 0.2344    mrr@10 : 0.0894    ndcg@10 : 0.1231    hit@10 : 0.2344    precision@10 : 0.0234
  Test result:
  recall@10 : 0.2397    mrr@10 : 0.0901    ndcg@10 : 0.1248    hit@10 : 0.2397    precision@10 : 0.024

  learning_rate:0.003
  Valid result:
  recall@10 : 0.2322    mrr@10 : 0.0891    ndcg@10 : 0.1224    hit@10 : 0.2322    precision@10 : 0.0232
  Test result:
  recall@10 : 0.2334    mrr@10 : 0.0912    ndcg@10 : 0.1241    hit@10 : 0.2334    precision@10 : 0.0233

  learning_rate:0.001
  Valid result:
  recall@10 : 0.2369    mrr@10 : 0.0877    ndcg@10 : 0.1223    hit@10 : 0.2369    precision@10 : 0.0237
  Test result:
  recall@10 : 0.2349    mrr@10 : 0.0887    ndcg@10 : 0.1226    hit@10 : 0.2349    precision@10 : 0.0235
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 5/5 [12:47:37<00:00, 9211.53s/trial, best loss: -0.1267]
  best params:  {'learning_rate': 0.0005}
  best result: 
  {'model': 'FDSA', 'best_valid_score': 0.1267, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.2418), ('mrr@10', 0.0918), ('ndcg@10', 0.1267), ('hit@10', 0.2418), ('precision@10', 0.0242)]), 'test_result': OrderedDict([('recall@10', 0.2428), ('mrr@10', 0.0952), ('ndcg@10', 0.1294), ('hit@10', 0.2428), ('precision@10', 0.0243)])}
  ```
