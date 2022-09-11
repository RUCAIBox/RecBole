# General Recommendation

- **Dataset**: [Amazon-Books](../../md/amazon-books_general.md)

- **Model**: [MultiDAE](https://recbole.io/docs/user_guide/model/general/multidae.html)

- **Time cost**: 1553.21s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [5e-5,1e-4,5e-4,7e-4,1e-3,5e-3,7e-3]
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 0.0005
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  learning_rate:0.0005
  Valid result:
  recall@10 : 0.1808    mrr@10 : 0.1321    ndcg@10 : 0.121     hit@10 : 0.284     precision@10 : 0.0352
  Test result:
  recall@10 : 0.1838    mrr@10 : 0.1442    ndcg@10 : 0.1292    hit@10 : 0.2868    precision@10 : 0.0367

  learning_rate:0.001
  Valid result:
  recall@10 : 0.1799    mrr@10 : 0.1318    ndcg@10 : 0.1207    hit@10 : 0.284     precision@10 : 0.0351
  Test result:
  recall@10 : 0.1838    mrr@10 : 0.1445    ndcg@10 : 0.1294    hit@10 : 0.2867    precision@10 : 0.0364

  learning_rate:0.0007
  Valid result:
  recall@10 : 0.1791    mrr@10 : 0.132     ndcg@10 : 0.1205    hit@10 : 0.2819    precision@10 : 0.0349
  Test result:
  recall@10 : 0.1836    mrr@10 : 0.1439    ndcg@10 : 0.1291    hit@10 : 0.2867    precision@10 : 0.0366
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 7/7 [3:01:12<00:00, 1553.21s/trial, best loss: -0.121]
  best params:  {'learning_rate': 0.0005}
  best result: 
  {'model': 'MultiDAE', 'best_valid_score': 0.121, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.1808), ('mrr@10', 0.1321), ('ndcg@10', 0.121), ('hit@10', 0.284), ('precision@10', 0.0352)]), 'test_result': OrderedDict([('recall@10', 0.1838), ('mrr@10', 0.1442), ('ndcg@10', 0.1292), ('hit@10', 0.2868), ('precision@10', 0.0367)])}
  ```
