# Sequential Recommendation

- **Dataset**: [Amazon-Books](../../md/amazon-books_seq.md)

- **Model**: [FPMC](https://recbole.io/docs/user_guide/model/sequential/fpmc.html)

- **Time cost**: 4628.09s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [0.0003, 0.0005, 0.001, 0.003, 0.005]
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 0.0003
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  learning_rate:0.003
  Valid result:
  recall@10 : 0.1496    mrr@10 : 0.0728    ndcg@10 : 0.0908    hit@10 : 0.1496    precision@10 : 0.015
  Test result:
  recall@10 : 0.1106    mrr@10 : 0.0519    ndcg@10 : 0.0657    hit@10 : 0.1106    precision@10 : 0.0111

  learning_rate:0.005
  Valid result:
  recall@10 : 0.1155    mrr@10 : 0.0564    ndcg@10 : 0.0701    hit@10 : 0.1155    precision@10 : 0.0115
  Test result:
  recall@10 : 0.0866    mrr@10 : 0.0404    ndcg@10 : 0.0512    hit@10 : 0.0866    precision@10 : 0.0087

  learning_rate:0.0003
  Valid result:
  recall@10 : 0.1918    mrr@10 : 0.0967    ndcg@10 : 0.119    hit@10 : 0.1918    precision@10 : 0.0192
  Test result:
  recall@10 : 0.1464    mrr@10 : 0.0717    ndcg@10 : 0.0892    hit@10 : 0.1464    precision@10 : 0.0146

  learning_rate:0.0005
  Valid result:
  recall@10 : 0.1843    mrr@10 : 0.0918    ndcg@10 : 0.1135    hit@10 : 0.1843    precision@10 : 0.0184
  Test result:
  recall@10 : 0.1401    mrr@10 : 0.0682    ndcg@10 : 0.085    hit@10 : 0.1401    precision@10 : 0.014

  learning_rate:0.001
  Valid result:
  recall@10 : 0.1822    mrr@10 : 0.0911    ndcg@10 : 0.1124    hit@10 : 0.1822    precision@10 : 0.0182
  Test result:
  recall@10 : 0.14    mrr@10 : 0.0673    ndcg@10 : 0.0842    hit@10 : 0.14    precision@10 : 0.014
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 5/5 [6:25:40<00:00, 4628.09s/trial, best loss: -0.119]
  best params:  {'learning_rate': 0.0003}
  best result: 
  {'model': 'FPMC', 'best_valid_score': 0.119, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.1918), ('mrr@10', 0.0967), ('ndcg@10', 0.119), ('hit@10', 0.1918), ('precision@10', 0.0192)]), 'test_result': OrderedDict([('recall@10', 0.1464), ('mrr@10', 0.0717), ('ndcg@10', 0.0892), ('hit@10', 0.1464), ('precision@10', 0.0146)])}
  ```
