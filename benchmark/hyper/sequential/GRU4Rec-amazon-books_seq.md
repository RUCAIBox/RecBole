# Sequential Recommendation

- **Dataset**: [Amazon-Books](../../md/amazon-books_seq.md)

- **Model**: [GRU4Rec](https://recbole.io/docs/user_guide/model/sequential/gru4rec.html)

- **Time cost**: 8549.40s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [0.005,0.001,0.0005,0.0001]
  num_layers choice [1,2]
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 0.001
  num_layers: 1
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  learning_rate:0.001, num_layers:1
  Valid result:
  recall@10 : 0.2367    mrr@10 : 0.1278    ndcg@10 : 0.1534    hit@10 : 0.2367    precision@10 : 0.0237
  Test result:
  recall@10 : 0.188    mrr@10 : 0.0966    ndcg@10 : 0.1181    hit@10 : 0.188    precision@10 : 0.0188

  learning_rate:0.0001, num_layers:1
  Valid result:
  recall@10 : 0.229    mrr@10 : 0.1186    ndcg@10 : 0.1445    hit@10 : 0.229    precision@10 : 0.0229
  Test result:
  recall@10 : 0.178    mrr@10 : 0.0891    ndcg@10 : 0.1099    hit@10 : 0.178    precision@10 : 0.0178

  learning_rate:0.0005, num_layers:2
  Valid result:
  recall@10 : 0.2316    mrr@10 : 0.1235    ndcg@10 : 0.1489    hit@10 : 0.2316    precision@10 : 0.0232
  Test result:
  recall@10 : 0.1787    mrr@10 : 0.0935    ndcg@10 : 0.1135    hit@10 : 0.1787    precision@10 : 0.0179

  learning_rate:0.001, num_layers:2
  Valid result:
  recall@10 : 0.2313    mrr@10 : 0.1247    ndcg@10 : 0.1498    hit@10 : 0.2313    precision@10 : 0.0231
  Test result:
  recall@10 : 0.18    mrr@10 : 0.0929    ndcg@10 : 0.1134    hit@10 : 0.18    precision@10 : 0.018

  learning_rate:0.005, num_layers:2
  Valid result:
  recall@10 : 0.2423    mrr@10 : 0.1252    ndcg@10 : 0.1528    hit@10 : 0.2423    precision@10 : 0.0242
  Test result:
  recall@10 : 0.1938    mrr@10 : 0.0986    ndcg@10 : 0.121    hit@10 : 0.1938    precision@10 : 0.0194

  learning_rate:0.0005, num_layers:1
  Valid result:
  recall@10 : 0.2326    mrr@10 : 0.1258    ndcg@10 : 0.1509    hit@10 : 0.2326    precision@10 : 0.0233
  Test result:
  recall@10 : 0.1811    mrr@10 : 0.0958    ndcg@10 : 0.1159    hit@10 : 0.1811    precision@10 : 0.0181

  learning_rate:0.005, num_layers:1
  Valid result:
  recall@10 : 0.2409    mrr@10 : 0.1255    ndcg@10 : 0.1527    hit@10 : 0.2409    precision@10 : 0.0241
  Test result:
  recall@10 : 0.1963    mrr@10 : 0.1015    ndcg@10 : 0.1238    hit@10 : 0.1963    precision@10 : 0.0196

  learning_rate:0.0001, num_layers:2
  Valid result:
  recall@10 : 0.2228    mrr@10 : 0.1159    ndcg@10 : 0.141    hit@10 : 0.2228    precision@10 : 0.0223
  Test result:
  recall@10 : 0.1748    mrr@10 : 0.0868    ndcg@10 : 0.1074    hit@10 : 0.1748    precision@10 : 0.0175
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 8/8 [18:59:55<00:00, 8549.40s/trial, best loss: -0.1534]
  best params:  {'learning_rate': 0.001, 'num_layers': 1}
  best result: 
  {'model': 'GRU4Rec', 'best_valid_score': 0.1534, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.2367), ('mrr@10', 0.1278), ('ndcg@10', 0.1534), ('hit@10', 0.2367), ('precision@10', 0.0237)]), 'test_result': OrderedDict([('recall@10', 0.188), ('mrr@10', 0.0966), ('ndcg@10', 0.1181), ('hit@10', 0.188), ('precision@10', 0.0188)])}
  ```
