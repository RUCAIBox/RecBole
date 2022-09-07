# Sequential Recommendation

- **Dataset**: [Amazon-Books](../../md/amazon-books_seq.md)

- **Model**: [GRU4RecF](https://recbole.io/docs/user_guide/model/sequential/gru4recf.html)

- **Time cost**: 10319.41s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [0.005,0.001,0.0005,0.0001]
  num_layers choice [1,2]
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 0.005
  num_layers: 1
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  learning_rate:0.0005, num_layers:1
  Valid result:
  recall@10 : 0.2222    mrr@10 : 0.1159    ndcg@10 : 0.1409    hit@10 : 0.2222    precision@10 : 0.0222
  Test result:
  recall@10 : 0.1747    mrr@10 : 0.0895    ndcg@10 : 0.1095    hit@10 : 0.1747    precision@10 : 0.0175

  learning_rate:0.005, num_layers:1
  Valid result:
  recall@10 : 0.2443    mrr@10 : 0.1271    ndcg@10 : 0.1546    hit@10 : 0.2443    precision@10 : 0.0244
  Test result:
  recall@10 : 0.1982    mrr@10 : 0.1012    ndcg@10 : 0.124    hit@10 : 0.1982    precision@10 : 0.0198

  learning_rate:0.0001, num_layers:1
  Valid result:
  recall@10 : 0.2087    mrr@10 : 0.1058    ndcg@10 : 0.1299    hit@10 : 0.2087    precision@10 : 0.0209
  Test result:
  recall@10 : 0.1584    mrr@10 : 0.0795    ndcg@10 : 0.098    hit@10 : 0.1584    precision@10 : 0.0158

  learning_rate:0.001, num_layers:2
  Valid result:
  recall@10 : 0.2324    mrr@10 : 0.1195    ndcg@10 : 0.146    hit@10 : 0.2324    precision@10 : 0.0232
  Test result:
  recall@10 : 0.1839    mrr@10 : 0.0937    ndcg@10 : 0.1149    hit@10 : 0.1839    precision@10 : 0.0184

  learning_rate:0.001, num_layers:1
  Valid result:
  recall@10 : 0.232    mrr@10 : 0.1228    ndcg@10 : 0.1485    hit@10 : 0.232    precision@10 : 0.0232
  Test result:
  recall@10 : 0.1809    mrr@10 : 0.0929    ndcg@10 : 0.1136    hit@10 : 0.1809    precision@10 : 0.0181

  learning_rate:0.0001, num_layers:2
  Valid result:
  recall@10 : 0.2106    mrr@10 : 0.1075    ndcg@10 : 0.1317    hit@10 : 0.2106    precision@10 : 0.0211
  Test result:
  recall@10 : 0.1594    mrr@10 : 0.0798    ndcg@10 : 0.0984    hit@10 : 0.1594    precision@10 : 0.0159

  learning_rate:0.005, num_layers:2
  Valid result:
  recall@10 : 0.2369    mrr@10 : 0.1165    ndcg@10 : 0.1449    hit@10 : 0.2369    precision@10 : 0.0237
  Test result:
  recall@10 : 0.1912    mrr@10 : 0.0924    ndcg@10 : 0.1155    hit@10 : 0.1912    precision@10 : 0.0191

  learning_rate:0.0005, num_layers:2
  Valid result:
  recall@10 : 0.2255    mrr@10 : 0.1158    ndcg@10 : 0.1416    hit@10 : 0.2255    precision@10 : 0.0225
  Test result:
  recall@10 : 0.1755    mrr@10 : 0.0877    ndcg@10 : 0.1083    hit@10 : 0.1755    precision@10 : 0.0176
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 8/8 [22:55:55<00:00, 10319.41s/trial, best loss: -0.1546]
  best params:  {'learning_rate': 0.005, 'num_layers': 1}
  best result: 
  {'model': 'GRU4RecF', 'best_valid_score': 0.1546, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.2443), ('mrr@10', 0.1271), ('ndcg@10', 0.1546), ('hit@10', 0.2443), ('precision@10', 0.0244)]), 'test_result': OrderedDict([('recall@10', 0.1982), ('mrr@10', 0.1012), ('ndcg@10', 0.124), ('hit@10', 0.1982), ('precision@10', 0.0198)])}
  ```
