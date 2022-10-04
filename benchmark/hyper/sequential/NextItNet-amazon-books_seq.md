# Sequential Recommendation

- **Dataset**: [Amazon-Books](../../md/amazon-books_seq.md)

- **Model**: [NextItNet](https://recbole.io/docs/user_guide/model/sequential/nextitnet.html)

- **Time cost**: 14503.48s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [0.0005, 0.001, 0.003]
  kernel_size choice [2, 3, 4]
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 0.001
  kernel_size: 4
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  kernel_size:2, learning_rate:0.001
  Valid result:
  recall@10 : 0.1599    mrr@10 : 0.0729    ndcg@10 : 0.0933    hit@10 : 0.1599    precision@10 : 0.016
  Test result:
  recall@10 : 0.1148    mrr@10 : 0.0514    ndcg@10 : 0.0662    hit@10 : 0.1148    precision@10 : 0.0115

  kernel_size:4, learning_rate:0.001
  Valid result:
  recall@10 : 0.1773    mrr@10 : 0.0835    ndcg@10 : 0.1055    hit@10 : 0.1773    precision@10 : 0.0177
  Test result:
  recall@10 : 0.1291    mrr@10 : 0.0588    ndcg@10 : 0.0752    hit@10 : 0.1291    precision@10 : 0.0129

  kernel_size:4, learning_rate:0.0005
  Valid result:
  recall@10 : 0.1803    mrr@10 : 0.0809    ndcg@10 : 0.1042    hit@10 : 0.1803    precision@10 : 0.018
  Test result:
  recall@10 : 0.1287    mrr@10 : 0.0566    ndcg@10 : 0.0734    hit@10 : 0.1287    precision@10 : 0.0129

  kernel_size:3, learning_rate:0.0005
  Valid result:
  recall@10 : 0.17    mrr@10 : 0.0787    ndcg@10 : 0.1    hit@10 : 0.17    precision@10 : 0.017
  Test result:
  recall@10 : 0.1246    mrr@10 : 0.056    ndcg@10 : 0.072    hit@10 : 0.1246    precision@10 : 0.0125

  kernel_size:4, learning_rate:0.003
  Valid result:
  recall@10 : 0.1738    mrr@10 : 0.0824    ndcg@10 : 0.1038    hit@10 : 0.1738    precision@10 : 0.0174
  Test result:
  recall@10 : 0.1294    mrr@10 : 0.0602    ndcg@10 : 0.0764    hit@10 : 0.1294    precision@10 : 0.0129

  kernel_size:3, learning_rate:0.003
  Valid result:
  recall@10 : 0.168    mrr@10 : 0.0779    ndcg@10 : 0.099    hit@10 : 0.168    precision@10 : 0.0168
  Test result:
  recall@10 : 0.1218    mrr@10 : 0.0545    ndcg@10 : 0.0703    hit@10 : 0.1218    precision@10 : 0.0122

  kernel_size:2, learning_rate:0.0005
  Valid result:
  recall@10 : 0.1592    mrr@10 : 0.0697    ndcg@10 : 0.0906    hit@10 : 0.1592    precision@10 : 0.0159
  Test result:
  recall@10 : 0.115    mrr@10 : 0.0487    ndcg@10 : 0.0642    hit@10 : 0.115    precision@10 : 0.0115

  kernel_size:2, learning_rate:0.003
  Valid result:
  recall@10 : 0.156    mrr@10 : 0.0694    ndcg@10 : 0.0896    hit@10 : 0.156    precision@10 : 0.0156
  Test result:
  recall@10 : 0.1132    mrr@10 : 0.0495    ndcg@10 : 0.0643    hit@10 : 0.1132    precision@10 : 0.0113

  kernel_size:3, learning_rate:0.001
  Valid result:
  recall@10 : 0.1731    mrr@10 : 0.0794    ndcg@10 : 0.1013    hit@10 : 0.1731    precision@10 : 0.0173
  Test result:
  recall@10 : 0.1214    mrr@10 : 0.0536    ndcg@10 : 0.0694    hit@10 : 0.1214    precision@10 : 0.0121
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 9/9 [36:15:31<00:00, 14503.48s/trial, best loss: -0.1055]
  best params:  {'kernel_size': 4, 'learning_rate': 0.001}
  best result: 
  {'model': 'NextItNet', 'best_valid_score': 0.1055, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.1773), ('mrr@10', 0.0835), ('ndcg@10', 0.1055), ('hit@10', 0.1773), ('precision@10', 0.0177)]), 'test_result': OrderedDict([('recall@10', 0.1291), ('mrr@10', 0.0588), ('ndcg@10', 0.0752), ('hit@10', 0.1291), ('precision@10', 0.0129)])}
  ```
