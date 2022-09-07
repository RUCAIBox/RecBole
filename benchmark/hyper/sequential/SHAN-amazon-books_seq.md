# Sequential Recommendation

- **Dataset**: [Amazon-Books](../../md/amazon-books_seq.md)

- **Model**: [SHAN](https://recbole.io/docs/user_guide/model/sequential/shan.html)

- **Time cost**: 18719.36s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [0.0005, 0.001, 0.003]
  short_item_length choice [1, 2, 3]
  ```

- **Best parameters**:

  ```
  learning_rate: 0.003
  short_item_length: 1
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  learning_rate:0.001, short_item_length:1
  Valid result:
  recall@10 : 0.0925    mrr@10 : 0.0239    ndcg@10 : 0.0396    hit@10 : 0.0925    precision@10 : 0.0092
  Test result:
  recall@10 : 0.0646    mrr@10 : 0.0174    ndcg@10 : 0.0282    hit@10 : 0.0646    precision@10 : 0.0065

  learning_rate:0.003, short_item_length:3
  Valid result:
  recall@10 : 0.1195    mrr@10 : 0.035    ndcg@10 : 0.0546    hit@10 : 0.1195    precision@10 : 0.012
  Test result:
  recall@10 : 0.0885    mrr@10 : 0.0257    ndcg@10 : 0.0402    hit@10 : 0.0885    precision@10 : 0.0089

  learning_rate:0.003, short_item_length:1
  Valid result:
  recall@10 : 0.154    mrr@10 : 0.0462    ndcg@10 : 0.0712    hit@10 : 0.154    precision@10 : 0.0154
  Test result:
  recall@10 : 0.1072    mrr@10 : 0.0318    ndcg@10 : 0.0492    hit@10 : 0.1072    precision@10 : 0.0107

  learning_rate:0.001, short_item_length:3
  Valid result:
  recall@10 : 0.0938    mrr@10 : 0.0223    ndcg@10 : 0.0385    hit@10 : 0.0938    precision@10 : 0.0094
  Test result:
  recall@10 : 0.0652    mrr@10 : 0.0162    ndcg@10 : 0.0273    hit@10 : 0.0652    precision@10 : 0.0065

  learning_rate:0.0005, short_item_length:1
  Valid result:
  recall@10 : 0.0925    mrr@10 : 0.0237    ndcg@10 : 0.0395    hit@10 : 0.0925    precision@10 : 0.0093
  Test result:
  recall@10 : 0.0658    mrr@10 : 0.0176    ndcg@10 : 0.0286    hit@10 : 0.0658    precision@10 : 0.0066

  learning_rate:0.0005, short_item_length:2
  Valid result:
  recall@10 : 0.0908    mrr@10 : 0.0222    ndcg@10 : 0.0378    hit@10 : 0.0908    precision@10 : 0.0091
  Test result:
  recall@10 : 0.0635    mrr@10 : 0.0162    ndcg@10 : 0.027    hit@10 : 0.0635    precision@10 : 0.0063

  learning_rate:0.0005, short_item_length:3
  Valid result:
  recall@10 : 0.0904    mrr@10 : 0.0218    ndcg@10 : 0.0374    hit@10 : 0.0904    precision@10 : 0.009
  Test result:
  recall@10 : 0.0615    mrr@10 : 0.0155    ndcg@10 : 0.026    hit@10 : 0.0615    precision@10 : 0.0062

  learning_rate:0.003, short_item_length:2
  Valid result:
  recall@10 : 0.1287    mrr@10 : 0.0375    ndcg@10 : 0.0586    hit@10 : 0.1287    precision@10 : 0.0129
  Test result:
  recall@10 : 0.0909    mrr@10 : 0.0261    ndcg@10 : 0.0411    hit@10 : 0.0909    precision@10 : 0.0091

  learning_rate:0.001, short_item_length:2
  Valid result:
  recall@10 : 0.0923    mrr@10 : 0.0227    ndcg@10 : 0.0386    hit@10 : 0.0923    precision@10 : 0.0092
  Test result:
  recall@10 : 0.0652    mrr@10 : 0.0171    ndcg@10 : 0.0281    hit@10 : 0.0652    precision@10 : 0.0065
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 9/9 [46:47:54<00:00, 18719.36s/trial, best loss: -0.0712]
  best params:  {'learning_rate': 0.003, 'short_item_length': 1}
  best result: 
  {'model': 'SHAN', 'best_valid_score': 0.0712, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.154), ('mrr@10', 0.0462), ('ndcg@10', 0.0712), ('hit@10', 0.154), ('precision@10', 0.0154)]), 'test_result': OrderedDict([('recall@10', 0.1072), ('mrr@10', 0.0318), ('ndcg@10', 0.0492), ('hit@10', 0.1072), ('precision@10', 0.0107)])}
  ```
