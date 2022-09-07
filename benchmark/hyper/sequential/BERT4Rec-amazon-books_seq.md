# Sequential Recommendation

- **Dataset**: [Amazon-Books](../../md/amazon-books_seq.md)

- **Model**: [BERT4Rec](https://recbole.io/docs/user_guide/model/sequential/bert4rec.html)

- **Time cost**: 24425.43s/trial

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
  learning_rate:0.005
  Valid result:
  recall@10 : 0.0685    mrr@10 : 0.0217    ndcg@10 : 0.0324    hit@10 : 0.0685    precision@10 : 0.0069
  Test result:
  recall@10 : 0.0456    mrr@10 : 0.0142    ndcg@10 : 0.0214    hit@10 : 0.0456    precision@10 : 0.0046

  learning_rate:0.0003
  Valid result:
  recall@10 : 0.0831    mrr@10 : 0.0255    ndcg@10 : 0.0387    hit@10 : 0.0831    precision@10 : 0.0083
  Test result:
  recall@10 : 0.0526    mrr@10 : 0.016    ndcg@10 : 0.0244    hit@10 : 0.0526    precision@10 : 0.0053

  learning_rate:0.001
  Valid result:
  recall@10 : 0.0738    mrr@10 : 0.023    ndcg@10 : 0.0346    hit@10 : 0.0738    precision@10 : 0.0074
  Test result:
  recall@10 : 0.0474    mrr@10 : 0.0149    ndcg@10 : 0.0224    hit@10 : 0.0474    precision@10 : 0.0047

  learning_rate:0.0005
  Valid result:
  recall@10 : 0.0746    mrr@10 : 0.0235    ndcg@10 : 0.0353    hit@10 : 0.0746    precision@10 : 0.0075
  Test result:
  recall@10 : 0.0487    mrr@10 : 0.0154    ndcg@10 : 0.023    hit@10 : 0.0487    precision@10 : 0.0049

  learning_rate:0.003
  Valid result:
  recall@10 : 0.0706    mrr@10 : 0.0214    ndcg@10 : 0.0327    hit@10 : 0.0706    precision@10 : 0.0071
  Test result:
  recall@10 : 0.047    mrr@10 : 0.0147    ndcg@10 : 0.0221    hit@10 : 0.047    precision@10 : 0.0047
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 5/5 [33:55:27<00:00, 24425.43s/trial, best loss: -0.0387]
  best params:  {'learning_rate': 0.0003}
  best result: 
  {'model': 'BERT4Rec', 'best_valid_score': 0.0387, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.0831), ('mrr@10', 0.0255), ('ndcg@10', 0.0387), ('hit@10', 0.0831), ('precision@10', 0.0083)]), 'test_result': OrderedDict([('recall@10', 0.0526), ('mrr@10', 0.016), ('ndcg@10', 0.0244), ('hit@10', 0.0526), ('precision@10', 0.0053)])}
  ```
