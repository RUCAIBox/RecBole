# Sequential Recommendation

- **Dataset**: [Yelp](../../md/yelp_seq.md)

- **Model**: [BERT4Rec](https://recbole.io/docs/user_guide/model/sequential/bert4rec.html)

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [0.005,0.001,0.0005,0.0001]
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 0.0001
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  learning_rate:0.005
  Valid result:
  recall@10 : 0.0468    mrr@10 : 0.0155    ndcg@10 : 0.0226    hit@10 : 0.0468    precision@10 : 0.0047
  Test result:
  recall@10 : 0.0433    mrr@10 : 0.015    ndcg@10 : 0.0215    hit@10 : 0.0433    precision@10 : 0.0043

  learning_rate:0.001
  Valid result:
  recall@10 : 0.0532    mrr@10 : 0.0173    ndcg@10 : 0.0256    hit@10 : 0.0532    precision@10 : 0.0053
  Test result:
  recall@10 : 0.0485    mrr@10 : 0.0166    ndcg@10 : 0.0239    hit@10 : 0.0485    precision@10 : 0.0048

  learning_rate:0.0005
  Valid result:
  recall@10 : 0.0514    mrr@10 : 0.0172    ndcg@10 : 0.0251    hit@10 : 0.0514    precision@10 : 0.0051
  Test result:
  recall@10 : 0.0474    mrr@10 : 0.0159    ndcg@10 : 0.0231    hit@10 : 0.0474    precision@10 : 0.0047

  learning_rate:0.0001
  Valid result:
  recall@10 : 0.0533    mrr@10 : 0.018    ndcg@10 : 0.0261    hit@10 : 0.0533    precision@10 : 0.0053
  Test result:
  recall@10 : 0.0487    mrr@10 : 0.0162    ndcg@10 : 0.0237    hit@10 : 0.0487    precision@10 : 0.0049
  ```

- **Logging Result**:

  ```yaml
  best params:  {'learning_rate': 0.0001}
  best result: 
  {'model': 'BERT4Rec', 'best_valid_result': OrderedDict([('recall@10', 0.0533), ('mrr@10', 0.018), ('ndcg@10', 0.0261), ('hit@10', 0.0533), ('precision@10', 0.0053)]), 'test_result': OrderedDict([('recall@10', 0.0487), ('mrr@10', 0.0162), ('ndcg@10', 0.0237), ('hit@10', 0.0487), ('precision@10', 0.0049)])}
  ```
