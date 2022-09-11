# Sequential Recommendation

- **Dataset**: [Yelp](../../md/yelp_seq.md)

- **Model**: [SHAN](https://recbole.io/docs/user_guide/model/sequential/shan.html)

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [0.005,0.001,0.0005]
  short_item_length choice [1,2,4]
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 0.0005
  short_item_length: 4
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  learning_rate:0.001, short_item_length:2
  Valid result:
  recall@10 : 0.0581    mrr@10 : 0.019    ndcg@10 : 0.028    hit@10 : 0.0581    precision@10 : 0.0058
  Test result:
  recall@10 : 0.0546    mrr@10 : 0.0188    ndcg@10 : 0.0271    hit@10 : 0.0546    precision@10 : 0.0055

  learning_rate:0.0005, short_item_length:4
  Valid result:
  recall@10 : 0.0586    mrr@10 : 0.0192    ndcg@10 : 0.0283    hit@10 : 0.0586    precision@10 : 0.0059
  Test result:
  recall@10 : 0.0564    mrr@10 : 0.019    ndcg@10 : 0.0276    hit@10 : 0.0564    precision@10 : 0.0056

  learning_rate:0.0005, short_item_length:1
  Valid result:
  recall@10 : 0.0576    mrr@10 : 0.0189    ndcg@10 : 0.0278    hit@10 : 0.0576    precision@10 : 0.0058
  Test result:
  recall@10 : 0.0541    mrr@10 : 0.0188    ndcg@10 : 0.0269    hit@10 : 0.0541    precision@10 : 0.0054

  learning_rate:0.001, short_item_length:4
  Valid result:
  recall@10 : 0.0581    mrr@10 : 0.0192    ndcg@10 : 0.0281    hit@10 : 0.0581    precision@10 : 0.0058
  Test result:
  recall@10 : 0.0557    mrr@10 : 0.0188    ndcg@10 : 0.0273    hit@10 : 0.0557    precision@10 : 0.0056

  learning_rate:0.005, short_item_length:1
  Valid result:
  recall@10 : 0.0549    mrr@10 : 0.0186    ndcg@10 : 0.027    hit@10 : 0.0549    precision@10 : 0.0055
  Test result:
  recall@10 : 0.0525    mrr@10 : 0.0185    ndcg@10 : 0.0263    hit@10 : 0.0525    precision@10 : 0.0053

  learning_rate:0.005, short_item_length:4
  Valid result:
  recall@10 : 0.0571    mrr@10 : 0.0186    ndcg@10 : 0.0275    hit@10 : 0.0571    precision@10 : 0.0057
  Test result:
  recall@10 : 0.0543    mrr@10 : 0.0185    ndcg@10 : 0.0267    hit@10 : 0.0543    precision@10 : 0.0054

  learning_rate:0.0005, short_item_length:2
  Valid result:
  recall@10 : 0.058    mrr@10 : 0.0191    ndcg@10 : 0.028    hit@10 : 0.058    precision@10 : 0.0058
  Test result:
  recall@10 : 0.0551    mrr@10 : 0.0189    ndcg@10 : 0.0273    hit@10 : 0.0551    precision@10 : 0.0055

  learning_rate:0.005, short_item_length:2
  Valid result:
  recall@10 : 0.0558    mrr@10 : 0.0188    ndcg@10 : 0.0273    hit@10 : 0.0558    precision@10 : 0.0056
  Test result:
  recall@10 : 0.054    mrr@10 : 0.0187    ndcg@10 : 0.0269    hit@10 : 0.054    precision@10 : 0.0054

  learning_rate:0.001, short_item_length:1
  Valid result:
  recall@10 : 0.0573    mrr@10 : 0.0192    ndcg@10 : 0.028    hit@10 : 0.0573    precision@10 : 0.0057
  Test result:
  recall@10 : 0.054    mrr@10 : 0.0188    ndcg@10 : 0.0269    hit@10 : 0.054    precision@10 : 0.0054
  ```

- **Logging Result**:

  ```yaml
  best params:  {'learning_rate': 0.0005, 'short_item_length': 4}
  best result:
  {'model': 'SHAN', 'best_valid_score': 0.0283, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.0586), ('mrr@10', 0.0192), ('ndcg@10', 0.0283), ('hit@10', 0.0586), ('precision@10', 0.0059)]), 'test_result': OrderedDict([('recall@10', 0.0564), ('mrr@10', 0.019), ('ndcg@10', 0.0276), ('hit@10', 0.0564), ('precision@10', 0.0056)])}
```
