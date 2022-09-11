# Sequential Recommendation

- **Dataset**: [Yelp](../../md/yelp_seq.md)

- **Model**: [LightSANs](https://recbole.io/docs/user_guide/model/sequential/lightsans.html)

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [0.005,0.001,0.0005,0.0001]
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 0.0005
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  learning_rate:0.0005
  Valid result:
  recall@10 : 0.0774    mrr@10 : 0.0274    ndcg@10 : 0.0389    hit@10 : 0.0774    precision@10 : 0.0077
  Test result:
  recall@10 : 0.0746    mrr@10 : 0.0273    ndcg@10 : 0.0383    hit@10 : 0.0746    precision@10 : 0.0075

  learning_rate:0.001
  Valid result:
  recall@10 : 0.0774    mrr@10 : 0.0271    ndcg@10 : 0.0387    hit@10 : 0.0774    precision@10 : 0.0077
  Test result:
  recall@10 : 0.0731    mrr@10 : 0.0275    ndcg@10 : 0.0381    hit@10 : 0.0731    precision@10 : 0.0073

  learning_rate:0.005
  Valid result:
  recall@10 : 0.0514    mrr@10 : 0.0269    ndcg@10 : 0.0382    hit@10 : 0.0761    precision@10 : 0.0076
  Test result:
  recall@10 : 0.0474    mrr@10 : 0.0272    ndcg@10 : 0.0379    hit@10 : 0.0736    precision@10 : 0.0074

  learning_rate:0.0001
  Valid result:
  recall@10 : 0.0767    mrr@10 : 0.0274    ndcg@10 : 0.0388    hit@10 : 0.0767    precision@10 : 0.0077
  Test result:
  recall@10 : 0.0743    mrr@10 : 0.0273    ndcg@10 : 0.0381    hit@10 : 0.0743    precision@10 : 0.0074
  ```

- **Logging Result**:

  ```yaml
  best params:  {'learning_rate': 0.0005}
  best result: 
  {'model': 'LightSANs', 'best_valid_result': OrderedDict([('recall@10', 0.0774), ('mrr@10', 0.0274), ('ndcg@10', 0.0389), ('hit@10', 0.0774), ('precision@10', 0.0077)]), 'test_result': OrderedDict([('recall@10', 0.0746), ('mrr@10', 0.0273), ('ndcg@10', 0.0383), ('hit@10', 0.0746), ('precision@10', 0.0075)])}
  ```
