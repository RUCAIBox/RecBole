# Sequential Recommendation

- **Dataset**: [Yelp](../../md/yelp_seq.md)

- **Model**: [FDSA](https://recbole.io/docs/user_guide/model/sequential/fdsa.html)

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
  learning_rate:0.0001
  Valid result:
  recall@10 : 0.0663    mrr@10 : 0.0228    ndcg@10 : 0.0328    hit@10 : 0.0663    precision@10 : 0.0066
  Test result:
  recall@10 : 0.0609    mrr@10 : 0.021    ndcg@10 : 0.0302    hit@10 : 0.0609    precision@10 : 0.0061

  learning_rate:0.001
  Valid result:
  recall@10 : 0.0632    mrr@10 : 0.0216    ndcg@10 : 0.0311    hit@10 : 0.0632    precision@10 : 0.0063
  Test result:
  recall@10 : 0.0585    mrr@10 : 0.0201    ndcg@10 : 0.029    hit@10 : 0.0585    precision@10 : 0.0059

  learning_rate:0.005
  Valid result:
  recall@10 : 0.0671    mrr@10 : 0.0233    ndcg@10 : 0.0334    hit@10 : 0.0671    precision@10 : 0.0067
  Test result:
  recall@10 : 0.0618    mrr@10 : 0.0215    ndcg@10 : 0.0308    hit@10 : 0.0618    precision@10 : 0.0062

  learning_rate:0.0005
  Valid result:
  recall@10 : 0.0669    mrr@10 : 0.023    ndcg@10 : 0.0331    hit@10 : 0.0669    precision@10 : 0.0067
  Test result:
  recall@10 : 0.0623    mrr@10 : 0.0211    ndcg@10 : 0.0306    hit@10 : 0.0623    precision@10 : 0.0062
  ```

- **Logging Result**:

  ```yaml
  best params:  {'learning_rate': 0.0005}
  best result: 
  {'model': 'FDSA', 'best_valid_result': OrderedDict([('recall@10', 0.0669), ('mrr@10', 0.023), ('ndcg@10', 0.0331), ('hit@10', 0.0669), ('precision@10', 0.0067)]), 'test_result': OrderedDict([('recall@10', 0.0623), ('mrr@10', 0.0211), ('ndcg@10', 0.0306), ('hit@10', 0.0623), ('precision@10', 0.0062)])}
  ```
