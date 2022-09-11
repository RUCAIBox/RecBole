# Sequential Recommendation

- **Dataset**: [Yelp](../../md/yelp_seq.md)

- **Model**: [GCSAN](https://recbole.io/docs/user_guide/model/sequential/gcsan.html)

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [0.005,0.001,0.0005,0.0001]
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 0.005
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  learning_rate:0.0001
  Valid result:
  recall@10 : 0.0591    mrr@10 : 0.02    ndcg@10 : 0.029    hit@10 : 0.0591    precision@10 : 0.0059
  Test result:
  recall@10 : 0.0553    mrr@10 : 0.0192    ndcg@10 : 0.0275    hit@10 : 0.0553    precision@10 : 0.0055

  learning_rate:0.001
  Valid result:
  recall@10 : 0.0603    mrr@10 : 0.0211    ndcg@10 : 0.0301    hit@10 : 0.0603    precision@10 : 0.006
  Test result:
  recall@10 : 0.0581    mrr@10 : 0.0198    ndcg@10 : 0.0286    hit@10 : 0.0581    precision@10 : 0.0058

  learning_rate:0.005
  Valid result:
  recall@10 : 0.0652    mrr@10 : 0.0232    ndcg@10 : 0.0329    hit@10 : 0.0652    precision@10 : 0.0065
  Test result:
  recall@10 : 0.0624    mrr@10 : 0.0232    ndcg@10 : 0.0323    hit@10 : 0.0624    precision@10 : 0.0062

  learning_rate:0.0005
  Valid result:
  recall@10 : 0.0597    mrr@10 : 0.0207    ndcg@10 : 0.0297    hit@10 : 0.0597    precision@10 : 0.006
  Test result:
  recall@10 : 0.0558    mrr@10 : 0.0193    ndcg@10 : 0.0277    hit@10 : 0.0558    precision@10 : 0.0056
  ```

- **Logging Result**:

  ```yaml
  best params:  {'learning_rate': 0.005}
  best result: 
  {'model': 'GCSAN', 'best_valid_result': OrderedDict([('recall@10', 0.0652), ('mrr@10', 0.0232), ('ndcg@10', 0.0329), ('hit@10', 0.0652), ('precision@10', 0.0065)]), 'test_result': OrderedDict([('recall@10', 0.0624), ('mrr@10', 0.0232), ('ndcg@10', 0.0323), ('hit@10', 0.0624), ('precision@10', 0.0062)])}
  ```
