# Sequential Recommendation

- **Dataset**: [Yelp](../../md/yelp_seq.md)

- **Model**: [SINE](https://recbole.io/docs/user_guide/model/sequential/sine.html)

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
  recall@10 : 0.06    mrr@10 : 0.0201    ndcg@10 : 0.0293    hit@10 : 0.06    precision@10 : 0.006
  Test result:
  recall@10 : 0.055    mrr@10 : 0.0188    ndcg@10 : 0.0271    hit@10 : 0.055    precision@10 : 0.0055

  learning_rate:0.001
  Valid result:
  recall@10 : 0.0583    mrr@10 : 0.0199    ndcg@10 : 0.0288    hit@10 : 0.0583    precision@10 : 0.0058
  Test result:
  recall@10 : 0.054    mrr@10 : 0.0188    ndcg@10 : 0.0269    hit@10 : 0.054    precision@10 : 0.0054

  learning_rate:0.0005
  Valid result:
  recall@10 : 0.0489    mrr@10 : 0.0167    ndcg@10 : 0.0241    hit@10 : 0.0489    precision@10 : 0.0049
  Test result:
  recall@10 : 0.0446    mrr@10 : 0.0154    ndcg@10 : 0.0221    hit@10 : 0.0446    precision@10 : 0.0045

  learning_rate:0.005
  Valid result:
  recall@10 : 0.0615    mrr@10 : 0.0206    ndcg@10 : 0.03    hit@10 : 0.0615    precision@10 : 0.0061
  Test result:
  recall@10 : 0.0569    mrr@10 : 0.0195    ndcg@10 : 0.0281    hit@10 : 0.0569    precision@10 : 0.0057
  ```

- **Logging Result**:

  ```yaml
  best params:  {'learning_rate': 0.005}
  best result: 
  {'model': 'SINE', 'best_valid_result': OrderedDict([('recall@10', 0.0615), ('mrr@10', 0.0206), ('ndcg@10', 0.03), ('hit@10', 0.0615), ('precision@10', 0.0061)]), 'test_result': OrderedDict([('recall@10', 0.0569), ('mrr@10', 0.0195), ('ndcg@10', 0.0281), ('hit@10', 0.0569), ('precision@10', 0.0057)])}
  ```
