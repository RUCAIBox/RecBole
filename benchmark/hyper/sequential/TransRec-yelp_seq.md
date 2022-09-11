# Sequential Recommendation

- **Dataset**: [Yelp](../../md/yelp_seq.md)

- **Model**: [TransRec](https://recbole.io/docs/user_guide/model/sequential/transrec.html)

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [0.005,0.001,0.0005,0.0001]
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 0.001
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  learning_rate:0.0005
  Valid result:
  recall@10 : 0.0561    mrr@10 : 0.0207    ndcg@10 : 0.0289    hit@10 : 0.0561    precision@10 : 0.0056
  Test result:
  recall@10 : 0.0544    mrr@10 : 0.0219    ndcg@10 : 0.0294    hit@10 : 0.0544    precision@10 : 0.0054

  learning_rate:0.001
  Valid result:
  recall@10 : 0.0563    mrr@10 : 0.0205    ndcg@10 : 0.0288    hit@10 : 0.0563    precision@10 : 0.0056
  Test result:
  recall@10 : 0.0554    mrr@10 : 0.0223    ndcg@10 : 0.03    hit@10 : 0.0554    precision@10 : 0.0055

  learning_rate:0.005
  Valid result:
  recall@10 : 0.052    mrr@10 : 0.0189    ndcg@10 : 0.0265    hit@10 : 0.052    precision@10 : 0.0052
  Test result:
  recall@10 : 0.0488    mrr@10 : 0.0184    ndcg@10 : 0.0255    hit@10 : 0.0488    precision@10 : 0.0049

  learning_rate:0.0001
  Valid result:
  recall@10 : 0.0494    mrr@10 : 0.0189    ndcg@10 : 0.0259    hit@10 : 0.0494    precision@10 : 0.0049
  Test result:
  recall@10 : 0.0492    mrr@10 : 0.0207    ndcg@10 : 0.0272    hit@10 : 0.0492    precision@10 : 0.0049
  ```

- **Logging Result**:

  ```yaml
  best params:  {'learning_rate': 0.001}
  best result: 
  {'model': 'TransRec', 'best_valid_result': OrderedDict([('recall@10', 0.0563), ('mrr@10', 0.0205), ('ndcg@10', 0.0288), ('hit@10', 0.0563), ('precision@10', 0.0056)]), 'test_result': OrderedDict([('recall@10', 0.0554), ('mrr@10', 0.0223), ('ndcg@10', 0.03), ('hit@10', 0.0554), ('precision@10', 0.0055)])}
  ```
