# Sequential Recommendation

- **Dataset**: [Yelp](../../md/yelp_seq.md)

- **Model**: [HRM](https://recbole.io/docs/user_guide/model/sequential/hrm.html)

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [0.001,0.0005,0.0001]
  high_order choice [1,2,4]
  ```

- **Best parameters**:

  ```yaml
  high_order:1
  learning_rate:0.0001
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
    high_order:4, learning_rate:0.0001
    Valid result:
    recall@10 : 0.0579    mrr@10 : 0.0201    ndcg@10 : 0.0288    hit@10 : 0.0579    precision@10 : 0.0058
    Test result:
    recall@10 : 0.0554    mrr@10 : 0.0193    ndcg@10 : 0.0277    hit@10 : 0.0554    precision@10 : 0.0055

    high_order:4, learning_rate:0.001
    Valid result:
    recall@10 : 0.0585    mrr@10 : 0.0196    ndcg@10 : 0.0286    hit@10 : 0.0585    precision@10 : 0.0058
    Test result:
    recall@10 : 0.0556    mrr@10 : 0.0191    ndcg@10 : 0.0275    hit@10 : 0.0556    precision@10 : 0.0056

    high_order:4, learning_rate:0.0005
    Valid result:
    recall@10 : 0.0577    mrr@10 : 0.0198    ndcg@10 : 0.0286    hit@10 : 0.0577    precision@10 : 0.0058
    Test result:
    recall@10 : 0.0548    mrr@10 : 0.0191    ndcg@10 : 0.0273    hit@10 : 0.0548    precision@10 : 0.0055

    high_order:2, learning_rate:0.0001
    Valid result:
    recall@10 : 0.0595    mrr@10 : 0.0205    ndcg@10 : 0.0295    hit@10 : 0.0595    precision@10 : 0.0059
    Test result:
    recall@10 : 0.0577    mrr@10 : 0.0202    ndcg@10 : 0.0289    hit@10 : 0.0577    precision@10 : 0.0058

    high_order:2, learning_rate:0.0005
    Valid result:
    recall@10 : 0.0594    mrr@10 : 0.0204    ndcg@10 : 0.0294    hit@10 : 0.0594    precision@10 : 0.0059
    Test result:
    recall@10 : 0.0565    mrr@10 : 0.0204    ndcg@10 : 0.0287    hit@10 : 0.0565    precision@10 : 0.0057

    high_order:1, learning_rate:0.001
    Valid result:
    recall@10 : 0.0613    mrr@10 : 0.0212    ndcg@10 : 0.0305    hit@10 : 0.0613    precision@10 : 0.0061
    Test result:
    recall@10 : 0.0557    mrr@10 : 0.0198    ndcg@10 : 0.028    hit@10 : 0.0557    precision@10 : 0.0056

    high_order:2, learning_rate:0.001
    Valid result:
    recall@10 : 0.0594    mrr@10 : 0.0206    ndcg@10 : 0.0296    hit@10 : 0.0594    precision@10 : 0.0059
    Test result:
    recall@10 : 0.057    mrr@10 : 0.0195    ndcg@10 : 0.0281    hit@10 : 0.057    precision@10 : 0.0057

    high_order:1, learning_rate:0.0005
    Valid result:
    recall@10 : 0.0613    mrr@10 : 0.0214    ndcg@10 : 0.0306    hit@10 : 0.0613    precision@10 : 0.0061
    Test result:
    recall@10 : 0.0578    mrr@10 : 0.0202    ndcg@10 : 0.0289    hit@10 : 0.0578    precision@10 : 0.0058

    high_order:1, learning_rate:0.0001
    Valid result:
    recall@10 : 0.0614    mrr@10 : 0.0215    ndcg@10 : 0.0307    hit@10 : 0.0614    precision@10 : 0.0061
    Test result:
    recall@10 : 0.0565    mrr@10 : 0.02    ndcg@10 : 0.0284    hit@10 : 0.0565    precision@10 : 0.0057

  ```

- **Logging Result**:

  ```yaml
  best params:  {'high_order': 1, 'learning_rate': 0.0001}
  best result: 
  {'model': 'HRM', 'best_valid_score': 0.0307, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.0614), ('mrr@10', 0.0215), ('ndcg@10', 0.0307), ('hit@10', 0.0614), ('precision@10', 0.0061)]), 'test_result': OrderedDict([('recall@10', 0.0565), ('mrr@10', 0.02), ('ndcg@10', 0.0284), ('hit@10', 0.0565), ('precision@10', 0.0057)])}
  ```
