# Sequential Recommendation

- **Dataset**: [Yelp](../../md/yelp_seq.md)

- **Model**: [NPE](https://recbole.io/docs/user_guide/model/sequential/npe.html)

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [0.001,0.0005,0.0001]
  dropout_prob choice [0.2,0.3,0.5]
  ```

- **Best parameters**:

  ```yaml
  dropout_prob:0.2
  learning_rate:0.0005
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
    dropout_prob:0.5, learning_rate:0.0001
    Valid result:
    recall@10 : 0.0464    mrr@10 : 0.0156    ndcg@10 : 0.0227    hit@10 : 0.0464    precision@10 : 0.0046
    Test result:
    recall@10 : 0.0425    mrr@10 : 0.0146    ndcg@10 : 0.021    hit@10 : 0.0425    precision@10 : 0.0043

    dropout_prob:0.2, learning_rate:0.001
    Valid result:
    recall@10 : 0.0532    mrr@10 : 0.0182    ndcg@10 : 0.0262    hit@10 : 0.0532    precision@10 : 0.0053
    Test result:
    recall@10 : 0.0475    mrr@10 : 0.0158    ndcg@10 : 0.0231    hit@10 : 0.0475    precision@10 : 0.0048

    dropout_prob:0.2, learning_rate:0.0001
    Valid result:
    recall@10 : 0.0523    mrr@10 : 0.0179    ndcg@10 : 0.0259    hit@10 : 0.0523    precision@10 : 0.0052
    Test result:
    recall@10 : 0.0481    mrr@10 : 0.0165    ndcg@10 : 0.0238    hit@10 : 0.0481    precision@10 : 0.0048

    dropout_prob:0.3, learning_rate:0.0005
    Valid result:
    recall@10 : 0.0527    mrr@10 : 0.0183    ndcg@10 : 0.0262    hit@10 : 0.0527    precision@10 : 0.0053
    Test result:
    recall@10 : 0.0476    mrr@10 : 0.0162    ndcg@10 : 0.0234    hit@10 : 0.0476    precision@10 : 0.0048

    dropout_prob:0.2, learning_rate:0.0005
    Valid result:
    recall@10 : 0.054    mrr@10 : 0.0181    ndcg@10 : 0.0263    hit@10 : 0.054    precision@10 : 0.0054
    Test result:
    recall@10 : 0.0493    mrr@10 : 0.0169    ndcg@10 : 0.0243    hit@10 : 0.0493    precision@10 : 0.0049

    dropout_prob:0.5, learning_rate:0.0005
    Valid result:
    recall@10 : 0.0465    mrr@10 : 0.0156    ndcg@10 : 0.0227    hit@10 : 0.0465    precision@10 : 0.0047
    Test result:
    recall@10 : 0.0421    mrr@10 : 0.0143    ndcg@10 : 0.0207    hit@10 : 0.0421    precision@10 : 0.0042

    dropout_prob:0.3, learning_rate:0.001
    Valid result:
    recall@10 : 0.0523    mrr@10 : 0.0178    ndcg@10 : 0.0258    hit@10 : 0.0523    precision@10 : 0.0052
    Test result:
    recall@10 : 0.0466    mrr@10 : 0.0158    ndcg@10 : 0.0229    hit@10 : 0.0466    precision@10 : 0.0047

    dropout_prob:0.3, learning_rate:0.0001
    Valid result:
    recall@10 : 0.0508    mrr@10 : 0.0174    ndcg@10 : 0.0251    hit@10 : 0.0508    precision@10 : 0.0051
    Test result:
    recall@10 : 0.0461    mrr@10 : 0.0158    ndcg@10 : 0.0227    hit@10 : 0.0461    precision@10 : 0.0046

    dropout_prob:0.5, learning_rate:0.001
    Valid result:
    recall@10 : 0.0481    mrr@10 : 0.0161    ndcg@10 : 0.0235    hit@10 : 0.0481    precision@10 : 0.0048
    Test result:
    recall@10 : 0.0423    mrr@10 : 0.0146    ndcg@10 : 0.021    hit@10 : 0.0423    precision@10 : 0.0042

  ```

- **Logging Result**:

  ```yaml
  best params:  {'dropout_prob': 0.2, 'learning_rate': 0.0005}
  best result:
  {'model': 'NPE', 'best_valid_score': 0.0263, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.054), ('mrr@10', 0.0181), ('ndcg@10', 0.0263), ('hit@10', 0.054), ('precision@10', 0.0054)]), 'test_result': OrderedDict([('recall@10', 0.0493), ('mrr@10', 0.0169), ('ndcg@10', 0.0243), ('hit@10', 0.0493), ('precision@10', 0.0049)])}

```
