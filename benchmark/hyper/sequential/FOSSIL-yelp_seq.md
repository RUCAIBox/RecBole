# Sequential Recommendation

- **Dataset**: [Yelp](../../md/yelp_seq.md)

- **Model**: [FOSSIL](https://recbole.io/docs/user_guide/model/sequential/fossil.html)

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [0.01,0.001]
  reg_weight choice [0,0.0001]
  alpha choice [0.2,0.5]
  order_len choice [1,2]
  ```

- **Best parameters**:

  ```yaml
  alpha: 0.5 
  learning_rate:0.001
  order_len:2
  reg_weight:0
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  alpha:0.5, learning_rate:0.01, order_len:2, reg_weight:0.0001
  Valid result:
  recall@10 : 0.06    mrr@10 : 0.0211    ndcg@10 : 0.0301    hit@10 : 0.06    precision@10 : 0.006
  Test result:
  recall@10 : 0.0589    mrr@10 : 0.021    ndcg@10 : 0.0297    hit@10 : 0.0589    precision@10 : 0.0059

  alpha:0.2, learning_rate:0.001, order_len:1, reg_weight:0.0001
  Valid result:
  recall@10 : 0.0596    mrr@10 : 0.0215    ndcg@10 : 0.0303    hit@10 : 0.0596    precision@10 : 0.006
  Test result:
  recall@10 : 0.0598    mrr@10 : 0.0228    ndcg@10 : 0.0314    hit@10 : 0.0598    precision@10 : 0.006

  alpha:0.2, learning_rate:0.01, order_len:2, reg_weight:0.0001
  Valid result:
  recall@10 : 0.0567    mrr@10 : 0.0203    ndcg@10 : 0.0287    hit@10 : 0.0567    precision@10 : 0.0057
  Test result:
  recall@10 : 0.0558    mrr@10 : 0.0208    ndcg@10 : 0.0289    hit@10 : 0.0558    precision@10 : 0.0056

  alpha:0.5, learning_rate:0.01, order_len:1, reg_weight:0.0001
  Valid result:
  recall@10 : 0.0593    mrr@10 : 0.021    ndcg@10 : 0.0299    hit@10 : 0.0593    precision@10 : 0.0059
  Test result:
  recall@10 : 0.0593    mrr@10 : 0.0216    ndcg@10 : 0.0303    hit@10 : 0.0593    precision@10 : 0.0059

  alpha:0.5, learning_rate:0.001, order_len:1, reg_weight:0.0001
  Valid result:
  recall@10 : 0.0648    mrr@10 : 0.0226    ndcg@10 : 0.0324    hit@10 : 0.0648    precision@10 : 0.0065
  Test result:
  recall@10 : 0.0657    mrr@10 : 0.0243    ndcg@10 : 0.0339    hit@10 : 0.0657    precision@10 : 0.0066

  alpha:0.2, learning_rate:0.01, order_len:1, reg_weight:0.0001
  Valid result:
  recall@10 : 0.0559    mrr@10 : 0.0201    ndcg@10 : 0.0284    hit@10 : 0.0559    precision@10 : 0.0056
  Test result:
  recall@10 : 0.0554    mrr@10 : 0.0212    ndcg@10 : 0.0291    hit@10 : 0.0554    precision@10 : 0.0055

  alpha:0.5, learning_rate:0.001, order_len:2, reg_weight:0
  Valid result:
  recall@10 : 0.0666    mrr@10 : 0.0225    ndcg@10 : 0.0326    hit@10 : 0.0666    precision@10 : 0.0067
  Test result:
  recall@10 : 0.0654    mrr@10 : 0.0231    ndcg@10 : 0.0329    hit@10 : 0.0654    precision@10 : 0.0065

  alpha:0.5, learning_rate:0.001, order_len:2, reg_weight:0.0001
  Valid result:
  recall@10 : 0.0664    mrr@10 : 0.0225    ndcg@10 : 0.0326    hit@10 : 0.0664    precision@10 : 0.0066
  Test result:
  recall@10 : 0.0641    mrr@10 : 0.023    ndcg@10 : 0.0325    hit@10 : 0.0641    precision@10 : 0.0064

  alpha:0.2, learning_rate:0.001, order_len:1, reg_weight:0
  Valid result:
  recall@10 : 0.0599    mrr@10 : 0.0216    ndcg@10 : 0.0304    hit@10 : 0.0599    precision@10 : 0.006
  Test result:
  recall@10 : 0.0601    mrr@10 : 0.0228    ndcg@10 : 0.0314    hit@10 : 0.0601    precision@10 : 0.006

  alpha:0.2, learning_rate:0.001, order_len:2, reg_weight:0
  Valid result:
  recall@10 : 0.062    mrr@10 : 0.0218    ndcg@10 : 0.0311    hit@10 : 0.062    precision@10 : 0.0062
  Test result:
  recall@10 : 0.0629    mrr@10 : 0.023    ndcg@10 : 0.0322    hit@10 : 0.0629    precision@10 : 0.0063

  alpha:0.5, learning_rate:0.001, order_len:1, reg_weight:0
  Valid result:
  recall@10 : 0.0643    mrr@10 : 0.0226    ndcg@10 : 0.0323    hit@10 : 0.0643    precision@10 : 0.0064
  Test result:
  recall@10 : 0.0649    mrr@10 : 0.0245    ndcg@10 : 0.0338    hit@10 : 0.0649    precision@10 : 0.0065

  alpha:0.2, learning_rate:0.001, order_len:2, reg_weight:0.0001
  Valid result:
  recall@10 : 0.0616    mrr@10 : 0.0217    ndcg@10 : 0.0309    hit@10 : 0.0616    precision@10 : 0.0062
  Test result:
  recall@10 : 0.0621    mrr@10 : 0.0227    ndcg@10 : 0.0318    hit@10 : 0.0621    precision@10 : 0.0062

  alpha:0.5, learning_rate:0.01, order_len:1, reg_weight:0
  Valid result:
  recall@10 : 0.0596    mrr@10 : 0.021    ndcg@10 : 0.0299    hit@10 : 0.0596    precision@10 : 0.006
  Test result:
  recall@10 : 0.059    mrr@10 : 0.0215    ndcg@10 : 0.0302    hit@10 : 0.059    precision@10 : 0.0059

  alpha:0.2, learning_rate:0.01, order_len:1, reg_weight:0
  Valid result:
  recall@10 : 0.056    mrr@10 : 0.0202    ndcg@10 : 0.0284    hit@10 : 0.056    precision@10 : 0.0056
  Test result:
  recall@10 : 0.0552    mrr@10 : 0.0212    ndcg@10 : 0.0291    hit@10 : 0.0552    precision@10 : 0.0055

  alpha:0.2, learning_rate:0.01, order_len:2, reg_weight:0
  Valid result:
  recall@10 : 0.0566    mrr@10 : 0.0204    ndcg@10 : 0.0287    hit@10 : 0.0566    precision@10 : 0.0057
  Test result:
  recall@10 : 0.0573    mrr@10 : 0.021    ndcg@10 : 0.0294    hit@10 : 0.0573    precision@10 : 0.0057

  alpha:0.5, learning_rate:0.01, order_len:2, reg_weight:0
  Valid result:
  recall@10 : 0.0608    mrr@10 : 0.0211    ndcg@10 : 0.0303    hit@10 : 0.0608    precision@10 : 0.0061
  Test result:
  recall@10 : 0.0588    mrr@10 : 0.0213    ndcg@10 : 0.03    hit@10 : 0.0588    precision@10 : 0.0059
  ```

- **Logging Result**:

  ```yaml
  best params:  {'alpha': 0.5, 'learning_rate': 0.001, 'order_len': 2, 'reg_weight': 0}
  best result:
  {'model': 'FOSSIL', 'best_valid_score': 0.0326, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.0666), ('mrr@10', 0.0225), ('ndcg@10', 0.0326), ('hit@10', 0.0666), ('precision@10', 0.0067)]), 'test_result': OrderedDict([('recall@10', 0.0654), ('mrr@10', 0.0231), ('ndcg@10', 0.0329), ('hit@10', 0.0654), ('precision@10', 0.0065)])}
```
