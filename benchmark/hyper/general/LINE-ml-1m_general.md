# General Recommendation

- **Dataset**: [MovieLens-1m](../../md/ml-1m_general.md)

- **Model**: [LINE](https://recbole.io/docs/user_guide/model/general/line.html)

- **Time cost**: 1080.37s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [5e-4,0.001,0.002]
  sample_num choice [1,3,5]
  second_order_loss_weight choice [0.3,0.6,1]
  ```

- **Best parameters**:

  ```yaml
  sample_num: 5  
  learning_rate: 0.001  
  second_order_loss_weight: 1.0
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  learning_rate:0.001, sample_num:5, second_order_loss_weight:1.0
  Valid result:
  recall@10 : 0.1505    mrr@10 : 0.3404    ndcg@10 : 0.1843    hit@10 : 0.6759    precision@10 : 0.1333
  Test result:
  recall@10 : 0.1615    mrr@10 : 0.3903    ndcg@10 : 0.2169    hit@10 : 0.6915    precision@10 : 0.1596

  learning_rate:0.0005, sample_num:1, second_order_loss_weight:0.6
  Valid result:
  recall@10 : 0.0535    mrr@10 : 0.2141    ndcg@10 : 0.0985    hit@10 : 0.4263    precision@10 : 0.0848
  Test result:
  recall@10 : 0.0597    mrr@10 : 0.2553    ndcg@10 : 0.1198    hit@10 : 0.4489    precision@10 : 0.1002

  learning_rate:0.0005, sample_num:3, second_order_loss_weight:0.3
  Valid result:
  recall@10 : 0.1454    mrr@10 : 0.3392    ndcg@10 : 0.1822    hit@10 : 0.6685    precision@10 : 0.1325
  Test result:
  recall@10 : 0.158     mrr@10 : 0.3908    ndcg@10 : 0.2156    hit@10 : 0.6842    precision@10 : 0.1589
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 9/9 [2:42:03<00:00, 1080.37s/trial, best loss: -0.1843]
  best params:  {'learning_rate': 0.001, 'second_order_loss_weight': 1.0}
  best result: 
  {'model': 'LINE', 'best_valid_score': 0.1843, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.1505), ('mrr@10', 0.3404), ('ndcg@10', 0.1843), ('hit@10', 0.6759), ('precision@10', 0.1333)]), 'test_result': OrderedDict([('recall@10', 0.1615), ('mrr@10', 0.3903), ('ndcg@10', 0.2169), ('hit@10', 0.6915), ('precision@10', 0.1596)])}
  ```
