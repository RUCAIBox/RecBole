# General Recommendation

- **Dataset**: [Amazon-Books](../../md/amazon-books_general.md)

- **Model**: [LINE](https://recbole.io/docs/user_guide/model/general/line.html)

- **Time cost**: 945.14s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [5e-4,0.001,0.002]
  sample_num choice [1,5]
  second_order_loss_weight choice [0.3,0.6,1]
  ```

- **Best parameters**:

  ```yaml
  sample_num: 5  
  learning_rate: 0.002  
  second_order_loss_weight: 0.3
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  learning_rate:0.002, sample_num:1, second_order_loss_weight:1.0
  Valid result:
  recall@10 : 0.0535    mrr@10 : 0.0392    ndcg@10 : 0.0349    hit@10 : 0.0936    precision@10 : 0.0108
  Test result:
  recall@10 : 0.0525    mrr@10 : 0.0403    ndcg@10 : 0.0349    hit@10 : 0.093     precision@10 : 0.0108

  learning_rate:0.001, sample_num:5, second_order_loss_weight:0.3
  Valid result:
  recall@10 : 0.0467    mrr@10 : 0.0351    ndcg@10 : 0.0306    hit@10 : 0.0831    precision@10 : 0.0095
  Test result:
  recall@10 : 0.0446    mrr@10 : 0.035     ndcg@10 : 0.0297    hit@10 : 0.0802    precision@10 : 0.0093

  learning_rate:0.002, sample_num:5, second_order_loss_weight:1.0
  Valid result:
  recall@10 : 0.1331    mrr@10 : 0.0875    ndcg@10 : 0.084     hit@10 : 0.2086    precision@10 : 0.0242
  Test result:
  recall@10 : 0.137     mrr@10 : 0.0944    ndcg@10 : 0.0891    hit@10 : 0.214     precision@10 : 0.0254
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 9/9 [2:21:46<00:00, 945.14s/trial, best loss: -0.0851]
  best params:  {'learning_rate': 0.002, 'second_order_loss_weight': 0.3}
  best result: 
  {'model': 'LINE', 'best_valid_score': 0.0851, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.1343), ('mrr@10', 0.0889), ('ndcg@10', 0.0851), ('hit@10', 0.2109), ('precision@10', 0.0246)]), 'test_result': OrderedDict([('recall@10', 0.1366), ('mrr@10', 0.0935), ('ndcg@10', 0.0886), ('hit@10', 0.2143), ('precision@10', 0.0254)])}
  ```
