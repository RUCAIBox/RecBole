# General Recommendation

- **Dataset**: [Amazon-Books](../../md/amazon-books_general.md)

- **Model**: [BPR](https://recbole.io/docs/user_guide/model/general/bpr.html)

- **Time cost**: 1158.66s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [5e-5,1e-4,5e-4,7e-4,1e-3,5e-3,7e-3]
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 0.0001
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  learning_rate:0.0001
  Valid result:
  recall@10 : 0.1519    mrr@10 : 0.1068    ndcg@10 : 0.0978    hit@10 : 0.2464    precision@10 : 0.03
  Test result:
  recall@10 : 0.156     mrr@10 : 0.1147    ndcg@10 : 0.1043    hit@10 : 0.2501    precision@10 : 0.0314

  learning_rate:0.0005
  Valid result:
  recall@10 : 0.1592    mrr@10 : 0.1106    ndcg@10 : 0.1027    hit@10 : 0.2535    precision@10 : 0.0307
  Test result:
  recall@10 : 0.1634    mrr@10 : 0.1185    ndcg@10 : 0.1084    hit@10 : 0.2584    precision@10 : 0.0321

  learning_rate:0.005
  Valid result:
  recall@10 : 0.1098    mrr@10 : 0.0753    ndcg@10 : 0.0692    hit@10 : 0.1815    precision@10 : 0.0212
  Test result:
  recall@10 : 0.1108    mrr@10 : 0.0781    ndcg@10 : 0.0714    hit@10 : 0.1831    precision@10 : 0.0217
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 7/7 [2:56:52<00:00, 1158.66s/trial, best loss: -0.1027]
  best params:  {'learning_rate': 0.0005}
  best result: 
  {'model': 'BPR', 'best_valid_score': 0.1027, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.1592), ('mrr@10', 0.1106), ('ndcg@10', 0.1027), ('hit@10', 0.2535), ('precision@10', 0.0307)]), 'test_result': OrderedDict([('recall@10', 0.1634), ('mrr@10', 0.1185), ('ndcg@10', 0.1084), ('hit@10', 0.2584), ('precision@10', 0.0321)])}
  ```
