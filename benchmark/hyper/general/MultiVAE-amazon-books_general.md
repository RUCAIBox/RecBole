# General Recommendation

- **Dataset**: [Amazon-Books](../../md/amazon-books_general.md)

- **Model**: [BPR](https://recbole.io/docs/user_guide/model/general/bpr.html)

- **Time cost**: 159.82s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [5e-5,1e-4,5e-4,7e-4,1e-3,5e-3,7e-3]
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 0.001
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  learning_rate:0.001
  Valid result:
  recall@10 : 0.1725    mrr@10 : 0.126    ndcg@10 : 0.1157    hit@10 : 0.273    precision@10 : 0.0338
  Test result:
  recall@10 : 0.1751    mrr@10 : 0.1355    ndcg@10 : 0.1219    hit@10 : 0.2755    precision@10 : 0.035

  learning_rate:0.005
  Valid result:
  recall@10 : 0.1709    mrr@10 : 0.1229    ndcg@10 : 0.1134    hit@10 : 0.2692    precision@10 : 0.0328
  Test result:
  recall@10 : 0.1716    mrr@10 : 0.1318    ndcg@10 : 0.119    hit@10 : 0.2701    precision@10 : 0.0341

  learning_rate:0.0007
  Valid result:
  recall@10 : 0.1707    mrr@10 : 0.1246    ndcg@10 : 0.1143    hit@10 : 0.2692    precision@10 : 0.0333
  Test result:
  recall@10 : 0.1748    mrr@10 : 0.1354    ndcg@10 : 0.1217    hit@10 : 0.274    precision@10 : 0.0347

  learning_rate:0.007
  Valid result:
  recall@10 : 0.1674    mrr@10 : 0.121    ndcg@10 : 0.111    hit@10 : 0.2659    precision@10 : 0.0324
  Test result:
  recall@10 : 0.1709    mrr@10 : 0.1306    ndcg@10 : 0.1177    hit@10 : 0.2701    precision@10 : 0.0337

  learning_rate:5e-05
  Valid result:
  recall@10 : 0.1542    mrr@10 : 0.1129    ndcg@10 : 0.1024    hit@10 : 0.2465    precision@10 : 0.0303
  Test result:
  recall@10 : 0.1552    mrr@10 : 0.1217    ndcg@10 : 0.1079    hit@10 : 0.2465    precision@10 : 0.0317

  learning_rate:0.0005
  Valid result:
  recall@10 : 0.1686    mrr@10 : 0.1234    ndcg@10 : 0.1129    hit@10 : 0.2669    precision@10 : 0.0328
  Test result:
  recall@10 : 0.1724    mrr@10 : 0.1345    ndcg@10 : 0.1207    hit@10 : 0.2706    precision@10 : 0.0342

  learning_rate:0.0001
  Valid result:
  recall@10 : 0.1668    mrr@10 : 0.122    ndcg@10 : 0.1114    hit@10 : 0.2639    precision@10 : 0.0325
  Test result:
  recall@10 : 0.1706    mrr@10 : 0.1316    ndcg@10 : 0.1187    hit@10 : 0.2676    precision@10 : 0.0337
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 7/7 [3:35:52<00:00, 1850.38s/trial, best loss: -0.1157]
  best params:  {'learning_rate': 0.001}
  best result: 
  {'model': 'MultiVAE', 'best_valid_score': 0.1157, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.1725), ('mrr@10', 0.126), ('ndcg@10', 0.1157), ('hit@10', 0.273), ('precision@10', 0.0338)]), 'test_result': OrderedDict([('recall@10', 0.1751), ('mrr@10', 0.1355), ('ndcg@10', 0.1219), ('hit@10', 0.2755), ('precision@10', 0.035)])}
  ```
