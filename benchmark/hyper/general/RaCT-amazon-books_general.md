# General Recommendation

- **Dataset**: [Amazon-Books](../../md/amazon-books_general.md)

- **Model**: [RaCT](https://recbole.io/docs/user_guide/model/general/slimelastic.html)

- **Time cost**: 61.01s/trial

- **Hyper-parameter searching** :

  ```yaml
  dropout_prob in [0.1,0.3,0.5]                                    
  anneal_cap in [0.2,0.5]
  ```

- **Best parameters**:

  ```yaml
  dropout_prob: 0.3
  anneal_cap: 0.2
  ```

- **Hyper-parameter logging** :

  ```yaml
  anneal_cap:0.5, dropout_prob:0.5
  Valid result:
  recall@10 : 0.0938    mrr@10 : 0.0662    ndcg@10 : 0.0599    hit@10 : 0.1579    precision@10 : 0.018
  Test result:
  recall@10 : 0.0972    mrr@10 : 0.0707    ndcg@10 : 0.0632    hit@10 : 0.162    precision@10 : 0.0187

  anneal_cap:0.2, dropout_prob:0.3
  Valid result:
  recall@10 : 0.0971    mrr@10 : 0.0671    ndcg@10 : 0.0616    hit@10 : 0.1598    precision@10 : 0.0181
  Test result:
  recall@10 : 0.0956    mrr@10 : 0.0687    ndcg@10 : 0.0623    hit@10 : 0.158    precision@10 : 0.0181

  anneal_cap:0.2, dropout_prob:0.1
  Valid result:
  recall@10 : 0.0895    mrr@10 : 0.061    ndcg@10 : 0.0564    hit@10 : 0.1477    precision@10 : 0.0166
  Test result:
  recall@10 : 0.093    mrr@10 : 0.0635    ndcg@10 : 0.0591    hit@10 : 0.1523    precision@10 : 0.0173

  anneal_cap:0.2, dropout_prob:0.5
  Valid result:
  recall@10 : 0.0938    mrr@10 : 0.0662    ndcg@10 : 0.0599    hit@10 : 0.1579    precision@10 : 0.018
  Test result:
  recall@10 : 0.0972    mrr@10 : 0.0707    ndcg@10 : 0.0632    hit@10 : 0.162    precision@10 : 0.0187

  anneal_cap:0.5, dropout_prob:0.1
  Valid result:
  recall@10 : 0.0895    mrr@10 : 0.061    ndcg@10 : 0.0564    hit@10 : 0.1477    precision@10 : 0.0166
  Test result:
  recall@10 : 0.093    mrr@10 : 0.0635    ndcg@10 : 0.0591    hit@10 : 0.1523    precision@10 : 0.0173

  anneal_cap:0.5, dropout_prob:0.3
  Valid result:
  recall@10 : 0.0971    mrr@10 : 0.0671    ndcg@10 : 0.0616    hit@10 : 0.1598    precision@10 : 0.0181
  Test result:
  recall@10 : 0.0956    mrr@10 : 0.0687    ndcg@10 : 0.0623    hit@10 : 0.158    precision@10 : 0.0181
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 6/6 [06:06<00:00, 61.01s/trial, best loss: -0.0616]
  best params:  {'anneal_cap': 0.2, 'dropout_prob': 0.3}
  best result: 
  {'model': 'RaCT', 'best_valid_score': 0.0616, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.0971), ('mrr@10', 0.0671), ('ndcg@10', 0.0616), ('hit@10', 0.1598), ('precision@10', 0.0181)]), 'test_result': OrderedDict([('recall@10', 0.0956), ('mrr@10', 0.0687), ('ndcg@10', 0.0623), ('hit@10', 0.158), ('precision@10', 0.0181)])}
  ```
