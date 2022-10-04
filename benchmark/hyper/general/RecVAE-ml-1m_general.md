# General Recommendation

- **Dataset**: [MovieLens-1m](../../md/ml-1m_general.md)

- **Model**: [RecVAE](https://recbole.io/docs/user_guide/model/general/recvae.html)

- **Time cost**: 143.19s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [5e-5,1e-4,5e-4,7e-4,1e-3,5e-3,7e-3]
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 0.0007
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  learning_rate:0.0007
  Valid result:
  recall@10 : 0.191     mrr@10 : 0.3933    ndcg@10 : 0.2278    hit@10 : 0.755     precision@10 : 0.1656
  Test result:
  recall@10 : 0.2163    mrr@10 : 0.4632    ndcg@10 : 0.2795    hit@10 : 0.7877    precision@10 : 0.2033

  learning_rate:0.005
  Valid result:
  recall@10 : 0.185     mrr@10 : 0.3858    ndcg@10 : 0.2214    hit@10 : 0.7427    precision@10 : 0.1605
  Test result:
  recall@10 : 0.208     mrr@10 : 0.4477    ndcg@10 : 0.2684    hit@10 : 0.7729    precision@10 : 0.197

  learning_rate:0.001
  Valid result:
  recall@10 : 0.1855    mrr@10 : 0.3903    ndcg@10 : 0.2236    hit@10 : 0.7456    precision@10 : 0.1617
  Test result:
  recall@10 : 0.21      mrr@10 : 0.4571    ndcg@10 : 0.2728    hit@10 : 0.7771    precision@10 : 0.199
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 7/7 [16:42<00:00, 143.19s/trial, best loss: -0.2278]
  best params:  {'learning_rate': 0.0007}
  best result: 
  {'model': 'RecVAE', 'best_valid_score': 0.2278, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.191), ('mrr@10', 0.3933), ('ndcg@10', 0.2278), ('hit@10', 0.755), ('precision@10', 0.1656)]), 'test_result': OrderedDict([('recall@10', 0.2163), ('mrr@10', 0.4632), ('ndcg@10', 0.2795), ('hit@10', 0.7877), ('precision@10', 0.2033)])}
  ```
