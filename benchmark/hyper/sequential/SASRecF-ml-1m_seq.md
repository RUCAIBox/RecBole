# Sequential Recommendation

- **Dataset**: [MovieLens-1m](../../md/ml-1m_seq.md)

- **Model**: [SASRecF](https://recbole.io/docs/user_guide/model/sequential/sasrecf.html)

- **Time cost**: 3250.15s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [0.0003, 0.0005, 0.001, 0.003, 0.005]
  ```

- **Best parameters**:

  ```
  learning_rate: 0.001
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  learning_rate:0.005
  Valid result:
  recall@10 : 0.2476    mrr@10 : 0.0947    ndcg@10 : 0.1302    hit@10 : 0.2476    precision@10 : 0.0248
  Test result:
  recall@10 : 0.2389    mrr@10 : 0.0952    ndcg@10 : 0.1286    hit@10 : 0.2389    precision@10 : 0.0239

  learning_rate:0.001
  Valid result:
  recall@10 : 0.2591    mrr@10 : 0.1    ndcg@10 : 0.1369    hit@10 : 0.2591    precision@10 : 0.0259
  Test result:
  recall@10 : 0.2518    mrr@10 : 0.101    ndcg@10 : 0.136    hit@10 : 0.2518    precision@10 : 0.0252

  learning_rate:0.003
  Valid result:
  recall@10 : 0.2569    mrr@10 : 0.0991    ndcg@10 : 0.1357    hit@10 : 0.2569    precision@10 : 0.0257
  Test result:
  recall@10 : 0.2478    mrr@10 : 0.0996    ndcg@10 : 0.1342    hit@10 : 0.2478    precision@10 : 0.0248

  learning_rate:0.0003
  Valid result:
  recall@10 : 0.2536    mrr@10 : 0.0983    ndcg@10 : 0.1344    hit@10 : 0.2536    precision@10 : 0.0254
  Test result:
  recall@10 : 0.243    mrr@10 : 0.0988    ndcg@10 : 0.1324    hit@10 : 0.243    precision@10 : 0.0243

  learning_rate:0.0005
  Valid result:
  recall@10 : 0.2576    mrr@10 : 0.0992    ndcg@10 : 0.1359    hit@10 : 0.2576    precision@10 : 0.0258
  Test result:
  recall@10 : 0.2443    mrr@10 : 0.0998    ndcg@10 : 0.1335    hit@10 : 0.2443    precision@10 : 0.0244
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 5/5 [4:30:50<00:00, 3250.15s/trial, best loss: -0.1369]
  best params:  {'learning_rate': 0.001}
  best result: 
  {'model': 'SASRecF', 'best_valid_score': 0.1369, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.2591), ('mrr@10', 0.1), ('ndcg@10', 0.1369), ('hit@10', 0.2591), ('precision@10', 0.0259)]), 'test_result': OrderedDict([('recall@10', 0.2518), ('mrr@10', 0.101), ('ndcg@10', 0.136), ('hit@10', 0.2518), ('precision@10', 0.0252)])}
  ```
