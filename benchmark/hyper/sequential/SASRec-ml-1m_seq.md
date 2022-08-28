# Sequential Recommendation

- **Dataset**: [MovieLens-1m](../../md/ml-1m_seq.md)

- **Model**: [SASRec](https://recbole.io/docs/user_guide/model/sequential/sasrec.html)

- **Time cost**: 9322.13s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [0.0003, 0.0005, 0.001, 0.003, 0.005]
  ```

- **Best parameters**:

  ```
  learning_rate: 0.0005
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  learning_rate:0.003
  Valid result:
  recall@10 : 0.247    mrr@10 : 0.0912    ndcg@10 : 0.1274    hit@10 : 0.247    precision@10 : 0.0247
  Test result:
  recall@10 : 0.2316    mrr@10 : 0.0883    ndcg@10 : 0.1217    hit@10 : 0.2316    precision@10 : 0.0232
  
  learning_rate:0.0003
  Valid result:
  recall@10 : 0.2503    mrr@10 : 0.089    ndcg@10 : 0.1264    hit@10 : 0.2503    precision@10 : 0.025
  Test result:
  recall@10 : 0.244    mrr@10 : 0.0897    ndcg@10 : 0.1256    hit@10 : 0.244    precision@10 : 0.0244
  
  learning_rate:0.0005
  Valid result:
  recall@10 : 0.2496    mrr@10 : 0.0916    ndcg@10 : 0.1284    hit@10 : 0.2496    precision@10 : 0.025
  Test result:
  recall@10 : 0.2468    mrr@10 : 0.0929    ndcg@10 : 0.1286    hit@10 : 0.2468    precision@10 : 0.0247
  
  learning_rate:0.001
  Valid result:
  recall@10 : 0.2465    mrr@10 : 0.0906    ndcg@10 : 0.1268    hit@10 : 0.2465    precision@10 : 0.0246
  Test result:
  recall@10 : 0.2385    mrr@10 : 0.0906    ndcg@10 : 0.125    hit@10 : 0.2385    precision@10 : 0.0239
  
  learning_rate:0.005
  Valid result:
  recall@10 : 0.2403    mrr@10 : 0.0861    ndcg@10 : 0.1219    hit@10 : 0.2403    precision@10 : 0.024
  Test result:
  recall@10 : 0.2335    mrr@10 : 0.088    ndcg@10 : 0.1219    hit@10 : 0.2335    precision@10 : 0.0234
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 5/5 [12:56:50<00:00, 9322.13s/trial, best loss: -0.1284]
  best params:  {'learning_rate': 0.0005}
  best result: 
  {'model': 'SASRec', 'best_valid_score': 0.1284, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.2496), ('mrr@10', 0.0916), ('ndcg@10', 0.1284), ('hit@10', 0.2496), ('precision@10', 0.025)]), 'test_result': OrderedDict([('recall@10', 0.2468), ('mrr@10', 0.0929), ('ndcg@10', 0.1286), ('hit@10', 0.2468), ('precision@10', 0.0247)])}
  ```
