# Sequential Recommendation

- **Dataset**: [MovieLens-1m](../../md/ml-1m_seq.md)

- **Model**: [LightSANs](https://recbole.io/docs/user_guide/model/sequential/lightsans.html)

- **Time cost**: 8328.24s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [0.0005, 0.001, 0.003]
  k_interests choice [3, 5, 7, 9]
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 0.001
  k_interests: 3
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  k_interests:7, learning_rate:0.003
  Valid result:
  recall@10 : 0.2523    mrr@10 : 0.0977    ndcg@10 : 0.1336    hit@10 : 0.2523    precision@10 : 0.0252
  Test result:
  recall@10 : 0.244    mrr@10 : 0.096    ndcg@10 : 0.1305    hit@10 : 0.244    precision@10 : 0.0244

  k_interests:9, learning_rate:0.0005
  Valid result:
  recall@10 : 0.2501    mrr@10 : 0.0934    ndcg@10 : 0.1298    hit@10 : 0.2501    precision@10 : 0.025
  Test result:
  recall@10 : 0.2465    mrr@10 : 0.0918    ndcg@10 : 0.1277    hit@10 : 0.2465    precision@10 : 0.0246

  k_interests:5, learning_rate:0.003
  Valid result:
  recall@10 : 0.2556    mrr@10 : 0.0978    ndcg@10 : 0.1345    hit@10 : 0.2556    precision@10 : 0.0256
  Test result:
  recall@10 : 0.241    mrr@10 : 0.0927    ndcg@10 : 0.1272    hit@10 : 0.241    precision@10 : 0.0241

  k_interests:9, learning_rate:0.001
  Valid result:
  recall@10 : 0.2574    mrr@10 : 0.0975    ndcg@10 : 0.1346    hit@10 : 0.2574    precision@10 : 0.0257
  Test result:
  recall@10 : 0.2447    mrr@10 : 0.0953    ndcg@10 : 0.13    hit@10 : 0.2447    precision@10 : 0.0245

  k_interests:3, learning_rate:0.003
  Valid result:
  recall@10 : 0.2468    mrr@10 : 0.0928    ndcg@10 : 0.1286    hit@10 : 0.2468    precision@10 : 0.0247
  Test result:
  recall@10 : 0.2372    mrr@10 : 0.0923    ndcg@10 : 0.1261    hit@10 : 0.2372    precision@10 : 0.0237

  k_interests:5, learning_rate:0.0005
  Valid result:
  recall@10 : 0.2533    mrr@10 : 0.0941    ndcg@10 : 0.1311    hit@10 : 0.2533    precision@10 : 0.0253
  Test result:
  recall@10 : 0.2445    mrr@10 : 0.0923    ndcg@10 : 0.1277    hit@10 : 0.2445    precision@10 : 0.0244

  k_interests:3, learning_rate:0.001
  Valid result:
  recall@10 : 0.2559    mrr@10 : 0.1    ndcg@10 : 0.1363    hit@10 : 0.2559    precision@10 : 0.0256
  Test result:
  recall@10 : 0.2458    mrr@10 : 0.0951    ndcg@10 : 0.1301    hit@10 : 0.2458    precision@10 : 0.0246

  k_interests:3, learning_rate:0.0005
  Valid result:
  recall@10 : 0.2587    mrr@10 : 0.096    ndcg@10 : 0.1337    hit@10 : 0.2587    precision@10 : 0.0259
  Test result:
  recall@10 : 0.2442    mrr@10 : 0.0955    ndcg@10 : 0.1301    hit@10 : 0.2442    precision@10 : 0.0244

  k_interests:5, learning_rate:0.001
  Valid result:
  recall@10 : 0.2553    mrr@10 : 0.0972    ndcg@10 : 0.134    hit@10 : 0.2553    precision@10 : 0.0255
  Test result:
  recall@10 : 0.2427    mrr@10 : 0.0916    ndcg@10 : 0.1268    hit@10 : 0.2427    precision@10 : 0.0243

  k_interests:7, learning_rate:0.0005
  Valid result:
  recall@10 : 0.2571    mrr@10 : 0.0979    ndcg@10 : 0.1348    hit@10 : 0.2571    precision@10 : 0.0257
  Test result:
  recall@10 : 0.2456    mrr@10 : 0.0955    ndcg@10 : 0.1305    hit@10 : 0.2456    precision@10 : 0.0246

  k_interests:7, learning_rate:0.001
  Valid result:
  recall@10 : 0.251    mrr@10 : 0.0955    ndcg@10 : 0.1315    hit@10 : 0.251    precision@10 : 0.0251
  Test result:
  recall@10 : 0.239    mrr@10 : 0.0925    ndcg@10 : 0.1265    hit@10 : 0.239    precision@10 : 0.0239

  k_interests:9, learning_rate:0.003
  Valid result:
  recall@10 : 0.2617    mrr@10 : 0.0985    ndcg@10 : 0.1363    hit@10 : 0.2617    precision@10 : 0.0262
  Test result:
  recall@10 : 0.246    mrr@10 : 0.0938    ndcg@10 : 0.1292    hit@10 : 0.246    precision@10 : 0.0246
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 12/12 [27:45:38<00:00, 8328.24s/trial, best loss: -0.1363] 
  best params:  {'k_interests': 3, 'learning_rate': 0.001}
  best result: 
  {'model': 'LightSANs', 'best_valid_score': 0.1363, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.2559), ('mrr@10', 0.1), ('ndcg@10', 0.1363), ('hit@10', 0.2559), ('precision@10', 0.0256)]), 'test_result': OrderedDict([('recall@10', 0.2458), ('mrr@10', 0.0951), ('ndcg@10', 0.1301), ('hit@10', 0.2458), ('precision@10', 0.0246)])}
  ```
