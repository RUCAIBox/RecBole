# Sequential Recommendation

- **Dataset**: [Amazon-Books](../../md/amazon-books_seq.md)

- **Model**: [LightSANs](https://recbole.io/docs/user_guide/model/sequential/lightsans.html)

- **Time cost**: 15779.21s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [0.0005, 0.001, 0.003]
  k_interests choice [3, 5, 7, 9]
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 0.001
  k_interests: 7
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  k_interests:3, learning_rate:0.003
  Valid result:
  recall@10 : 0.2472    mrr@10 : 0.102    ndcg@10 : 0.1363    hit@10 : 0.2472    precision@10 : 0.0247
  Test result:
  recall@10 : 0.1998    mrr@10 : 0.0837    ndcg@10 : 0.1109    hit@10 : 0.1998    precision@10 : 0.02

  k_interests:5, learning_rate:0.001
  Valid result:
  recall@10 : 0.253    mrr@10 : 0.1049    ndcg@10 : 0.1399    hit@10 : 0.253    precision@10 : 0.0253
  Test result:
  recall@10 : 0.2068    mrr@10 : 0.0861    ndcg@10 : 0.1145    hit@10 : 0.2068    precision@10 : 0.0207

  k_interests:5, learning_rate:0.003
  Valid result:
  recall@10 : 0.2437    mrr@10 : 0.1008    ndcg@10 : 0.1346    hit@10 : 0.2437    precision@10 : 0.0244
  Test result:
  recall@10 : 0.2012    mrr@10 : 0.0825    ndcg@10 : 0.1104    hit@10 : 0.2012    precision@10 : 0.0201

  k_interests:9, learning_rate:0.0005
  Valid result:
  recall@10 : 0.2506    mrr@10 : 0.1039    ndcg@10 : 0.1386    hit@10 : 0.2506    precision@10 : 0.0251
  Test result:
  recall@10 : 0.2052    mrr@10 : 0.0845    ndcg@10 : 0.1129    hit@10 : 0.2052    precision@10 : 0.0205

  k_interests:9, learning_rate:0.003
  Valid result:
  recall@10 : 0.2501    mrr@10 : 0.1031    ndcg@10 : 0.1378    hit@10 : 0.2501    precision@10 : 0.025
  Test result:
  recall@10 : 0.204    mrr@10 : 0.0848    ndcg@10 : 0.1128    hit@10 : 0.204    precision@10 : 0.0204

  k_interests:3, learning_rate:0.001
  Valid result:
  recall@10 : 0.255    mrr@10 : 0.1065    ndcg@10 : 0.1416    hit@10 : 0.255    precision@10 : 0.0255
  Test result:
  recall@10 : 0.2075    mrr@10 : 0.0872    ndcg@10 : 0.1155    hit@10 : 0.2075    precision@10 : 0.0208

  k_interests:7, learning_rate:0.0005
  Valid result:
  recall@10 : 0.2477    mrr@10 : 0.103    ndcg@10 : 0.1371    hit@10 : 0.2477    precision@10 : 0.0248
  Test result:
  recall@10 : 0.2007    mrr@10 : 0.0824    ndcg@10 : 0.1102    hit@10 : 0.2007    precision@10 : 0.0201

  k_interests:5, learning_rate:0.0005
  Valid result:
  recall@10 : 0.2544    mrr@10 : 0.1055    ndcg@10 : 0.1407    hit@10 : 0.2544    precision@10 : 0.0254
  Test result:
  recall@10 : 0.2081    mrr@10 : 0.0855    ndcg@10 : 0.1143    hit@10 : 0.2081    precision@10 : 0.0208

  k_interests:9, learning_rate:0.001
  Valid result:
  recall@10 : 0.2536    mrr@10 : 0.1061    ndcg@10 : 0.141    hit@10 : 0.2536    precision@10 : 0.0254
  Test result:
  recall@10 : 0.2078    mrr@10 : 0.0866    ndcg@10 : 0.1151    hit@10 : 0.2078    precision@10 : 0.0208

  k_interests:7, learning_rate:0.001
  Valid result:
  recall@10 : 0.2565    mrr@10 : 0.1066    ndcg@10 : 0.142    hit@10 : 0.2565    precision@10 : 0.0257
  Test result:
  recall@10 : 0.2065    mrr@10 : 0.0866    ndcg@10 : 0.1148    hit@10 : 0.2065    precision@10 : 0.0206

  k_interests:7, learning_rate:0.003
  Valid result:
  recall@10 : 0.2447    mrr@10 : 0.1017    ndcg@10 : 0.1354    hit@10 : 0.2447    precision@10 : 0.0245
  Test result:
  recall@10 : 0.1994    mrr@10 : 0.0835    ndcg@10 : 0.1108    hit@10 : 0.1994    precision@10 : 0.0199

  k_interests:3, learning_rate:0.0005
  Valid result:
  recall@10 : 0.2538    mrr@10 : 0.1051    ndcg@10 : 0.1402    hit@10 : 0.2538    precision@10 : 0.0254
  Test result:
  recall@10 : 0.205    mrr@10 : 0.0844    ndcg@10 : 0.1127    hit@10 : 0.205    precision@10 : 0.0205
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 12/12 [52:35:50<00:00, 15779.21s/trial, best loss: -0.142]
  best params:  {'k_interests': 7, 'learning_rate': 0.001}
  best result: 
  {'model': 'LightSANs', 'best_valid_score': 0.142, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.2565), ('mrr@10', 0.1066), ('ndcg@10', 0.142), ('hit@10', 0.2565), ('precision@10', 0.0257)]), 'test_result': OrderedDict([('recall@10', 0.2065), ('mrr@10', 0.0866), ('ndcg@10', 0.1148), ('hit@10', 0.2065), ('precision@10', 0.0206)])}
  ```
