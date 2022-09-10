# Sequential Recommendation

- **Dataset**: [MovieLens-1m](../../md/ml-1m_seq.md)

- **Model**: [NextItNet](https://recbole.io/docs/user_guide/model/sequential/nextitnet.html)

- **Time cost**: 8498.55s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [0.0005, 0.001, 0.003]
  kernel_size choice [2, 3, 4]
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 0.003
  kernel_size: 3
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  kernel_size:4, learning_rate:0.0005
  Valid result:
  recall@10 : 0.219    mrr@10 : 0.0843    ndcg@10 : 0.1155    hit@10 : 0.219    precision@10 : 0.0219
  Test result:
  recall@10 : 0.2122    mrr@10 : 0.0828    ndcg@10 : 0.1128    hit@10 : 0.2122    precision@10 : 0.0212

  kernel_size:4, learning_rate:0.001
  Valid result:
  recall@10 : 0.2345    mrr@10 : 0.091    ndcg@10 : 0.1242    hit@10 : 0.2345    precision@10 : 0.0235
  Test result:
  recall@10 : 0.2142    mrr@10 : 0.0822    ndcg@10 : 0.1129    hit@10 : 0.2142    precision@10 : 0.0214

  kernel_size:2, learning_rate:0.0005
  Valid result:
  recall@10 : 0.2277    mrr@10 : 0.0869    ndcg@10 : 0.1195    hit@10 : 0.2277    precision@10 : 0.0228
  Test result:
  recall@10 : 0.2115    mrr@10 : 0.0824    ndcg@10 : 0.1124    hit@10 : 0.2115    precision@10 : 0.0212

  kernel_size:3, learning_rate:0.0005
  Valid result:
  recall@10 : 0.2263    mrr@10 : 0.0871    ndcg@10 : 0.1194    hit@10 : 0.2263    precision@10 : 0.0226
  Test result:
  recall@10 : 0.2052    mrr@10 : 0.0789    ndcg@10 : 0.1082    hit@10 : 0.2052    precision@10 : 0.0205

  kernel_size:2, learning_rate:0.001
  Valid result:
  recall@10 : 0.2349    mrr@10 : 0.0925    ndcg@10 : 0.1256    hit@10 : 0.2349    precision@10 : 0.0235
  Test result:
  recall@10 : 0.221    mrr@10 : 0.0852    ndcg@10 : 0.1168    hit@10 : 0.221    precision@10 : 0.0221

  kernel_size:3, learning_rate:0.003
  Valid result:
  recall@10 : 0.247    mrr@10 : 0.0987    ndcg@10 : 0.1331    hit@10 : 0.247    precision@10 : 0.0247
  Test result:
  recall@10 : 0.2239    mrr@10 : 0.0911    ndcg@10 : 0.122    hit@10 : 0.2239    precision@10 : 0.0224

  kernel_size:2, learning_rate:0.003
  Valid result:
  recall@10 : 0.2407    mrr@10 : 0.0942    ndcg@10 : 0.1283    hit@10 : 0.2407    precision@10 : 0.0241
  Test result:
  recall@10 : 0.2268    mrr@10 : 0.0907    ndcg@10 : 0.1222    hit@10 : 0.2268    precision@10 : 0.0227

  kernel_size:3, learning_rate:0.001
  Valid result:
  recall@10 : 0.2317    mrr@10 : 0.0925    ndcg@10 : 0.1249    hit@10 : 0.2317    precision@10 : 0.0232
  Test result:
  recall@10 : 0.215    mrr@10 : 0.0814    ndcg@10 : 0.1124    hit@10 : 0.215    precision@10 : 0.0215

  kernel_size:4, learning_rate:0.003
  Valid result:
  recall@10 : 0.2463    mrr@10 : 0.0964    ndcg@10 : 0.1312    hit@10 : 0.2463    precision@10 : 0.0246
  Test result:
  recall@10 : 0.2334    mrr@10 : 0.0908    ndcg@10 : 0.1239    hit@10 : 0.2334    precision@10 : 0.0233
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 9/9 [21:14:46<00:00, 8498.55s/trial, best loss: -0.1331]
  best params:  {'kernel_size': 3, 'learning_rate': 0.003}
  best result: 
  {'model': 'NextItNet', 'best_valid_score': 0.1331, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.247), ('mrr@10', 0.0987), ('ndcg@10', 0.1331), ('hit@10', 0.247), ('precision@10', 0.0247)]), 'test_result': OrderedDict([('recall@10', 0.2239), ('mrr@10', 0.0911), ('ndcg@10', 0.122), ('hit@10', 0.2239), ('precision@10', 0.0224)])}
  ```
