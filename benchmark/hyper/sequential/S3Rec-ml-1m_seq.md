# Sequential Recommendation

- **Dataset**: [MovieLens-1m](../../md/ml-1m_seq.md)

- **Model**: [S3Rec](https://recbole.io/docs/user_guide/model/sequential/s3rec.html)

- **Time cost**: 8052.85s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [0.0003, 0.0005, 0.001, 0.003, 0.005]
  pretrain_epochs choice [100]
  ```

- **Best parameters**:

  ```
  learning_rate: 0.005
  pretrain_epochs: 100
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  learning_rate:0.0005
  Valid result:
  recall@10 : 0.2483    mrr@10 : 0.089    ndcg@10 : 0.126    hit@10 : 0.2483    precision@10 : 0.0248
  Test result:
  recall@10 : 0.2412    mrr@10 : 0.0904    ndcg@10 : 0.1254    hit@10 : 0.2412    precision@10 : 0.0241

  learning_rate:0.001
  Valid result:
  recall@10 : 0.2468    mrr@10 : 0.09    ndcg@10 : 0.1264    hit@10 : 0.2468    precision@10 : 0.0247
  Test result:
  recall@10 : 0.2478    mrr@10 : 0.0897    ndcg@10 : 0.1264    hit@10 : 0.2478    precision@10 : 0.0248

  learning_rate:0.0003
  Valid result:
  recall@10 : 0.2453    mrr@10 : 0.0864    ndcg@10 : 0.1233    hit@10 : 0.2453    precision@10 : 0.0245
  Test result:
  recall@10 : 0.2435    mrr@10 : 0.0895    ndcg@10 : 0.1253    hit@10 : 0.2435    precision@10 : 0.0243

  learning_rate:0.005
  Valid result:
  recall@10 : 0.2495    mrr@10 : 0.0912    ndcg@10 : 0.128    hit@10 : 0.2495    precision@10 : 0.0249
  Test result:
  recall@10 : 0.244    mrr@10 : 0.0919    ndcg@10 : 0.1274    hit@10 : 0.244    precision@10 : 0.0244

  learning_rate:0.003
  Valid result:
  recall@10 : 0.25    mrr@10 : 0.0899    ndcg@10 : 0.127    hit@10 : 0.25    precision@10 : 0.025
  Test result:
  recall@10 : 0.2455    mrr@10 : 0.0917    ndcg@10 : 0.1274    hit@10 : 0.2455    precision@10 : 0.0245
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 5/5 [11:11:04<00:00, 8052.85s/trial, best loss: -0.128]
  best params:  {'learning_rate': 0.005}
  best result: 
  {'model': 'S3Rec', 'best_valid_score': 0.128, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.2495), ('mrr@10', 0.0912), ('ndcg@10', 0.128), ('hit@10', 0.2495), ('precision@10', 0.0249)]), 'test_result': OrderedDict([('recall@10', 0.244), ('mrr@10', 0.0919), ('ndcg@10', 0.1274), ('hit@10', 0.244), ('precision@10', 0.0244)])}
  ```
