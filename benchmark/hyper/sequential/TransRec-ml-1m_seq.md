# Sequential Recommendation

- **Dataset**: [MovieLens-1m](../../md/ml-1m_seq.md)

- **Model**: [TransRec](https://recbole.io/docs/user_guide/model/sequential/transrec.html)

- **Time cost**: 1301.35s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [0.0003, 0.0005, 0.001, 0.003, 0.005]
  ```

- **Best parameters**:

  ```
  learning_rate: 0.003
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  learning_rate:0.0005
  Valid result:
  recall@10 : 0.1248    mrr@10 : 0.0328    ndcg@10 : 0.054    hit@10 : 0.1248    precision@10 : 0.0125
  Test result:
  recall@10 : 0.1104    mrr@10 : 0.029    ndcg@10 : 0.0477    hit@10 : 0.1104    precision@10 : 0.011

  learning_rate:0.003
  Valid result:
  recall@10 : 0.1354    mrr@10 : 0.0335    ndcg@10 : 0.0571    hit@10 : 0.1354    precision@10 : 0.0135
  Test result:
  recall@10 : 0.1238    mrr@10 : 0.0294    ndcg@10 : 0.0511    hit@10 : 0.1238    precision@10 : 0.0124

  learning_rate:0.005
  Valid result:
  recall@10 : 0.1367    mrr@10 : 0.0328    ndcg@10 : 0.0569    hit@10 : 0.1367    precision@10 : 0.0137
  Test result:
  recall@10 : 0.1251    mrr@10 : 0.0288    ndcg@10 : 0.0509    hit@10 : 0.1251    precision@10 : 0.0125

  learning_rate:0.0003
  Valid result:
  recall@10 : 0.1175    mrr@10 : 0.0323    ndcg@10 : 0.0519    hit@10 : 0.1175    precision@10 : 0.0118
  Test result:
  recall@10 : 0.1091    mrr@10 : 0.029    ndcg@10 : 0.0474    hit@10 : 0.1091    precision@10 : 0.0109

  learning_rate:0.001
  Valid result:
  recall@10 : 0.1316    mrr@10 : 0.0331    ndcg@10 : 0.0559    hit@10 : 0.1316    precision@10 : 0.0132
  Test result:
  recall@10 : 0.1192    mrr@10 : 0.0299    ndcg@10 : 0.0505    hit@10 : 0.1192    precision@10 : 0.0119
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 5/5 [1:48:26<00:00, 1301.35s/trial, best loss: -0.0571]
  best params:  {'learning_rate': 0.003}
  best result: 
  {'model': 'TransRec', 'best_valid_score': 0.0571, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.1354), ('mrr@10', 0.0335), ('ndcg@10', 0.0571), ('hit@10', 0.1354), ('precision@10', 0.0135)]), 'test_result': OrderedDict([('recall@10', 0.1238), ('mrr@10', 0.0294), ('ndcg@10', 0.0511), ('hit@10', 0.1238), ('precision@10', 0.0124)])}
  ```
