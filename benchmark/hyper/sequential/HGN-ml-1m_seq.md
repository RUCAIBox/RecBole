# Sequential Recommendation

- **Dataset**: [MovieLens-1m](../../md/ml-1m_seq.md)

- **Model**: [HGN](https://recbole.io/docs/user_guide/model/sequential/hgn.html)

- **Time cost**: 1698.05s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [0.0003, 0.0005, 0.001, 0.003, 0.005]
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 0.0003
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  learning_rate:0.005
  Valid result:
  recall@10 : 0.1228    mrr@10 : 0.0431    ndcg@10 : 0.0615    hit@10 : 0.1228    precision@10 : 0.0123
  Test result:
  recall@10 : 0.1139    mrr@10 : 0.039    ndcg@10 : 0.0562    hit@10 : 0.1139    precision@10 : 0.0114

  learning_rate:0.003
  Valid result:
  recall@10 : 0.1275    mrr@10 : 0.0442    ndcg@10 : 0.0634    hit@10 : 0.1275    precision@10 : 0.0127
  Test result:
  recall@10 : 0.1152    mrr@10 : 0.0406    ndcg@10 : 0.0578    hit@10 : 0.1152    precision@10 : 0.0115

  learning_rate:0.0003
  Valid result:
  recall@10 : 0.1291    mrr@10 : 0.0453    ndcg@10 : 0.0647    hit@10 : 0.1291    precision@10 : 0.0129
  Test result:
  recall@10 : 0.1207    mrr@10 : 0.0422    ndcg@10 : 0.0604    hit@10 : 0.1207    precision@10 : 0.0121

  learning_rate:0.001
  Valid result:
  recall@10 : 0.1283    mrr@10 : 0.0445    ndcg@10 : 0.0637    hit@10 : 0.1283    precision@10 : 0.0128
  Test result:
  recall@10 : 0.118    mrr@10 : 0.0413    ndcg@10 : 0.059    hit@10 : 0.118    precision@10 : 0.0118

  learning_rate:0.0005
  Valid result:
  recall@10 : 0.1293    mrr@10 : 0.0453    ndcg@10 : 0.0647    hit@10 : 0.1293    precision@10 : 0.0129
  Test result:
  recall@10 : 0.1215    mrr@10 : 0.0427    ndcg@10 : 0.0608    hit@10 : 0.1215    precision@10 : 0.0121
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 5/5 [2:21:30<00:00, 1698.05s/trial, best loss: -0.0647]
  best params:  {'learning_rate': 0.0003}
  best result: 
  {'model': 'HGN', 'best_valid_score': 0.0647, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.1291), ('mrr@10', 0.0453), ('ndcg@10', 0.0647), ('hit@10', 0.1291), ('precision@10', 0.0129)]), 'test_result': OrderedDict([('recall@10', 0.1207), ('mrr@10', 0.0422), ('ndcg@10', 0.0604), ('hit@10', 0.1207), ('precision@10', 0.0121)])}
  ```
