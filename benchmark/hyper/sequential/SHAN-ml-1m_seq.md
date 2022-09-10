# Sequential Recommendation

- **Dataset**: [MovieLens-1m](../../md/ml-1m_seq.md)

- **Model**: [SHAN](https://recbole.io/docs/user_guide/model/sequential/shan.html)

- **Time cost**: 1364.87s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [0.0005, 0.001, 0.003]
  short_item_length choice [1, 2, 3]
  ```

- **Best parameters**:

  ```
  learning_rate: 0.001
  short_item_length: 2
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  learning_rate:0.001, short_item_length:2
  Valid result:
  recall@10 : 0.123    mrr@10 : 0.032    ndcg@10 : 0.0528    hit@10 : 0.123    precision@10 : 0.0123
  Test result:
  recall@10 : 0.1162    mrr@10 : 0.0319    ndcg@10 : 0.0512    hit@10 : 0.1162    precision@10 : 0.0116

  learning_rate:0.0005, short_item_length:2
  Valid result:
  recall@10 : 0.121    mrr@10 : 0.0321    ndcg@10 : 0.0524    hit@10 : 0.121    precision@10 : 0.0121
  Test result:
  recall@10 : 0.1179    mrr@10 : 0.0323    ndcg@10 : 0.052    hit@10 : 0.1179    precision@10 : 0.0118

  learning_rate:0.001, short_item_length:3
  Valid result:
  recall@10 : 0.1145    mrr@10 : 0.0303    ndcg@10 : 0.0495    hit@10 : 0.1145    precision@10 : 0.0115
  Test result:
  recall@10 : 0.1182    mrr@10 : 0.0314    ndcg@10 : 0.0513    hit@10 : 0.1182    precision@10 : 0.0118

  learning_rate:0.001, short_item_length:1
  Valid result:
  recall@10 : 0.1112    mrr@10 : 0.0317    ndcg@10 : 0.05    hit@10 : 0.1112    precision@10 : 0.0111
  Test result:
  recall@10 : 0.1122    mrr@10 : 0.0312    ndcg@10 : 0.0499    hit@10 : 0.1122    precision@10 : 0.0112

  learning_rate:0.0005, short_item_length:1
  Valid result:
  recall@10 : 0.1092    mrr@10 : 0.0305    ndcg@10 : 0.0486    hit@10 : 0.1092    precision@10 : 0.0109
  Test result:
  recall@10 : 0.1062    mrr@10 : 0.0302    ndcg@10 : 0.0477    hit@10 : 0.1062    precision@10 : 0.0106

  learning_rate:0.003, short_item_length:2
  Valid result:
  recall@10 : 0.1208    mrr@10 : 0.032    ndcg@10 : 0.0523    hit@10 : 0.1208    precision@10 : 0.0121
  Test result:
  recall@10 : 0.1207    mrr@10 : 0.0318    ndcg@10 : 0.0522    hit@10 : 0.1207    precision@10 : 0.0121

  learning_rate:0.0005, short_item_length:3
  Valid result:
  recall@10 : 0.1121    mrr@10 : 0.0313    ndcg@10 : 0.0498    hit@10 : 0.1121    precision@10 : 0.0112
  Test result:
  recall@10 : 0.1137    mrr@10 : 0.0313    ndcg@10 : 0.0502    hit@10 : 0.1137    precision@10 : 0.0114

  learning_rate:0.003, short_item_length:3
  Valid result:
  recall@10 : 0.116    mrr@10 : 0.0303    ndcg@10 : 0.0498    hit@10 : 0.116    precision@10 : 0.0116
  Test result:
  recall@10 : 0.1192    mrr@10 : 0.0312    ndcg@10 : 0.0514    hit@10 : 0.1192    precision@10 : 0.0119

  learning_rate:0.003, short_item_length:1
  Valid result:
  recall@10 : 0.1179    mrr@10 : 0.0319    ndcg@10 : 0.0516    hit@10 : 0.1179    precision@10 : 0.0118
  Test result:
  recall@10 : 0.1147    mrr@10 : 0.0323    ndcg@10 : 0.0513    hit@10 : 0.1147    precision@10 : 0.0115
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 9/9 [3:24:43<00:00, 1364.87s/trial, best loss: -0.0528]
  best params:  {'learning_rate': 0.001, 'short_item_length': 2}
  best result: 
  {'model': 'SHAN', 'best_valid_score': 0.0528, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.123), ('mrr@10', 0.032), ('ndcg@10', 0.0528), ('hit@10', 0.123), ('precision@10', 0.0123)]), 'test_result': OrderedDict([('recall@10', 0.1162), ('mrr@10', 0.0319), ('ndcg@10', 0.0512), ('hit@10', 0.1162), ('precision@10', 0.0116)])}
  ```
