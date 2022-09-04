# Sequential Recommendation

- **Dataset**: [MovieLens-1m](../../md/ml-1m_seq.md)

- **Model**: [HRM](https://recbole.io/docs/user_guide/model/sequential/hrm.html)

- **Time cost**: 1218.51s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [0.0005, 0.001, 0.003]
  high_order choice [1, 2, 3]
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 0.0005
  high_order: 3
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  high_order:3, learning_rate:0.001
  Valid result:
  recall@10 : 0.1679    mrr@10 : 0.0448    ndcg@10 : 0.0731    hit@10 : 0.1679    precision@10 : 0.0168
  Test result:
  recall@10 : 0.1561    mrr@10 : 0.0433    ndcg@10 : 0.0693    hit@10 : 0.1561    precision@10 : 0.0156

  high_order:3, learning_rate:0.0005
  Valid result:
  recall@10 : 0.1704    mrr@10 : 0.0453    ndcg@10 : 0.074    hit@10 : 0.1704    precision@10 : 0.017
  Test result:
  recall@10 : 0.154    mrr@10 : 0.0433    ndcg@10 : 0.0688    hit@10 : 0.154    precision@10 : 0.0154

  high_order:2, learning_rate:0.003
  Valid result:
  recall@10 : 0.1644    mrr@10 : 0.0431    ndcg@10 : 0.0711    hit@10 : 0.1644    precision@10 : 0.0164
  Test result:
  recall@10 : 0.1523    mrr@10 : 0.0422    ndcg@10 : 0.0677    hit@10 : 0.1523    precision@10 : 0.0152

  high_order:1, learning_rate:0.003
  Valid result:
  recall@10 : 0.1493    mrr@10 : 0.0414    ndcg@10 : 0.0665    hit@10 : 0.1493    precision@10 : 0.0149
  Test result:
  recall@10 : 0.1379    mrr@10 : 0.0387    ndcg@10 : 0.0618    hit@10 : 0.1379    precision@10 : 0.0138

  high_order:2, learning_rate:0.0005
  Valid result:
  recall@10 : 0.1669    mrr@10 : 0.0449    ndcg@10 : 0.0731    hit@10 : 0.1669    precision@10 : 0.0167
  Test result:
  recall@10 : 0.1533    mrr@10 : 0.0427    ndcg@10 : 0.0682    hit@10 : 0.1533    precision@10 : 0.0153

  high_order:1, learning_rate:0.0005
  Valid result:
  recall@10 : 0.1537    mrr@10 : 0.0407    ndcg@10 : 0.0668    hit@10 : 0.1537    precision@10 : 0.0154
  Test result:
  recall@10 : 0.1414    mrr@10 : 0.0391    ndcg@10 : 0.0628    hit@10 : 0.1414    precision@10 : 0.0141

  high_order:2, learning_rate:0.001
  Valid result:
  recall@10 : 0.1676    mrr@10 : 0.0442    ndcg@10 : 0.0726    hit@10 : 0.1676    precision@10 : 0.0168
  Test result:
  recall@10 : 0.1548    mrr@10 : 0.0427    ndcg@10 : 0.0685    hit@10 : 0.1548    precision@10 : 0.0155

  high_order:1, learning_rate:0.001
  Valid result:
  recall@10 : 0.1527    mrr@10 : 0.0406    ndcg@10 : 0.0665    hit@10 : 0.1527    precision@10 : 0.0153
  Test result:
  recall@10 : 0.1429    mrr@10 : 0.0393    ndcg@10 : 0.0633    hit@10 : 0.1429    precision@10 : 0.0143

  high_order:3, learning_rate:0.003
  Valid result:
  recall@10 : 0.1658    mrr@10 : 0.0451    ndcg@10 : 0.0729    hit@10 : 0.1658    precision@10 : 0.0166
  Test result:
  recall@10 : 0.1545    mrr@10 : 0.0439    ndcg@10 : 0.0693    hit@10 : 0.1545    precision@10 : 0.0154
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 9/9 [3:02:46<00:00, 1218.51s/trial, best loss: -0.074]
  best params:  {'high_order': 3, 'learning_rate': 0.0005}
  best result: 
  {'model': 'HRM', 'best_valid_score': 0.074, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.1704), ('mrr@10', 0.0453), ('ndcg@10', 0.074), ('hit@10', 0.1704), ('precision@10', 0.017)]), 'test_result': OrderedDict([('recall@10', 0.154), ('mrr@10', 0.0433), ('ndcg@10', 0.0688), ('hit@10', 0.154), ('precision@10', 0.0154)])}
  ```
