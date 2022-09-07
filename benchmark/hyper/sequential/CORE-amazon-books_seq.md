# Sequential Recommendation

- **Dataset**: [Amazon-Books](../../md/amazon-books_seq.md)

- **Model**: [CORE](https://recbole.io/docs/user_guide/model/sequential/core.html)

- **Time cost**: 16473.21s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate in [0.0005, 0.001, 0.003]
  temperature in [0.05, 0.07, 0.1, 0.2]
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 0.0005
  temperature: 0.07
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  learning_rate:0.003, temperature:0.1
  Valid result:
  recall@10 : 0.1302    mrr@10 : 0.0333    ndcg@10 : 0.0556    hit@10 : 0.1302    precision@10 : 0.013
  Test result:
  recall@10 : 0.0926    mrr@10 : 0.0249    ndcg@10 : 0.0404    hit@10 : 0.0926    precision@10 : 0.0093

  learning_rate:0.0005, temperature:0.2
  Valid result:
  recall@10 : 0.0972    mrr@10 : 0.0286    ndcg@10 : 0.0445    hit@10 : 0.0972    precision@10 : 0.0097
  Test result:
  recall@10 : 0.0717    mrr@10 : 0.0219    ndcg@10 : 0.0335    hit@10 : 0.0717    precision@10 : 0.0072

  learning_rate:0.001, temperature:0.05
  Valid result:
  recall@10 : 0.17    mrr@10 : 0.042    ndcg@10 : 0.0715    hit@10 : 0.17    precision@10 : 0.017
  Test result:
  recall@10 : 0.1282    mrr@10 : 0.0325    ndcg@10 : 0.0545    hit@10 : 0.1282    precision@10 : 0.0128

  learning_rate:0.001, temperature:0.07
  Valid result:
  recall@10 : 0.1708    mrr@10 : 0.0423    ndcg@10 : 0.0719    hit@10 : 0.1708    precision@10 : 0.0171
  Test result:
  recall@10 : 0.1274    mrr@10 : 0.0325    ndcg@10 : 0.0543    hit@10 : 0.1274    precision@10 : 0.0127

  learning_rate:0.0005, temperature:0.1
  Valid result:
  recall@10 : 0.1617    mrr@10 : 0.0443    ndcg@10 : 0.0715    hit@10 : 0.1617    precision@10 : 0.0162
  Test result:
  recall@10 : 0.1216    mrr@10 : 0.0349    ndcg@10 : 0.0549    hit@10 : 0.1216    precision@10 : 0.0122

  learning_rate:0.001, temperature:0.2
  Valid result:
  recall@10 : 0.0925    mrr@10 : 0.0276    ndcg@10 : 0.0426    hit@10 : 0.0925    precision@10 : 0.0092
  Test result:
  recall@10 : 0.0683    mrr@10 : 0.0207    ndcg@10 : 0.0318    hit@10 : 0.0683    precision@10 : 0.0068

  learning_rate:0.003, temperature:0.2
  Valid result:
  recall@10 : 0.094    mrr@10 : 0.0271    ndcg@10 : 0.0426    hit@10 : 0.094    precision@10 : 0.0094
  Test result:
  recall@10 : 0.0677    mrr@10 : 0.0206    ndcg@10 : 0.0315    hit@10 : 0.0677    precision@10 : 0.0068

  learning_rate:0.0005, temperature:0.05
  Valid result:
  recall@10 : 0.1782    mrr@10 : 0.0443    ndcg@10 : 0.0752    hit@10 : 0.1782    precision@10 : 0.0178
  Test result:
  recall@10 : 0.1334    mrr@10 : 0.0347    ndcg@10 : 0.0575    hit@10 : 0.1334    precision@10 : 0.0133

  learning_rate:0.003, temperature:0.07
  Valid result:
  recall@10 : 0.1403    mrr@10 : 0.0338    ndcg@10 : 0.0581    hit@10 : 0.1403    precision@10 : 0.014
  Test result:
  recall@10 : 0.0976    mrr@10 : 0.0234    ndcg@10 : 0.0404    hit@10 : 0.0976    precision@10 : 0.0098

  learning_rate:0.001, temperature:0.1
  Valid result:
  recall@10 : 0.1495    mrr@10 : 0.0396    ndcg@10 : 0.0649    hit@10 : 0.1495    precision@10 : 0.0149
  Test result:
  recall@10 : 0.1089    mrr@10 : 0.0302    ndcg@10 : 0.0483    hit@10 : 0.1089    precision@10 : 0.0109

  learning_rate:0.003, temperature:0.05
  Valid result:
  recall@10 : 0.1359    mrr@10 : 0.0319    ndcg@10 : 0.0556    hit@10 : 0.1359    precision@10 : 0.0136
  Test result:
  recall@10 : 0.0963    mrr@10 : 0.0229    ndcg@10 : 0.0396    hit@10 : 0.0963    precision@10 : 0.0096

  learning_rate:0.0005, temperature:0.07
  Valid result:
  recall@10 : 0.1798    mrr@10 : 0.0457    ndcg@10 : 0.0767    hit@10 : 0.1798    precision@10 : 0.018
  Test result:
  recall@10 : 0.1377    mrr@10 : 0.0366    ndcg@10 : 0.06    hit@10 : 0.1377    precision@10 : 0.0138
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 12/12 [54:54:38<00:00, 16473.21s/trial, best loss: -0.0767]
  best params:  {'learning_rate': 0.0005, 'temperature': 0.07}
  best result: 
  {'model': 'CORE', 'best_valid_score': 0.0767, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.1798), ('mrr@10', 0.0457), ('ndcg@10', 0.0767), ('hit@10', 0.1798), ('precision@10', 0.018)]), 'test_result': OrderedDict([('recall@10', 0.1377), ('mrr@10', 0.0366), ('ndcg@10', 0.06), ('hit@10', 0.1377), ('precision@10', 0.0138)])}
  ```
