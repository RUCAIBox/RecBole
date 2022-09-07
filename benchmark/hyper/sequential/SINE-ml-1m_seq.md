# Sequential Recommendation

- **Dataset**: [MovieLens-1m](../../md/ml-1m_seq.md)

- **Model**: [SINE](https://recbole.io/docs/user_guide/model/sequential/sine.html)

- **Time cost**: 1026.29s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [0.0005, 0.001, 0.003]
  interest_size choice [2, 3, 4]
  tau_ratio choice [0.05, 0.07, 0.1, 0.2]
  ```

- **Best parameters**:

  ```
  learning_rate: 0.003
  interest_size: 3
  tau_ratio: 0.07
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  interest_size:4, learning_rate:0.0005, tau_ratio:0.05
  Valid result:
  recall@10 : 0.0582    mrr@10 : 0.019    ndcg@10 : 0.0279    hit@10 : 0.0582    precision@10 : 0.0058
  Test result:
  recall@10 : 0.0565    mrr@10 : 0.0169    ndcg@10 : 0.0259    hit@10 : 0.0565    precision@10 : 0.0057

  interest_size:3, learning_rate:0.003, tau_ratio:0.07
  Valid result:
  recall@10 : 0.0699    mrr@10 : 0.0217    ndcg@10 : 0.0328    hit@10 : 0.0699    precision@10 : 0.007
  Test result:
  recall@10 : 0.0632    mrr@10 : 0.0182    ndcg@10 : 0.0285    hit@10 : 0.0632    precision@10 : 0.0063

  interest_size:2, learning_rate:0.001, tau_ratio:0.2
  Valid result:
  recall@10 : 0.062    mrr@10 : 0.0184    ndcg@10 : 0.0283    hit@10 : 0.062    precision@10 : 0.0062
  Test result:
  recall@10 : 0.0612    mrr@10 : 0.0195    ndcg@10 : 0.0291    hit@10 : 0.0612    precision@10 : 0.0061

  interest_size:4, learning_rate:0.001, tau_ratio:0.2
  Valid result:
  recall@10 : 0.0602    mrr@10 : 0.0185    ndcg@10 : 0.028    hit@10 : 0.0602    precision@10 : 0.006
  Test result:
  recall@10 : 0.0618    mrr@10 : 0.0191    ndcg@10 : 0.0289    hit@10 : 0.0618    precision@10 : 0.0062

  interest_size:2, learning_rate:0.003, tau_ratio:0.1
  Valid result:
  recall@10 : 0.0627    mrr@10 : 0.0184    ndcg@10 : 0.0285    hit@10 : 0.0627    precision@10 : 0.0063
  Test result:
  recall@10 : 0.067    mrr@10 : 0.0205    ndcg@10 : 0.0311    hit@10 : 0.067    precision@10 : 0.0067

  interest_size:4, learning_rate:0.001, tau_ratio:0.07
  Valid result:
  recall@10 : 0.0695    mrr@10 : 0.0211    ndcg@10 : 0.0322    hit@10 : 0.0695    precision@10 : 0.0069
  Test result:
  recall@10 : 0.0661    mrr@10 : 0.0193    ndcg@10 : 0.03    hit@10 : 0.0661    precision@10 : 0.0066

  interest_size:4, learning_rate:0.0005, tau_ratio:0.07
  Valid result:
  recall@10 : 0.06    mrr@10 : 0.0191    ndcg@10 : 0.0285    hit@10 : 0.06    precision@10 : 0.006
  Test result:
  recall@10 : 0.0636    mrr@10 : 0.0187    ndcg@10 : 0.0289    hit@10 : 0.0636    precision@10 : 0.0064

  interest_size:4, learning_rate:0.001, tau_ratio:0.05
  Valid result:
  recall@10 : 0.0636    mrr@10 : 0.0194    ndcg@10 : 0.0296    hit@10 : 0.0636    precision@10 : 0.0064
  Test result:
  recall@10 : 0.0681    mrr@10 : 0.0198    ndcg@10 : 0.0309    hit@10 : 0.0681    precision@10 : 0.0068

  interest_size:3, learning_rate:0.003, tau_ratio:0.1
  Valid result:
  recall@10 : 0.0696    mrr@10 : 0.02    ndcg@10 : 0.0312    hit@10 : 0.0696    precision@10 : 0.007
  Test result:
  recall@10 : 0.0718    mrr@10 : 0.0211    ndcg@10 : 0.0326    hit@10 : 0.0718    precision@10 : 0.0072

  interest_size:3, learning_rate:0.0005, tau_ratio:0.07
  Valid result:
  recall@10 : 0.0646    mrr@10 : 0.0174    ndcg@10 : 0.0282    hit@10 : 0.0646    precision@10 : 0.0065
  Test result:
  recall@10 : 0.0628    mrr@10 : 0.0195    ndcg@10 : 0.0294    hit@10 : 0.0628    precision@10 : 0.0063

  interest_size:3, learning_rate:0.0005, tau_ratio:0.1
  Valid result:
  recall@10 : 0.0663    mrr@10 : 0.0199    ndcg@10 : 0.0305    hit@10 : 0.0663    precision@10 : 0.0066
  Test result:
  recall@10 : 0.061    mrr@10 : 0.0181    ndcg@10 : 0.0279    hit@10 : 0.061    precision@10 : 0.0061

  interest_size:3, learning_rate:0.001, tau_ratio:0.1
  Valid result:
  recall@10 : 0.0676    mrr@10 : 0.0193    ndcg@10 : 0.0303    hit@10 : 0.0676    precision@10 : 0.0068
  Test result:
  recall@10 : 0.065    mrr@10 : 0.0207    ndcg@10 : 0.0308    hit@10 : 0.065    precision@10 : 0.0065
  ```

- **Logging Result**:

  ```yaml
  INFO  Early stop triggered. Stopping iterations as condition is reach.
  33%|███▎      | 12/36 [3:25:15<6:50:30, 1026.29s/trial, best loss: -0.0328]
  best params:  {'interest_size': 3, 'learning_rate': 0.003, 'tau_ratio': 0.07}
  best result: 
  {'model': 'SINE', 'best_valid_score': 0.0328, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.0699), ('mrr@10', 0.0217), ('ndcg@10', 0.0328), ('hit@10', 0.0699), ('precision@10', 0.007)]), 'test_result': OrderedDict([('recall@10', 0.0632), ('mrr@10', 0.0182), ('ndcg@10', 0.0285), ('hit@10', 0.0632), ('precision@10', 0.0063)])}
  ```
