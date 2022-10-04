# Sequential Recommendation

- **Dataset**: [MovieLens-1m](../../md/ml-1m_seq.md)

- **Model**: [BERT4Rec](https://recbole.io/docs/user_guide/model/sequential/bert4rec.html)

- **Time cost**: 6057.53s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [0.0003, 0.0005, 0.001, 0.003, 0.005]
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 0.005
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  learning_rate:0.005
  Valid result:
  recall@10 : 0.0811    mrr@10 : 0.0233    ndcg@10 : 0.0365    hit@10 : 0.0811    precision@10 : 0.0081
  Test result:
  recall@10 : 0.0791    mrr@10 : 0.0259    ndcg@10 : 0.0381    hit@10 : 0.0791    precision@10 : 0.0079

  learning_rate:0.001
  Valid result:
  recall@10 : 0.0691    mrr@10 : 0.0199    ndcg@10 : 0.0312    hit@10 : 0.0691    precision@10 : 0.0069
  Test result:
  recall@10 : 0.0693    mrr@10 : 0.0202    ndcg@10 : 0.0315    hit@10 : 0.0693    precision@10 : 0.0069

  learning_rate:0.0005
  Valid result:
  recall@10 : 0.0759    mrr@10 : 0.0197    ndcg@10 : 0.0325    hit@10 : 0.0759    precision@10 : 0.0076
  Test result:
  recall@10 : 0.0704    mrr@10 : 0.0219    ndcg@10 : 0.033    hit@10 : 0.0704    precision@10 : 0.007

  learning_rate:0.003
  Valid result:
  recall@10 : 0.0789    mrr@10 : 0.0215    ndcg@10 : 0.0345    hit@10 : 0.0789    precision@10 : 0.0079
  Test result:
  recall@10 : 0.0762    mrr@10 : 0.0226    ndcg@10 : 0.0348    hit@10 : 0.0762    precision@10 : 0.0076

  learning_rate:0.0003
  Valid result:
  recall@10 : 0.0653    mrr@10 : 0.0181    ndcg@10 : 0.0289    hit@10 : 0.0653    precision@10 : 0.0065
  Test result:
  recall@10 : 0.0675    mrr@10 : 0.0195    ndcg@10 : 0.0305    hit@10 : 0.0675    precision@10 : 0.0067
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 5/5 [8:24:47<00:00, 6057.53s/trial, best loss: -0.0365]
  best params:  {'learning_rate': 0.005}
  best result: 
  {'model': 'BERT4Rec', 'best_valid_score': 0.0365, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.0811), ('mrr@10', 0.0233), ('ndcg@10', 0.0365), ('hit@10', 0.0811), ('precision@10', 0.0081)]), 'test_result': OrderedDict([('recall@10', 0.0791), ('mrr@10', 0.0259), ('ndcg@10', 0.0381), ('hit@10', 0.0791), ('precision@10', 0.0079)])}
  ```
