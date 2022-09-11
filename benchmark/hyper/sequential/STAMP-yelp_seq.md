# Sequential Recommendation

- **Dataset**: [Yelp](../../md/yelp_seq.md)

- **Model**: [STAMP](https://recbole.io/docs/user_guide/model/sequential/stamp.html)

- **Time cost**: 3228.50s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [0.005,0.001,0.0005,0.0001]
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 0.0005
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  learning_rate:0.0001
  Valid result:
  recall@10 : 0.0515    mrr@10 : 0.0178   ndcg@10 : 0.0256    hit@10 : 0.0515    precision@10 : 0.0051
  Test result:
  recall@10 : 0.0468    mrr@10 : 0.0165    ndcg@10 : 0.0235    hit@10 : 0.0468    precision@10 : 0.0047
  
  learning_rate:0.001
  Valid result:
  recall@10 : 0.0474    mrr@10 : 0.0174    ndcg@10 : 0.0244    hit@10 : 0.0474    precision@10 : 0.0047
  Test result:
  recall@10 : 0.045    mrr@10 : 0.0172    ndcg@10 : 0.0237    hit@10 : 0.045    precision@10 : 0.0045
  
  learning_rate:0.005
  Valid result:
  recall@10 : 0.0479    mrr@10 : 0.0166    ndcg@10 : 0.0238    hit@10 : 0.0479    precision@10 : 0.0048
  Test result:
  recall@10 : 0.0455    mrr@10 : 0.0161    ndcg@10 : 0.0229    hit@10 : 0.0455    precision@10 : 0.0045
  
  learning_rate:0.0005
  Valid result:
  recall@10 : 0.0528    mrr@10 : 0.0181    ndcg@10 : 0.0261    hit@10 : 0.0528    precision@10 : 0.0053
  Test result:
  recall@10 : 0.0487    mrr@10 : 0.017    ndcg@10 : 0.0243    hit@10 : 0.0487    precision@10 : 0.0049
  ```

- **Logging Result**:

  ```yaml
  best params:  {'learning_rate': 0.0005}
  best result: 
  {'model': 'STAMP', 'best_valid_result': OrderedDict([('recall@10', 0.0528), ('mrr@10', 0.0181), ('ndcg@10', 0.0261), ('hit@10', 0.0528), ('precision@10', 0.0053)]), 'test_result': OrderedDict([('recall@10', 0.0487), ('mrr@10', 0.017), ('ndcg@10', 0.0243), ('hit@10', 0.0487), ('precision@10', 0.0049)])}
  ```
