# Sequential Recommendation

- **Dataset**: [Yelp-feature](../../md/yelp_seq_feature.md)

- **Model**: [GRU4RecF](https://recbole.io/docs/user_guide/model/sequential/gru4recf.html)

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [0.005,0.001,0.0005]
  num_layers choice [1,2]
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 0.005
  num_layers: 1
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  learning_rate:0.0005, num_layers:1
  Valid result:
  recall@10 : 0.0645    mrr@10 : 0.0223    ndcg@10 : 0.032    hit@10 : 0.0645    precision@10 : 0.0065
  Test result:
  recall@10 : 0.0593    mrr@10 : 0.0206    ndcg@10 : 0.0295    hit@10 : 0.0593    precision@10 : 0.0059
  
  learning_rate:0.005, num_layers:2
  Valid result:
  recall@10 : 0.0635    mrr@10 : 0.0221   ndcg@10 : 0.0316    hit@10 : 0.0635    precision@10 : 0.0064
  Test result:
  recall@10 : 0.0585    mrr@10 : 0.0205    ndcg@10 : 0.0293    hit@10 : 0.0585    precision@10 : 0.0059
  
  learning_rate:0.001, num_layers:2
  Valid result:
  recall@10 : 0.0641    mrr@10 : 0.0222    ndcg@10 : 0.0318    hit@10 : 0.0641    precision@10 : 0.0064
  Test result:
  recall@10 : 0.0584    mrr@10 : 0.0202    ndcg@10 : 0.029    hit@10 : 0.0584    precision@10 : 0.0058
  
  learning_rate:0.001, num_layers:1
  Valid result:
  recall@10 : 0.064    mrr@10 : 0.022    ndcg@10 : 0.0316    hit@10 : 0.064    precision@10 : 0.0064
  Test result:
  recall@10 : 0.06    mrr@10 : 0.0212    ndcg@10 : 0.0302    hit@10 : 0.06    precision@10 : 0.006
  
  learning_rate:0.005, num_layers:1
  Valid result:
  recall@10 : 0.0692    mrr@10 : 0.0237    ndcg@10 : 0.0342    hit@10 : 0.0692    precision@10 : 0.0069
  Test result:
  recall@10 : 0.0651    mrr@10 : 0.0226    ndcg@10 : 0.0324    hit@10 : 0.0651    precision@10 : 0.0065
  
  learning_rate:0.0005, num_layers:2
  Valid result:
  recall@10 : 0.0662    mrr@10 : 0.0229    ndcg@10 : 0.0329    hit@10 : 0.0662    precision@10 : 0.0066
  Test result:
  recall@10 : 0.0626    mrr@10 : 0.0221    ndcg@10 : 0.0315    hit@10 : 0.0626    precision@10 : 0.0063
  
  ```

- **Logging Result**:

  ```yaml
  best params:  {'learning_rate': 0.005, 'num_layers': 1}
  best result: 
  {'model': 'GRU4Rec', 'best_valid_result': OrderedDict([('recall@10', 0.0692), ('mrr@10', 0.0237), ('ndcg@10', 0.0342), ('hit@10', 0.0692), ('precision@10', 0.0069)]), 'test_result': OrderedDict([('recall@10', 0.0651), ('mrr@10', 0.0226), ('ndcg@10', 0.0324), ('hit@10', 0.0651), ('precision@10', 0.0065)])}
  ```
