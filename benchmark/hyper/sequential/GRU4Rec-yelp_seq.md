# Sequential Recommendation

- **Dataset**: [Yelp](../../md/yelp_seq.md)

- **Model**: [GRU4Rec](https://recbole.io/docs/user_guide/model/sequential/gru4rec.html)

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [0.005,0.001,0.0005,0.0001]
  num_layers choice [1,2]
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 0.005
  num_layers: 1
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  learning_rate:0.0001, num_layers:1
  Valid result:
  recall@10 : 0.0649    mrr@10 : 0.0225    ndcg@10 : 0.0323    hit@10 : 0.0649    precision@10 : 0.0065
  Test result:
  recall@10 : 0.0602    mrr@10 : 0.0212    ndcg@10 : 0.0302    hit@10 : 0.0602    precision@10 : 0.006
  
  learning_rate:0.001, num_layers:1
  Valid result:
  recall@10 : 0.0648    mrr@10 : 0.0227   ndcg@10 : 0.0324    hit@10 : 0.0648    precision@10 : 0.0065
  Test result:
  recall@10 : 0.0607    mrr@10 : 0.0214    ndcg@10 : 0.0305    hit@10 : 0.0607    precision@10 : 0.0061
  
  learning_rate:0.0005, num_layers:1
  Valid result:
  recall@10 : 0.0651    mrr@10 : 0.0221    ndcg@10 : 0.032    hit@10 : 0.0651    precision@10 : 0.0065
  Test result:
  recall@10 : 0.0601    mrr@10 : 0.0212    ndcg@10 : 0.0302    hit@10 : 0.0601    precision@10 : 0.006
  
  learning_rate:0.005, num_layers:2
  Valid result:
  recall@10 : 0.0611    mrr@10 : 0.0209    ndcg@10 : 0.0302    hit@10 : 0.0611    precision@10 : 0.0061
  Test result:
  recall@10 : 0.0565    mrr@10 : 0.0195    ndcg@10 : 0.028    hit@10 : 0.0565    precision@10 : 0.0056
  
  learning_rate:0.005, num_layers:1
  Valid result:
  recall@10 : 0.0683    mrr@10 : 0.0235    ndcg@10 : 0.0338    hit@10 : 0.0683    precision@10 : 0.0068
  Test result:
  recall@10 : 0.0659    mrr@10 : 0.0234    ndcg@10 : 0.0332    hit@10 : 0.0659    precision@10 : 0.0066
  
  learning_rate:0.0005, num_layers:2
  Valid result:
  recall@10 : 0.066    mrr@10 : 0.0231    ndcg@10 : 0.033    hit@10 : 0.066    precision@10 : 0.0066
  Test result:
  recall@10 : 0.0622    mrr@10 : 0.0217    ndcg@10 : 0.031    hit@10 : 0.0622    precision@10 : 0.0062
  
  learning_rate:0.001, num_layers:2
  Valid result:
  recall@10 : 0.0647    mrr@10 : 0.0222    ndcg@10 : 0.032    hit@10 : 0.0647    precision@10 : 0.0065
  Test result:
  recall@10 : 0.0597    mrr@10 : 0.0212    ndcg@10 : 0.03    hit@10 : 0.0597    precision@10 : 0.006
  
  learning_rate:0.0001, num_layers:2
  Valid result:
  recall@10 : 0.0644    mrr@10 : 0.0224    ndcg@10 : 0.0321    hit@10 : 0.0644    precision@10 : 0.0064
  Test result:
  recall@10 : 0.0599    mrr@10 : 0.0205    ndcg@10 : 0.0296    hit@10 : 0.0599    precision@10 : 0.006
  ```

- **Logging Result**:

  ```yaml
  best params:  {'learning_rate': 0.005, 'num_layers': 1}
  best result: 
  {'model': 'GRU4Rec', 'best_valid_result': OrderedDict([('recall@10', 0.0683), ('mrr@10', 0.0235), ('ndcg@10', 0.0338), ('hit@10', 0.0683), ('precision@10', 0.0068)]), 'test_result': OrderedDict([('recall@10', 0.0659), ('mrr@10', 0.0234), ('ndcg@10', 0.0332), ('hit@10', 0.0659), ('precision@10', 0.0066)])}
  ```
