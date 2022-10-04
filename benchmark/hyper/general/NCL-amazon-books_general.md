# General Recommendation

- **Dataset**: [Amazon-Books](../../md/amazon-books_general.md)

- **Model**: [NCL](https://recbole.io/docs/user_guide/model/general/ncl.html)

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  num_clusters in [100,1000] 
  proto_reg in [1e-6,1e-7,1e-8] 
  ssl_reg in [1e-6,1e-7] 
  ssl_temp in [0.05,0.07,0.1]
  ```

- **Current Best parameters**:

  ```yaml
  num_clusters: 1000  
  proto_reg: 1e-8  
  ssl_reg: 1e-7  
  ssl_temp: 0.05
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  num_clusters:1000, proto_reg:1e-08, ssl_reg:1e-07, ssl_temp:0.05
  Valid result:
  recall@10 : 0.177     mrr@10 : 0.1271    ndcg@10 : 0.1165    hit@10 : 0.2817    precision@10 : 0.0349
  Test result:
  recall@10 : 0.1826    mrr@10 : 0.1365    ndcg@10 : 0.1238    hit@10 : 0.2861    precision@10 : 0.0367
  
  num_clusters:1000, proto_reg:1e-08, ssl_reg:1e-07, ssl_temp:0.1
  Valid result:
  recall@10 : 0.1634    mrr@10 : 0.1171    ndcg@10 : 0.1067    hit@10 : 0.2621    precision@10 : 0.0321
  Test result:
  recall@10 : 0.1656    mrr@10 : 0.1243    ndcg@10 : 0.1118    hit@10 : 0.2634    precision@10 : 0.0334
  ```

- **Logging Result**:

  ```yaml
  Test result: OrderedDict([('recall@10', 0.1826), ('mrr@10', 0.1365), ('ndcg@10', 0.1238), ('hit@10', 0.2861), ('precision@10', 0.0367)])}
  ```