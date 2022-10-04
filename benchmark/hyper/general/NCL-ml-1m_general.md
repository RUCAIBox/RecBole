# General Recommendation

- **Dataset**: [MovieLens-1m](../../md/ml-1m_general.md)

- **Model**: [NCL](https://recbole.io/docs/user_guide/model/general/ncl.html)

- **Time cost**: 5449.66s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  num_clusters in [100,1000] 
  proto_reg in [1e-6,1e-7,1e-8] 
  ssl_reg in [1e-6,1e-7] 
  ssl_temp in [0.05,0.07,0.1]
  ```

- **Best parameters**:

  ```yaml
  num_clusters: 100  
  proto_reg: 1e-8  
  ssl_reg: 1e-6  
  ssl_temp: 0.05
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  num_clusters:100, proto_reg:1e-08, ssl_reg:1e-06, ssl_temp:0.05
  Valid result:
  recall@10 : 0.1805    mrr@10 : 0.3861    ndcg@10 : 0.2195    hit@10 : 0.7373    precision@10 : 0.1607
  Test result:
  recall@10 : 0.2003    mrr@10 : 0.4587    ndcg@10 : 0.2679    hit@10 : 0.7638    precision@10 : 0.1963

  num_clusters:1000, proto_reg:1e-06, ssl_reg:1e-06, ssl_temp:0.05
  Valid result:
  recall@10 : 0.1786    mrr@10 : 0.3856    ndcg@10 : 0.2186    hit@10 : 0.7341    precision@10 : 0.1595
  Test result:
  recall@10 : 0.2009    mrr@10 : 0.4599    ndcg@10 : 0.2683    hit@10 : 0.7658    precision@10 : 0.1955

  num_clusters:100, proto_reg:1e-06, ssl_reg:1e-07, ssl_temp:0.07
  Valid result:
  recall@10 : 0.1752    mrr@10 : 0.3771    ndcg@10 : 0.2138    hit@10 : 0.7283    precision@10 : 0.1572
  Test result:
  recall@10 : 0.1932    mrr@10 : 0.4506    ndcg@10 : 0.2609    hit@10 : 0.751     precision@10 : 0.1914
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 36/36 [52:58:58<00:00, 5449.66s/trial, best loss: -0.2195]
  best params:  {'num_clusters': 100, 'proto_reg': 1e-08, 'ssl_reg': 1e-06, 'ssl_temp': 0.05}
  best result: 
  {'model': 'NCL', 'best_valid_score': 0.2195, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.1805), ('mrr@10', 0.3861), ('ndcg@10', 0.2195), ('hit@10', 0.7373), ('precision@10', 0.1607)]), 'test_result': OrderedDict([('recall@10', 0.2003), ('mrr@10', 0.4587), ('ndcg@10', 0.2679), ('hit@10', 0.7638), ('precision@10', 0.1963)])}
  ```
