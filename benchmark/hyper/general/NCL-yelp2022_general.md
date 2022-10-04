# General Recommendation

- **Dataset**: [Yelp2022](../../md/yelp_general.md)

- **Model**: [NCL](https://recbole.io/docs/user_guide/model/general/ncl.html)

- **Time cost**: 33426.95s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  num_clusters choice [1000]
  proto_reg choice [1e-7,1e-8]
  ssl_reg choice [1e-6,1e-7,1e-8]
  ssl_temp choice [0.07,0.1]
  ```
  
- **Best parameters**:

  ```yaml
  num_clusters: 1000
  proto_reg: 1e-8
  ssl_reg: 1e-7
  ssl_temp: 0.07
  ```
  
- **Hyper-parameter logging**:

  ```yaml
  num_clusters:1000, proto_reg:1e-07, ssl_reg:1e-08, ssl_temp:0.07
  Valid result:
  recall@10 : 0.0954    mrr@10 : 0.0704    ndcg@10 : 0.0611    hit@10 : 0.1716    precision@10 : 0.0203
  Test result:
  recall@10 : 0.0969    mrr@10 : 0.071    ndcg@10 : 0.0619    hit@10 : 0.1735    precision@10 : 0.0207
  
  num_clusters:1000, proto_reg:1e-07, ssl_reg:1e-07, ssl_temp:0.07
  Valid result:
  recall@10 : 0.0988    mrr@10 : 0.0763    ndcg@10 : 0.0656    hit@10 : 0.1778    precision@10 : 0.0212
  Test result:
  recall@10 : 0.1011    mrr@10 : 0.0779    ndcg@10 : 0.0667    hit@10 : 0.1812    precision@10 : 0.0219
  
  num_clusters:1000, proto_reg:1e-07, ssl_reg:1e-08, ssl_temp:0.1
  Valid result:
  recall@10 : 0.0913    mrr@10 : 0.0662    ndcg@10 : 0.0574    hit@10 : 0.1656    precision@10 : 0.0195
  Test result:
  recall@10 : 0.0924    mrr@10 : 0.0673    ndcg@10 : 0.0582    hit@10 : 0.1668    precision@10 : 0.0198
  
  num_clusters:1000, proto_reg:1e-08, ssl_reg:1e-06, ssl_temp:0.07
  Valid result:
  recall@10 : 0.0976    mrr@10 : 0.0776    ndcg@10 : 0.0655    hit@10 : 0.1782    precision@10 : 0.0212
  Test result:
  recall@10 : 0.0997    mrr@10 : 0.0796    ndcg@10 : 0.0672    hit@10 : 0.1799    precision@10 : 0.0217
  
  num_clusters:1000, proto_reg:1e-08, ssl_reg:1e-08, ssl_temp:0.1
  Valid result:
  recall@10 : 0.0927    mrr@10 : 0.0683    ndcg@10 : 0.0591    hit@10 : 0.1685    precision@10 : 0.0198
  Test result:
  recall@10 : 0.0947    mrr@10 : 0.0685    ndcg@10 : 0.0596    hit@10 : 0.1698    precision@10 : 0.0202
  
  num_clusters:1000, proto_reg:1e-07, ssl_reg:1e-06, ssl_temp:0.07
  Valid result:
  recall@10 : 0.0975    mrr@10 : 0.0776    ndcg@10 : 0.0655    hit@10 : 0.178    precision@10 : 0.0213
  Test result:
  recall@10 : 0.0998    mrr@10 : 0.0796    ndcg@10 : 0.0673    hit@10 : 0.1801    precision@10 : 0.0217
  
  num_clusters:1000, proto_reg:1e-07, ssl_reg:1e-07, ssl_temp:0.1
  Valid result:
  recall@10 : 0.0959    mrr@10 : 0.0713    ndcg@10 : 0.0619    hit@10 : 0.1721    precision@10 : 0.0205
  Test result:
  recall@10 : 0.0975    mrr@10 : 0.0732    ndcg@10 : 0.0631    hit@10 : 0.174    precision@10 : 0.0208
  
  num_clusters:1000, proto_reg:1e-08, ssl_reg:1e-07, ssl_temp:0.07
  Valid result:
  recall@10 : 0.1002    mrr@10 : 0.0768    ndcg@10 : 0.0663    hit@10 : 0.1793    precision@10 : 0.0214
  Test result:
  recall@10 : 0.1017    mrr@10 : 0.0783    ndcg@10 : 0.0672    hit@10 : 0.182    precision@10 : 0.0218
  
  num_clusters:1000, proto_reg:1e-07, ssl_reg:1e-06, ssl_temp:0.1
  Valid result:
  recall@10 : 0.0931    mrr@10 : 0.0727    ndcg@10 : 0.0618    hit@10 : 0.1705    precision@10 : 0.0203
  Test result:
  recall@10 : 0.0946    mrr@10 : 0.0749    ndcg@10 : 0.0633    hit@10 : 0.1721    precision@10 : 0.0205
  
  num_clusters:1000, proto_reg:1e-08, ssl_reg:1e-08, ssl_temp:0.07
  Valid result:
  recall@10 : 0.0971    mrr@10 : 0.0717    ndcg@10 : 0.0624    hit@10 : 0.1739    precision@10 : 0.0206
  Test result:
  recall@10 : 0.0989    mrr@10 : 0.0727    ndcg@10 : 0.0634    hit@10 : 0.1759    precision@10 : 0.0211
  
  num_clusters:1000, proto_reg:1e-08, ssl_reg:1e-07, ssl_temp:0.1
  Valid result:
  recall@10 : 0.0972    mrr@10 : 0.0726    ndcg@10 : 0.0631    hit@10 : 0.1745    precision@10 : 0.0208
  Test result:
  recall@10 : 0.0991    mrr@10 : 0.0742    ndcg@10 : 0.0642    hit@10 : 0.1762    precision@10 : 0.021
  
  num_clusters:1000, proto_reg:1e-08, ssl_reg:1e-06, ssl_temp:0.1
  Valid result:
  recall@10 : 0.0928    mrr@10 : 0.0733    ndcg@10 : 0.062    hit@10 : 0.1698    precision@10 : 0.0202
  Test result:
  recall@10 : 0.0941    mrr@10 : 0.0751    ndcg@10 : 0.0634    hit@10 : 0.1714    precision@10 : 0.0205
  ```
  
- **Logging Result**:

  ```yaml
  100%|██████████| 12/12 [111:25:23<00:00, 33426.95s/trial, best loss: -0.0663]
  best params:  {'num_clusters': 1000, 'proto_reg': 1e-08, 'ssl_reg': 1e-07, 'ssl_temp': 0.07}
  best result: 
  {'model': 'NCL', 'best_valid_score': 0.0663, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.1002), ('mrr@10', 0.0768), ('ndcg@10', 0.0663), ('hit@10', 0.1793), ('precision@10', 0.0214)]), 'test_result': OrderedDict([('recall@10', 0.1017), ('mrr@10', 0.0783), ('ndcg@10', 0.0672), ('hit@10', 0.182), ('precision@10', 0.0218)])}
  ```
