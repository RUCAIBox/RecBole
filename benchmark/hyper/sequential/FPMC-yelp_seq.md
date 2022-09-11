# Sequential Recommendation

- **Dataset**: [Yelp](../../md/yelp_seq.md)

- **Model**: [FPMC](https://recbole.io/docs/user_guide/model/sequential/fpmc.html)

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [0.01,0.005,0.001,0.0005,0.0001]
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 0.0005
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  learning_rate:0.001
  Valid result:
  recall@10 : 0.0317    mrr@10 : 0.0106    ndcg@10 : 0.0155    hit@10 : 0.0317    precision@10 : 0.0032
  Test result:
  recall@10 : 0.0292    mrr@10 : 0.01    ndcg@10 : 0.0144    hit@10 : 0.0292    precision@10 : 0.0029

  learning_rate:0.005
  Valid result:
  recall@10 : 0.0439    mrr@10 : 0.0151    ndcg@10 : 0.0217    hit@10 : 0.0439    precision@10 : 0.0044
  Test result:
  recall@10 : 0.0386    mrr@10 : 0.0135    ndcg@10 : 0.0193    hit@10 : 0.0386    precision@10 : 0.0039

  learning_rate:0.0005
  Valid result:
  recall@10 : 0.0451    mrr@10 : 0.0155    ndcg@10 : 0.0224    hit@10 : 0.0451    precision@10 : 0.0045
  Test result:
  recall@10 : 0.0392    mrr@10 : 0.0135    ndcg@10 : 0.0194    hit@10 : 0.0392    precision@10 : 0.0039

  learning_rate:0.01
  Valid result:
  recall@10 : 0.0309    mrr@10 : 0.0104    ndcg@10 : 0.0151    hit@10 : 0.0309    precision@10 : 0.0031
  Test result:
  recall@10 : 0.0261    mrr@10 : 0.0088    ndcg@10 : 0.0128    hit@10 : 0.0261    precision@10 : 0.0026
  
  learning_rate:0.0001
  Valid result:
  recall@10 : 0.0303    mrr@10 : 0.0106    ndcg@10 : 0.0152    hit@10 : 0.0303    precision@10 : 0.003
  Test result:
  recall@10 : 0.0273    mrr@10 : 0.0095    ndcg@10 : 0.0136    hit@10 : 0.0273    precision@10 : 0.0027
  ```

- **Logging Result**:

  ```yaml
  best params:  {'learning_rate': 0.0005}
  best result: 
  {'model': 'FPMC', 'best_valid_result': OrderedDict([('recall@10', 0.0451), ('mrr@10', 0.0155), ('ndcg@10', 0.0224), ('hit@10', 0.0451), ('precision@10', 0.0045)]), 'test_result': OrderedDict([('recall@10', 0.0392), ('mrr@10', 0.0135), ('ndcg@10', 0.0194), ('hit@10', 0.0392), ('precision@10', 0.0039)])}
  ```
