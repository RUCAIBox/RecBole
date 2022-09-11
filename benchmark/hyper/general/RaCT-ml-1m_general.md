# General Recommendation

- **Dataset**: [MovieLens-1m](../../md/ml-1m_general.md)

- **Model**: [RaCT](https://recbole.io/docs/user_guide/model/general/slimelastic.html)

- **Hyper-parameter searching** :

  ```yaml
  dropout_prob choice [0.1,0.3,0.5]                                   
  anneal_cap choice [0.2,0.5]
  ```

- **Best parameters**:

  ```yaml
  dropout_prob: 0.5  
  anneal_cap: 0.2
  ```

- **Hyper-parameter logging** :

  ```yaml
  dropout_prob:0.5, anneal_cap:0.2
  Test result:
  recall@10 : 0.1893    mrr@10 : 0.4054    ndcg@10 : 0.2367    hit@10 : 0.7406    precision@10 : 0.1746

  dropout_prob:0.1, anneal_cap:0.2
  Test result:
  recall@10 : 0.1883    mrr@10 : 0.3948    ndcg@10 : 0.2302    hit@10 : 0.7389    precision@10 : 0.1693

  dropout_prob:0.3, anneal_cap:0.5
  Test result:
  recall@10 : 0.1901    mrr@10 : 0.3986    ndcg@10 : 0.234     hit@10 : 0.7446    precision@10 : 0.1725
  ```

- **Logging Result**:

  ```yaml
  Test result: OrderedDict([('recall@10', 0.1893), ('mrr@10', 0.4054), ('ndcg@10', 0.2367), ('hit@10', 0.7406), ('precision@10', 0.1746)])
  ```
