# General Recommendation

- **Dataset**: [MovieLens-1m](../../md/ml-1m_general.md)

- **Model**: [ItemKNN](https://recbole.io/docs/user_guide/model/general/itemknn.html)

- **Time cost**: 81.61s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  k choice [10,50,100,200,250,300,400]      
  shrink choice [0.0,0.1,0.5,1,2]
  ```

- **Best parameters**:

  ```yaml
  k: 250  
  shrink: 1.2
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  k:50, shrink:0.1
  Valid result:
  recall@10 : 0.1503    mrr@10 : 0.3479    ndcg@10 : 0.19      hit@10 : 0.6822    precision@10 : 0.1417
  Test result:
  recall@10 : 0.1646    mrr@10 : 0.4068    ndcg@10 : 0.2286    hit@10 : 0.703     precision@10 : 0.1709

  k:200, shrink:0.1
  Valid result:
  recall@10 : 0.1567    mrr@10 : 0.3586    ndcg@10 : 0.1971    hit@10 : 0.6905    precision@10 : 0.1457
  Test result:
  recall@10 : 0.1725    mrr@10 : 0.4188    ndcg@10 : 0.2379    hit@10 : 0.7129    precision@10 : 0.1757

  k:250, shrink:1.2
  Valid result:
  recall@10 : 0.1566    mrr@10 : 0.3619    ndcg@10 : 0.1982    hit@10 : 0.6943    precision@10 : 0.1462
  Test result:
  recall@10 : 0.1723    mrr@10 : 0.4254    ndcg@10 : 0.2389    hit@10 : 0.7139    precision@10 : 0.1752
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 28/28 [38:05<00:00, 81.61s/trial, best loss: -0.1982]
  best params:  {'k': 250, 'shrink': 1.2}
  best result: 
  {'model': 'ItemKNN', 'best_valid_score': 0.1982, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.1566), ('mrr@10', 0.3619), ('ndcg@10', 0.1982), ('hit@10', 0.6943), ('precision@10', 0.1462)]), 'test_result': OrderedDict([('recall@10', 0.1723), ('mrr@10', 0.4254), ('ndcg@10', 0.2389), ('hit@10', 0.7139), ('precision@10', 0.1752)])}
  ```
