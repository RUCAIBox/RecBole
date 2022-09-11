# General Recommendation

- **Dataset**: [Amazon-Books](../../md/amazon-books_general.md)

- **Model**: [ItemKNN](https://recbole.io/docs/user_guide/model/general/itemknn.html)

- **Time cost**: 452.47s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  k choice [10,50,100,200,250,300,400]      
  shrink choice [0.0,0.1,0.5,1,2]
  ```

- **Best parameters**:

  ```yaml
  k: 10  
  shrink: 1.2
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  k:10, shrink:0.5
  Valid result:
  recall@10 : 0.2543    mrr@10 : 0.2046    ndcg@10 : 0.1829    hit@10 : 0.3899    precision@10 : 0.0534
  Test result:
  recall@10 : 0.2671    mrr@10 : 0.2321    ndcg@10 : 0.2017    hit@10 : 0.4028    precision@10 : 0.0579

  k:10, shrink:0.0
  Valid result:
  recall@10 : 0.2546    mrr@10 : 0.2048    ndcg@10 : 0.183     hit@10 : 0.3902    precision@10 : 0.0535
  Test result:
  recall@10 : 0.2672    mrr@10 : 0.2323    ndcg@10 : 0.2018    hit@10 : 0.4028    precision@10 : 0.0579

  k:10, shrink:1.2
  Valid result:
  recall@10 : 0.2546    mrr@10 : 0.2049    ndcg@10 : 0.1831    hit@10 : 0.3906    precision@10 : 0.0535
  Test result:
  recall@10 : 0.267     mrr@10 : 0.2325    ndcg@10 : 0.2019    hit@10 : 0.4026    precision@10 : 0.0579
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 28/28 [3:31:09<00:00, 452.47s/trial, best loss: -0.1831]
  best params:  {'k': 10, 'shrink': 1.2}
  best result: 
  {'model': 'ItemKNN', 'best_valid_score': 0.1831, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.2546), ('mrr@10', 0.2049), ('ndcg@10', 0.1831), ('hit@10', 0.3906), ('precision@10', 0.0535)]), 'test_result': OrderedDict([('recall@10', 0.267), ('mrr@10', 0.2325), ('ndcg@10', 0.2019), ('hit@10', 0.4026), ('precision@10', 0.0579)])}
  ```
