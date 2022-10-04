# General Recommendation

- **Dataset**: [Amazon-Books](../../md/amazon-books_general.md)

- **Model**: [EASE](https://recbole.io/docs/user_guide/model/general/ease.html)

- **Time cost**: 1335.09s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  reg_weight choice [1.0,10.0,100.0,500.0,1000.0,2000.0]
  ```

- **Best parameters**:

  ```yaml
   reg_weight: 10.0
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  reg_weight:1.0
  Valid result:
  recall@10 : 0.2499    mrr@10 : 0.2068    ndcg@10 : 0.1829    hit@10 : 0.3847    precision@10 : 0.0524
  Test result:
  recall@10 : 0.2577    mrr@10 : 0.2296    ndcg@10 : 0.1981    hit@10 : 0.3943    precision@10 : 0.0555

  reg_weight:10.0
  Valid result:
  recall@10 : 0.2782    mrr@10 : 0.2303    ndcg@10 : 0.204     hit@10 : 0.4213    precision@10 : 0.0595
  Test result:
  recall@10 : 0.2915    mrr@10 : 0.2603    ndcg@10 : 0.225     hit@10 : 0.4353    precision@10 : 0.0645
  
  reg_weight:100.0
  Valid result:
  recall@10 : 0.2719    mrr@10 : 0.218     ndcg@10 : 0.1946    hit@10 : 0.4138    precision@10 : 0.0582
  Test result:
  recall@10 : 0.2849    mrr@10 : 0.2471    ndcg@10 : 0.2148    hit@10 : 0.4274    precision@10 : 0.0631
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 6/6 [2:08:18<00:00, 1335.09s/trial, best loss: -0.204]
  best params:  {'reg_weight': 10.0}
  best result: 
  {'model': 'EASE', 'best_valid_score': 0.204, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.2782), ('mrr@10', 0.2303), ('ndcg@10', 0.204), ('hit@10', 0.4213), ('precision@10', 0.0595)]), 'test_result': OrderedDict([('recall@10', 0.2915), ('mrr@10', 0.2603), ('ndcg@10', 0.225), ('hit@10', 0.4354), ('precision@10', 0.0645)])}
  ```
