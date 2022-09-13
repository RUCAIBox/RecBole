# General Recommendation

- **Dataset**: [Amazon-Books](../../md/amazon-books_general.md)

- **Model**: [FISM](https://recbole.io/docs/user_guide/model/general/fism.html)

- **Time cost**: 1515.00s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate in [5e-4,1e-4,5e-3,1e-3] 
  regs in ['[1e-7, 1e-7]', '[0, 0]']
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 0.0001
  regs: [1e-7,1e-7]
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  learning_rate:0.001, regs:[1e-7,1e-7]
  Valid result:
  recall@10 : 0.011    mrr@10 : 0.0087    ndcg@10 : 0.0081    hit@10 : 0.0198    precision@10 : 0.0021
  Test result:
  recall@10 : 0.01    mrr@10 : 0.0078    ndcg@10 : 0.0071    hit@10 : 0.0183    precision@10 : 0.0019

  learning_rate:0.005, regs:[1e-7,1e-7]
  Valid result:
  recall@10 : 0.0108    mrr@10 : 0.0087    ndcg@10 : 0.008    hit@10 : 0.0199    precision@10 : 0.0021
  Test result:
  recall@10 : 0.0096    mrr@10 : 0.0078    ndcg@10 : 0.007    hit@10 : 0.0182    precision@10 : 0.0019

  learning_rate:0.0001, regs:[1e-7,1e-7]
  Valid result:
  recall@10 : 0.0114    mrr@10 : 0.0086    ndcg@10 : 0.0082    hit@10 : 0.0196    precision@10 : 0.002
  Test result:
  recall@10 : 0.0103    mrr@10 : 0.0078    ndcg@10 : 0.0072    hit@10 : 0.0182    precision@10 : 0.0019

  learning_rate:0.001, regs:[0,0]
  Valid result:
  recall@10 : 0.011    mrr@10 : 0.0087    ndcg@10 : 0.0081    hit@10 : 0.0198    precision@10 : 0.0021
  Test result:
  recall@10 : 0.01    mrr@10 : 0.0078    ndcg@10 : 0.0071    hit@10 : 0.0183    precision@10 : 0.0019

  learning_rate:0.0005, regs:[1e-7,1e-7]
  Valid result:
  recall@10 : 0.0114    mrr@10 : 0.0086    ndcg@10 : 0.0082    hit@10 : 0.0196    precision@10 : 0.002
  Test result:
  recall@10 : 0.0103    mrr@10 : 0.0078    ndcg@10 : 0.0072    hit@10 : 0.0182    precision@10 : 0.0019

  learning_rate:0.0005, regs:[0,0]
  Valid result:
  recall@10 : 0.0114    mrr@10 : 0.0086    ndcg@10 : 0.0082    hit@10 : 0.0196    precision@10 : 0.002
  Test result:
  recall@10 : 0.0103    mrr@10 : 0.0078    ndcg@10 : 0.0072    hit@10 : 0.0182    precision@10 : 0.0019

  learning_rate:0.005, regs:[0,0]
  Valid result:
  recall@10 : 0.0108    mrr@10 : 0.0087    ndcg@10 : 0.008    hit@10 : 0.0199    precision@10 : 0.0021
  Test result:
  recall@10 : 0.0096    mrr@10 : 0.0078    ndcg@10 : 0.007    hit@10 : 0.0182    precision@10 : 0.0019

  learning_rate:0.0001, regs:[0,0]
  Valid result:
  recall@10 : 0.0114    mrr@10 : 0.0086    ndcg@10 : 0.0082    hit@10 : 0.0196    precision@10 : 0.002
  Test result:
  recall@10 : 0.0103    mrr@10 : 0.0078    ndcg@10 : 0.0072    hit@10 : 0.0182    precision@10 : 0.0019
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 8/8 [3:22:00<00:00, 1515.00s/trial, best loss: -0.0082]
  best params:  {'learning_rate': 0.0001, 'regs': '[1e-7,1e-7]'}
  best result: 
  {'model': 'FISM', 'best_valid_score': 0.0082, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.0114), ('mrr@10', 0.0086), ('ndcg@10', 0.0082), ('hit@10', 0.0196), ('precision@10', 0.002)]), 'test_result': OrderedDict([('recall@10', 0.0103), ('mrr@10', 0.0078), ('ndcg@10', 0.0072), ('hit@10', 0.0182), ('precision@10', 0.0019)])}
  ```
