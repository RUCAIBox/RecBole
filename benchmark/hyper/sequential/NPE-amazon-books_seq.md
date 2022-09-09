# Sequential Recommendation

- **Dataset**: [Amazon-Books](../../md/amazon-books_seq.md)

- **Model**: [NPE](https://recbole.io/docs/user_guide/model/sequential/npe.html)

- **Time cost**: 12285.31s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [0.0005, 0.001, 0.003]
  dropout_prob choice [0.1, 0.3, 0.5]
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 0.0005
  dropout_prob: 0.1
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  dropout_prob:0.5, learning_rate:0.0005
  Valid result:
  recall@10 : 0.0585    mrr@10 : 0.0185    ndcg@10 : 0.0276    hit@10 : 0.0585    precision@10 : 0.0059
  Test result:
  recall@10 : 0.0617    mrr@10 : 0.0193    ndcg@10 : 0.029    hit@10 : 0.0617    precision@10 : 0.0062

  dropout_prob:0.1, learning_rate:0.0005
  Valid result:
  recall@10 : 0.0668    mrr@10 : 0.0183    ndcg@10 : 0.0293    hit@10 : 0.0668    precision@10 : 0.0067
  Test result:
  recall@10 : 0.0648    mrr@10 : 0.0176    ndcg@10 : 0.0283    hit@10 : 0.0648    precision@10 : 0.0065

  dropout_prob:0.1, learning_rate:0.001
  Valid result:
  recall@10 : 0.0711    mrr@10 : 0.019    ndcg@10 : 0.0308    hit@10 : 0.0711    precision@10 : 0.0071
  Test result:
  recall@10 : 0.066    mrr@10 : 0.0174    ndcg@10 : 0.0285    hit@10 : 0.066    precision@10 : 0.0066

  dropout_prob:0.1, learning_rate:0.003
  Valid result:
  recall@10 : 0.0723    mrr@10 : 0.0198    ndcg@10 : 0.0318    hit@10 : 0.0723    precision@10 : 0.0072
  Test result:
  recall@10 : 0.0633    mrr@10 : 0.018    ndcg@10 : 0.0284    hit@10 : 0.0633    precision@10 : 0.0063

  dropout_prob:0.5, learning_rate:0.003
  Valid result:
  recall@10 : 0.0613    mrr@10 : 0.0183    ndcg@10 : 0.0281    hit@10 : 0.0613    precision@10 : 0.0061
  Test result:
  recall@10 : 0.0603    mrr@10 : 0.0194    ndcg@10 : 0.0288    hit@10 : 0.0603    precision@10 : 0.006

  dropout_prob:0.5, learning_rate:0.001
  Valid result:
  recall@10 : 0.0603    mrr@10 : 0.0197    ndcg@10 : 0.029    hit@10 : 0.0603    precision@10 : 0.006
  Test result:
  recall@10 : 0.0633    mrr@10 : 0.0188    ndcg@10 : 0.029    hit@10 : 0.0633    precision@10 : 0.0063

  dropout_prob:0.3, learning_rate:0.0005
  Valid result:
  recall@10 : 0.0632    mrr@10 : 0.0189    ndcg@10 : 0.029    hit@10 : 0.0632    precision@10 : 0.0063
  Test result:
  recall@10 : 0.0663    mrr@10 : 0.018    ndcg@10 : 0.0289    hit@10 : 0.0663    precision@10 : 0.0066

  dropout_prob:0.3, learning_rate:0.003
  Valid result:
  recall@10 : 0.064    mrr@10 : 0.0193    ndcg@10 : 0.0296    hit@10 : 0.064    precision@10 : 0.0064
  Test result:
  recall@10 : 0.0643    mrr@10 : 0.0184    ndcg@10 : 0.0288    hit@10 : 0.0643    precision@10 : 0.0064

  dropout_prob:0.3, learning_rate:0.001
  Valid result:
  recall@10 : 0.068    mrr@10 : 0.0192    ndcg@10 : 0.0303    hit@10 : 0.068    precision@10 : 0.0068
  Test result:
  recall@10 : 0.0706    mrr@10 : 0.0201    ndcg@10 : 0.0316    hit@10 : 0.0706    precision@10 : 0.0071
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 9/9 [30:42:47<00:00, 12285.31s/trial, best loss: -0.0483]
  best params:  {'dropout_prob': 0.1, 'learning_rate': 0.0005}
  best result: 
  {'model': 'NPE', 'best_valid_score': 0.0483, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.1007), ('mrr@10', 0.0325), ('ndcg@10', 0.0483), ('hit@10', 0.1007), ('precision@10', 0.0101)]), 'test_result': OrderedDict([('recall@10', 0.062), ('mrr@10', 0.0191), ('ndcg@10', 0.0289), ('hit@10', 0.062), ('precision@10', 0.0062)])}
  ```
