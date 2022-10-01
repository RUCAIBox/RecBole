# General Recommendation

- **Dataset**: [Amazon-Books](../../md/amazon-books_general.md)

- **Model**: [GCMC](https://recbole.io/docs/user_guide/model/general/gcmc.html)

- **Time cost**: 9380.44s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  accum in ['stack','sum'] 
  learning_rate in [0.001,0.005,0.01] 
  dropout_prob in [0.3,0.5,0.7] 
  ```

- **Best parameters**:

  ```yaml
  accum: stack
  learning_rate: 0.001
  dropout_prob: 0.3
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  accum:stack, dropout_prob:0.5, learning_rate:0.01
  Valid result:
  recall@10 : 0.0    mrr@10 : 0.0    ndcg@10 : 0.0    hit@10 : 0.0    precision@10 : 0.0
  Test result:
  recall@10 : 0.0    mrr@10 : 0.0    ndcg@10 : 0.0    hit@10 : 0.0    precision@10 : 0.0

  accum:stack, dropout_prob:0.7, learning_rate:0.001
  Valid result:
  recall@10 : 0.0706    mrr@10 : 0.0521    ndcg@10 : 0.0449    hit@10 : 0.1294    precision@10 : 0.0152
  Test result:
  recall@10 : 0.0704    mrr@10 : 0.0544    ndcg@10 : 0.0461    hit@10 : 0.1307    precision@10 : 0.0158

  accum:sum, dropout_prob:0.5, learning_rate:0.001
  Valid result:
  recall@10 : 0.107    mrr@10 : 0.0757    ndcg@10 : 0.068    hit@10 : 0.1806    precision@10 : 0.0217
  Test result:
  recall@10 : 0.1077    mrr@10 : 0.0797    ndcg@10 : 0.0705    hit@10 : 0.1824    precision@10 : 0.0226

  accum:stack, dropout_prob:0.7, learning_rate:0.005
  Valid result:
  recall@10 : 0.023    mrr@10 : 0.0204    ndcg@10 : 0.016    hit@10 : 0.047    precision@10 : 0.0054
  Test result:
  recall@10 : 0.0223    mrr@10 : 0.0192    ndcg@10 : 0.0151    hit@10 : 0.0468    precision@10 : 0.0054

  accum:stack, dropout_prob:0.3, learning_rate:0.001
  Valid result:
  recall@10 : 0.1228    mrr@10 : 0.0847    ndcg@10 : 0.0782    hit@10 : 0.2001    precision@10 : 0.0238
  Test result:
  recall@10 : 0.126    mrr@10 : 0.0908    ndcg@10 : 0.0825    hit@10 : 0.2046    precision@10 : 0.025

  accum:stack, dropout_prob:0.3, learning_rate:0.01
  Valid result:
  recall@10 : 0.0    mrr@10 : 0.0    ndcg@10 : 0.0    hit@10 : 0.0    precision@10 : 0.0
  Test result:
  recall@10 : 0.0    mrr@10 : 0.0    ndcg@10 : 0.0    hit@10 : 0.0    precision@10 : 0.0

  accum:sum, dropout_prob:0.3, learning_rate:0.005
  Valid result:
  recall@10 : 0.0585    mrr@10 : 0.045    ndcg@10 : 0.0376    hit@10 : 0.1121    precision@10 : 0.013
  Test result:
  recall@10 : 0.0579    mrr@10 : 0.0453    ndcg@10 : 0.0377    hit@10 : 0.1116    precision@10 : 0.0132

  accum:stack, dropout_prob:0.3, learning_rate:0.005
  Valid result:
  recall@10 : 0.0585    mrr@10 : 0.045    ndcg@10 : 0.0376    hit@10 : 0.1121    precision@10 : 0.013
  Test result:
  recall@10 : 0.0579    mrr@10 : 0.0453    ndcg@10 : 0.0377    hit@10 : 0.1116    precision@10 : 0.0132

  accum:sum, dropout_prob:0.3, learning_rate:0.001
  Valid result:
  recall@10 : 0.1228    mrr@10 : 0.0847    ndcg@10 : 0.0782    hit@10 : 0.2001    precision@10 : 0.0238
  Test result:
  recall@10 : 0.126    mrr@10 : 0.0908    ndcg@10 : 0.0825    hit@10 : 0.2046    precision@10 : 0.025

  accum:stack, dropout_prob:0.5, learning_rate:0.005
  Valid result:
  recall@10 : 0.0424    mrr@10 : 0.0343    ndcg@10 : 0.0277    hit@10 : 0.0848    precision@10 : 0.0097
  Test result:
  recall@10 : 0.0414    mrr@10 : 0.0343    ndcg@10 : 0.0274    hit@10 : 0.0845    precision@10 : 0.0099

  accum:sum, dropout_prob:0.5, learning_rate:0.01
  Valid result:
  recall@10 : 0.0    mrr@10 : 0.0    ndcg@10 : 0.0    hit@10 : 0.0    precision@10 : 0.0
  Test result:
  recall@10 : 0.0    mrr@10 : 0.0    ndcg@10 : 0.0    hit@10 : 0.0    precision@10 : 0.0

  accum:sum, dropout_prob:0.7, learning_rate:0.005
  Valid result:
  recall@10 : 0.023    mrr@10 : 0.0204    ndcg@10 : 0.016    hit@10 : 0.047    precision@10 : 0.0054
  Test result:
  recall@10 : 0.0223    mrr@10 : 0.0192    ndcg@10 : 0.0151    hit@10 : 0.0468    precision@10 : 0.0054

  accum:sum, dropout_prob:0.7, learning_rate:0.001
  Valid result:
  recall@10 : 0.0706    mrr@10 : 0.0521    ndcg@10 : 0.0449    hit@10 : 0.1294    precision@10 : 0.0152
  Test result:
  recall@10 : 0.0704    mrr@10 : 0.0544    ndcg@10 : 0.0461    hit@10 : 0.1307    precision@10 : 0.0158

  accum:stack, dropout_prob:0.5, learning_rate:0.001
  Valid result:
  recall@10 : 0.107    mrr@10 : 0.0757    ndcg@10 : 0.068    hit@10 : 0.1806    precision@10 : 0.0217
  Test result:
  recall@10 : 0.1077    mrr@10 : 0.0797    ndcg@10 : 0.0705    hit@10 : 0.1824    precision@10 : 0.0226

  accum:sum, dropout_prob:0.5, learning_rate:0.005
  Valid result:
  recall@10 : 0.0424    mrr@10 : 0.0343    ndcg@10 : 0.0277    hit@10 : 0.0848    precision@10 : 0.0097
  Test result:
  recall@10 : 0.0414    mrr@10 : 0.0343    ndcg@10 : 0.0274    hit@10 : 0.0845    precision@10 : 0.0099
  ```

- **Logging Result**:

  ```yaml
  INFO  Early stop triggered. Stopping iterations as condition is reach.
  83%|████████▎ | 15/18 [39:05:06<7:49:01, 9380.44s/trial, best loss: -0.0782]
  best params:  {'accum': 'stack', 'dropout_prob': 0.3, 'learning_rate': 0.001}
  best result: 
  {'model': 'GCMC', 'best_valid_score': 0.0782, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.1228), ('mrr@10', 0.0847), ('ndcg@10', 0.0782), ('hit@10', 0.2001), ('precision@10', 0.0238)]), 'test_result': OrderedDict([('recall@10', 0.126), ('mrr@10', 0.0908), ('ndcg@10', 0.0825), ('hit@10', 0.2046), ('precision@10', 0.025)])}
  ```
