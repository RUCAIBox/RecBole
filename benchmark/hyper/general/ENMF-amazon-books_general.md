# General Recommendation

- **Dataset**: [Amazon-Books](../../md/amazon-books_general.md)

- **Model**: [ENMF](https://recbole.io/docs/user_guide/model/general/enmf.html)

- **Time cost**: 884.48s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate in [0.005,0.01,0.05]
  dropout_prob in [0.3,0.5,0.7]
  negative_weight in [0.1,0.2,0.5]
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 0.01
  dropout_prob: 0.3
  negative_weight: 0.1
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  dropout_prob:0.5, learning_rate:0.05, negative_weight:0.2
  Valid result:
  recall@10 : 0.1125    mrr@10 : 0.094    ndcg@10 : 0.078    hit@10 : 0.201    precision@10 : 0.0253
  Test result:
  recall@10 : 0.1134    mrr@10 : 0.101    ndcg@10 : 0.0819    hit@10 : 0.2039    precision@10 : 0.0266

  dropout_prob:0.3, learning_rate:0.005, negative_weight:0.2
  Valid result:
  recall@10 : 0.119    mrr@10 : 0.0966    ndcg@10 : 0.0818    hit@10 : 0.2073    precision@10 : 0.0256
  Test result:
  recall@10 : 0.1216    mrr@10 : 0.104    ndcg@10 : 0.0866    hit@10 : 0.2119    precision@10 : 0.0271

  dropout_prob:0.7, learning_rate:0.01, negative_weight:0.5
  Valid result:
  recall@10 : 0.1041    mrr@10 : 0.0869    ndcg@10 : 0.072    hit@10 : 0.1868    precision@10 : 0.0234
  Test result:
  recall@10 : 0.1044    mrr@10 : 0.0942    ndcg@10 : 0.0757    hit@10 : 0.1883    precision@10 : 0.0246

  dropout_prob:0.5, learning_rate:0.05, negative_weight:0.1
  Valid result:
  recall@10 : 0.1239    mrr@10 : 0.099    ndcg@10 : 0.0843    hit@10 : 0.2164    precision@10 : 0.0267
  Test result:
  recall@10 : 0.1251    mrr@10 : 0.106    ndcg@10 : 0.0885    hit@10 : 0.2179    precision@10 : 0.0279

  dropout_prob:0.7, learning_rate:0.005, negative_weight:0.2
  Valid result:
  recall@10 : 0.1166    mrr@10 : 0.0939    ndcg@10 : 0.0796    hit@10 : 0.203    precision@10 : 0.025
  Test result:
  recall@10 : 0.1169    mrr@10 : 0.0996    ndcg@10 : 0.0828    hit@10 : 0.2042    precision@10 : 0.026

  dropout_prob:0.7, learning_rate:0.05, negative_weight:0.1
  Valid result:
  recall@10 : 0.0377    mrr@10 : 0.0342    ndcg@10 : 0.0264    hit@10 : 0.0802    precision@10 : 0.0097
  Test result:
  recall@10 : 0.0368    mrr@10 : 0.0346    ndcg@10 : 0.0265    hit@10 : 0.0795    precision@10 : 0.01

  dropout_prob:0.3, learning_rate:0.01, negative_weight:0.1
  Valid result:
  recall@10 : 0.1262    mrr@10 : 0.1009    ndcg@10 : 0.0864    hit@10 : 0.2176    precision@10 : 0.0266
  Test result:
  recall@10 : 0.1286    mrr@10 : 0.1077    ndcg@10 : 0.0912    hit@10 : 0.2212    precision@10 : 0.0281

  dropout_prob:0.7, learning_rate:0.005, negative_weight:0.1
  Valid result:
  recall@10 : 0.1252    mrr@10 : 0.0982    ndcg@10 : 0.0846    hit@10 : 0.2151    precision@10 : 0.0261
  Test result:
  recall@10 : 0.1266    mrr@10 : 0.1039    ndcg@10 : 0.0885    hit@10 : 0.2166    precision@10 : 0.0272

  dropout_prob:0.7, learning_rate:0.05, negative_weight:0.2
  Valid result:
  recall@10 : 0.0257    mrr@10 : 0.0246    ndcg@10 : 0.0189    hit@10 : 0.0558    precision@10 : 0.0067
  Test result:
  recall@10 : 0.0244    mrr@10 : 0.0237    ndcg@10 : 0.018    hit@10 : 0.0546    precision@10 : 0.0067

  dropout_prob:0.3, learning_rate:0.01, negative_weight:0.5
  Valid result:
  recall@10 : 0.1042    mrr@10 : 0.0878    ndcg@10 : 0.0722    hit@10 : 0.1887    precision@10 : 0.0237
  Test result:
  recall@10 : 0.1041    mrr@10 : 0.0947    ndcg@10 : 0.076    hit@10 : 0.189    precision@10 : 0.0249

  dropout_prob:0.5, learning_rate:0.01, negative_weight:0.1
  Valid result:
  recall@10 : 0.1256    mrr@10 : 0.0998    ndcg@10 : 0.0855    hit@10 : 0.2159    precision@10 : 0.0264
  Test result:
  recall@10 : 0.128    mrr@10 : 0.1066    ndcg@10 : 0.0901    hit@10 : 0.2207    precision@10 : 0.0279

  dropout_prob:0.5, learning_rate:0.005, negative_weight:0.1
  Valid result:
  recall@10 : 0.1273    mrr@10 : 0.1003    ndcg@10 : 0.0864    hit@10 : 0.2194    precision@10 : 0.0268
  Test result:
  recall@10 : 0.1281    mrr@10 : 0.1076    ndcg@10 : 0.0908    hit@10 : 0.2208    precision@10 : 0.0279

  dropout_prob:0.7, learning_rate:0.01, negative_weight:0.2
  Valid result:
  recall@10 : 0.1168    mrr@10 : 0.0951    ndcg@10 : 0.0802    hit@10 : 0.2054    precision@10 : 0.0254
  Test result:
  recall@10 : 0.1182    mrr@10 : 0.1017    ndcg@10 : 0.0841    hit@10 : 0.2075    precision@10 : 0.0267

  dropout_prob:0.3, learning_rate:0.05, negative_weight:0.2
  Valid result:
  recall@10 : 0.1077    mrr@10 : 0.0908    ndcg@10 : 0.0749    hit@10 : 0.1951    precision@10 : 0.0245
  Test result:
  recall@10 : 0.11    mrr@10 : 0.0971    ndcg@10 : 0.079    hit@10 : 0.1998    precision@10 : 0.0261

  dropout_prob:0.3, learning_rate:0.005, negative_weight:0.5
  Valid result:
  recall@10 : 0.1055    mrr@10 : 0.0896    ndcg@10 : 0.0735    hit@10 : 0.1906    precision@10 : 0.0239
  Test result:
  recall@10 : 0.1069    mrr@10 : 0.0966    ndcg@10 : 0.0778    hit@10 : 0.193    precision@10 : 0.0252

  dropout_prob:0.5, learning_rate:0.005, negative_weight:0.2
  Valid result:
  recall@10 : 0.1181    mrr@10 : 0.0964    ndcg@10 : 0.0814    hit@10 : 0.2069    precision@10 : 0.0256
  Test result:
  recall@10 : 0.1189    mrr@10 : 0.1034    ndcg@10 : 0.0852    hit@10 : 0.209    precision@10 : 0.0268

  dropout_prob:0.3, learning_rate:0.01, negative_weight:0.2
  Valid result:
  recall@10 : 0.1166    mrr@10 : 0.097    ndcg@10 : 0.0809    hit@10 : 0.2069    precision@10 : 0.0257
  Test result:
  recall@10 : 0.1188    mrr@10 : 0.1039    ndcg@10 : 0.0854    hit@10 : 0.2102    precision@10 : 0.0272
  ```

- **Logging Result**:

  ```yaml
  INFO  Early stop triggered. Stopping iterations as condition is reach.
  63%|██████▎   | 17/27 [4:10:36<2:27:24, 884.48s/trial, best loss: -0.0864]
  best params:  {'dropout_prob': 0.3, 'learning_rate': 0.01, 'negative_weight': 0.1}
  best result: 
  {'model': 'ENMF', 'best_valid_score': 0.0864, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.1262), ('mrr@10', 0.1009), ('ndcg@10', 0.0864), ('hit@10', 0.2176), ('precision@10', 0.0266)]), 'test_result': OrderedDict([('recall@10', 0.1286), ('mrr@10', 0.1077), ('ndcg@10', 0.0912), ('hit@10', 0.2212), ('precision@10', 0.0281)])}
  ```
