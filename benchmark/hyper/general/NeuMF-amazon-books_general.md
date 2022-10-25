# General Recommendation

- **Dataset**: [Amazon-Books](../../md/amazon-books_general.md)

- **Model**: [NeuMF](https://recbole.io/docs/user_guide/model/general/neumf.html)

- **Time cost**: 7845.05s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [5e-7,1e-6,5e-6,1e-5,1e-4,1e-3]
  mlp_hidden_size choice ['[64,32,16]']
  dropout_prob choice [0.1,0.0,0.3]
  ```
  
- **Best parameters**:

  ```yaml
  dropout_prob: 0.0
  learning_rate: 0.0001  
  mlp_hidden_size: '[64,32,16]'
  ```
  
- **Hyper-parameter logging** (hyper.result):

  ```yaml
  dropout_prob:0.0, learning_rate:1e-05, mlp_hidden_size:[64,32,16]
  Valid result:
  recall@10 : 0.1291    mrr@10 : 0.0814    ndcg@10 : 0.0789    hit@10 : 0.2083    precision@10 : 0.0244
  Test result:
  recall@10 : 0.1318    mrr@10 : 0.086    ndcg@10 : 0.0824    hit@10 : 0.2118    precision@10 : 0.0254
  
  dropout_prob:0.3, learning_rate:0.001, mlp_hidden_size:[64,32,16]
  Valid result:
  recall@10 : 0.1331    mrr@10 : 0.0871    ndcg@10 : 0.0822    hit@10 : 0.2142    precision@10 : 0.0252
  Test result:
  recall@10 : 0.1364    mrr@10 : 0.0934    ndcg@10 : 0.0874    hit@10 : 0.2193    precision@10 : 0.0265
  
  dropout_prob:0.1, learning_rate:0.0001, mlp_hidden_size:[64,32,16]
  Valid result:
  recall@10 : 0.1392    mrr@10 : 0.0906    ndcg@10 : 0.0861    hit@10 : 0.2222    precision@10 : 0.0263
  Test result:
  recall@10 : 0.1413    mrr@10 : 0.0954    ndcg@10 : 0.0901    hit@10 : 0.225    precision@10 : 0.0272
  
  dropout_prob:0.1, learning_rate:1e-05, mlp_hidden_size:[64,32,16]
  Valid result:
  recall@10 : 0.1312    mrr@10 : 0.081    ndcg@10 : 0.0792    hit@10 : 0.2105    precision@10 : 0.0246
  Test result:
  recall@10 : 0.1332    mrr@10 : 0.0848    ndcg@10 : 0.0824    hit@10 : 0.2131    precision@10 : 0.0255
  
  dropout_prob:0.3, learning_rate:0.0001, mlp_hidden_size:[64,32,16]
  Valid result:
  recall@10 : 0.1369    mrr@10 : 0.0904    ndcg@10 : 0.0852    hit@10 : 0.2193    precision@10 : 0.026
  Test result:
  recall@10 : 0.1396    mrr@10 : 0.0957    ndcg@10 : 0.0897    hit@10 : 0.2236    precision@10 : 0.0271
  
  dropout_prob:0.1, learning_rate:5e-07, mlp_hidden_size:[64,32,16]
  Valid result:
  recall@10 : 0.1316    mrr@10 : 0.079    ndcg@10 : 0.0784    hit@10 : 0.2105    precision@10 : 0.0246
  Test result:
  recall@10 : 0.1352    mrr@10 : 0.0834    ndcg@10 : 0.0822    hit@10 : 0.216    precision@10 : 0.0258
  
  dropout_prob:0.0, learning_rate:0.0001, mlp_hidden_size:[64,32,16]
  Valid result:
  recall@10 : 0.1403    mrr@10 : 0.0904    ndcg@10 : 0.0864    hit@10 : 0.2229    precision@10 : 0.0262
  Test result:
  recall@10 : 0.143    mrr@10 : 0.0957    ndcg@10 : 0.0909    hit@10 : 0.2267    precision@10 : 0.0273
  
  dropout_prob:0.1, learning_rate:5e-06, mlp_hidden_size:[64,32,16]
  Valid result:
  recall@10 : 0.1316    mrr@10 : 0.0809    ndcg@10 : 0.0792    hit@10 : 0.2108    precision@10 : 0.0247
  Test result:
  recall@10 : 0.1339    mrr@10 : 0.0849    ndcg@10 : 0.0826    hit@10 : 0.2141    precision@10 : 0.0256
  
  dropout_prob:0.0, learning_rate:5e-07, mlp_hidden_size:[64,32,16]
  Valid result:
  recall@10 : 0.1306    mrr@10 : 0.0816    ndcg@10 : 0.0795    hit@10 : 0.211    precision@10 : 0.0247
  Test result:
  recall@10 : 0.1332    mrr@10 : 0.0849    ndcg@10 : 0.0821    hit@10 : 0.2135    precision@10 : 0.0257
  
  dropout_prob:0.1, learning_rate:0.001, mlp_hidden_size:[64,32,16]
  Valid result:
  recall@10 : 0.1355    mrr@10 : 0.0872    ndcg@10 : 0.0833    hit@10 : 0.2161    precision@10 : 0.0254
  Test result:
  recall@10 : 0.1377    mrr@10 : 0.092    ndcg@10 : 0.0875    hit@10 : 0.2188    precision@10 : 0.0261
  
  dropout_prob:0.3, learning_rate:1e-05, mlp_hidden_size:[64,32,16]
  Valid result:
  recall@10 : 0.1249    mrr@10 : 0.0759    ndcg@10 : 0.0747    hit@10 : 0.2002    precision@10 : 0.0234
  Test result:
  recall@10 : 0.1277    mrr@10 : 0.0801    ndcg@10 : 0.0781    hit@10 : 0.2039    precision@10 : 0.0243
  
  dropout_prob:0.3, learning_rate:5e-07, mlp_hidden_size:[64,32,16]
  Valid result:
  recall@10 : 0.1263    mrr@10 : 0.073    ndcg@10 : 0.0738    hit@10 : 0.2005    precision@10 : 0.0233
  Test result:
  recall@10 : 0.129    mrr@10 : 0.0781    ndcg@10 : 0.0776    hit@10 : 0.2047    precision@10 : 0.0243
  
  dropout_prob:0.0, learning_rate:1e-06, mlp_hidden_size:[64,32,16]
  Valid result:
  recall@10 : 0.1305    mrr@10 : 0.0827    ndcg@10 : 0.0798    hit@10 : 0.2112    precision@10 : 0.0247
  Test result:
  recall@10 : 0.1331    mrr@10 : 0.0866    ndcg@10 : 0.0831    hit@10 : 0.2138    precision@10 : 0.0257
  
  dropout_prob:0.3, learning_rate:5e-06, mlp_hidden_size:[64,32,16]
  Valid result:
  recall@10 : 0.1246    mrr@10 : 0.0756    ndcg@10 : 0.0746    hit@10 : 0.1997    precision@10 : 0.0233
  Test result:
  recall@10 : 0.127    mrr@10 : 0.0803    ndcg@10 : 0.078    hit@10 : 0.2032    precision@10 : 0.0242
  
  dropout_prob:0.1, learning_rate:1e-06, mlp_hidden_size:[64,32,16]
  Valid result:
  recall@10 : 0.1322    mrr@10 : 0.0803    ndcg@10 : 0.0793    hit@10 : 0.2119    precision@10 : 0.0248
  Test result:
  recall@10 : 0.1353    mrr@10 : 0.0856    ndcg@10 : 0.0833    hit@10 : 0.216    precision@10 : 0.0259
  
  dropout_prob:0.3, learning_rate:1e-06, mlp_hidden_size:[64,32,16]
  Valid result:
  recall@10 : 0.1262    mrr@10 : 0.0747    ndcg@10 : 0.0745    hit@10 : 0.2009    precision@10 : 0.0234
  Test result:
  recall@10 : 0.1287    mrr@10 : 0.0792    ndcg@10 : 0.078    hit@10 : 0.2046    precision@10 : 0.0244
  
  dropout_prob:0.0, learning_rate:5e-06, mlp_hidden_size:[64,32,16]
  Valid result:
  recall@10 : 0.1303    mrr@10 : 0.0825    ndcg@10 : 0.0796    hit@10 : 0.211    precision@10 : 0.0246
  Test result:
  recall@10 : 0.1325    mrr@10 : 0.0863    ndcg@10 : 0.0827    hit@10 : 0.2125    precision@10 : 0.0255
  
  dropout_prob:0.0, learning_rate:0.001, mlp_hidden_size:[64,32,16]
  Valid result:
  recall@10 : 0.1369    mrr@10 : 0.0878    ndcg@10 : 0.0844    hit@10 : 0.2181    precision@10 : 0.0256
  Test result:
  recall@10 : 0.1393    mrr@10 : 0.0933    ndcg@10 : 0.0886    hit@10 : 0.2206    precision@10 : 0.0264
  ```

- **Logging Result**:

  ```yaml
  100%|██████████| 18/18 [39:13:30<00:00, 7845.05s/trial, best loss: -0.0864]
  best params:  {'dropout_prob': 0.0, 'learning_rate': 0.0001, 'mlp_hidden_size': '[64,32,16]'}
  best result: 
  {'model': 'NeuMF', 'best_valid_score': 0.0864, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.1403), ('mrr@10', 0.0904), ('ndcg@10', 0.0864), ('hit@10', 0.2229), ('precision@10', 0.0262)]), 'test_result': OrderedDict([('recall@10', 0.143), ('mrr@10', 0.0957), ('ndcg@10', 0.0909), ('hit@10', 0.2267), ('precision@10', 0.0273)])}
  ```
