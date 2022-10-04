# General Recommendation

- **Dataset**: [Yelp2022](../../md/yelp_general.md)

- **Model**: [NeuMF](https://recbole.io/docs/user_guide/model/general/neumf.html)

- **Time cost**: 10115.78s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [1e-4,5e-4,1e-3,5e-3]
  mlp_hidden_size choice ['[64,32,16]','[32,16,8]']
  dropout_prob choice [0.1,0.0]
  ```
  
- **Best parameters**:

  ```yaml
  learning_rate: 5e-3
  mlp_hidden_size: '[64,32,16]'
  dropout_prob: 0.0
  ```
  
- **Hyper-parameter logging**:

  ```yaml
  dropout_prob:0.0, learning_rate:0.001, mlp_hidden_size:[32,16,8]
  Valid result:
  recall@10 : 0.0508    mrr@10 : 0.0332    ndcg@10 : 0.0295    hit@10 : 0.0959    precision@10 : 0.0106
  Test result:
  recall@10 : 0.0519    mrr@10 : 0.034    ndcg@10 : 0.0304    hit@10 : 0.0968    precision@10 : 0.0108
  
  dropout_prob:0.0, learning_rate:0.0001, mlp_hidden_size:[32,16,8]
  Valid result:
  recall@10 : 0.0428    mrr@10 : 0.0318    ndcg@10 : 0.0265    hit@10 : 0.086    precision@10 : 0.0097
  Test result:
  recall@10 : 0.0426    mrr@10 : 0.0324    ndcg@10 : 0.0266    hit@10 : 0.0866    precision@10 : 0.0098
  
  dropout_prob:0.0, learning_rate:0.005, mlp_hidden_size:[32,16,8]
  Valid result:
  recall@10 : 0.0636    mrr@10 : 0.0382    ndcg@10 : 0.0362    hit@10 : 0.112    precision@10 : 0.0125
  Test result:
  recall@10 : 0.0642    mrr@10 : 0.0391    ndcg@10 : 0.0365    hit@10 : 0.1139    precision@10 : 0.0126
  
  dropout_prob:0.0, learning_rate:0.0001, mlp_hidden_size:[64,32,16]
  Valid result:
  recall@10 : 0.0477    mrr@10 : 0.0355    ndcg@10 : 0.0297    hit@10 : 0.0954    precision@10 : 0.0108
  Test result:
  recall@10 : 0.0476    mrr@10 : 0.0359    ndcg@10 : 0.0297    hit@10 : 0.0957    precision@10 : 0.0108
  
  dropout_prob:0.1, learning_rate:0.0001, mlp_hidden_size:[32,16,8]
  Valid result:
  recall@10 : 0.0521    mrr@10 : 0.0373    ndcg@10 : 0.0318    hit@10 : 0.1024    precision@10 : 0.0116
  Test result:
  recall@10 : 0.0519    mrr@10 : 0.0377    ndcg@10 : 0.0318    hit@10 : 0.1018    precision@10 : 0.0116
  
  dropout_prob:0.1, learning_rate:0.005, mlp_hidden_size:[32,16,8]
  Valid result:
  recall@10 : 0.063    mrr@10 : 0.0393    ndcg@10 : 0.0364    hit@10 : 0.1137    precision@10 : 0.0126
  Test result:
  recall@10 : 0.0629    mrr@10 : 0.0397    ndcg@10 : 0.0364    hit@10 : 0.1138    precision@10 : 0.0127
  
  dropout_prob:0.0, learning_rate:0.005, mlp_hidden_size:[64,32,16]
  Valid result:
  recall@10 : 0.0697    mrr@10 : 0.0435    ndcg@10 : 0.041    hit@10 : 0.1221    precision@10 : 0.0136
  Test result:
  recall@10 : 0.0706    mrr@10 : 0.0427    ndcg@10 : 0.0405    hit@10 : 0.1225    precision@10 : 0.0138
  
  dropout_prob:0.1, learning_rate:0.0005, mlp_hidden_size:[32,16,8]
  Valid result:
  recall@10 : 0.0512    mrr@10 : 0.0361    ndcg@10 : 0.031    hit@10 : 0.0996    precision@10 : 0.0112
  Test result:
  recall@10 : 0.0511    mrr@10 : 0.0362    ndcg@10 : 0.0309    hit@10 : 0.099    precision@10 : 0.0112
  
  dropout_prob:0.1, learning_rate:0.001, mlp_hidden_size:[64,32,16]
  Valid result:
  recall@10 : 0.0531    mrr@10 : 0.0346    ndcg@10 : 0.0309    hit@10 : 0.0996    precision@10 : 0.0111
  Test result:
  recall@10 : 0.0527    mrr@10 : 0.0349    ndcg@10 : 0.0309    hit@10 : 0.1004    precision@10 : 0.0112
  
  dropout_prob:0.0, learning_rate:0.001, mlp_hidden_size:[64,32,16]
  Valid result:
  recall@10 : 0.059    mrr@10 : 0.0368    ndcg@10 : 0.0341    hit@10 : 0.1057    precision@10 : 0.0118
  Test result:
  recall@10 : 0.0578    mrr@10 : 0.0369    ndcg@10 : 0.0335    hit@10 : 0.1058    precision@10 : 0.0119
  
  dropout_prob:0.1, learning_rate:0.0001, mlp_hidden_size:[64,32,16]
  Valid result:
  recall@10 : 0.0537    mrr@10 : 0.0387    ndcg@10 : 0.0331    hit@10 : 0.105    precision@10 : 0.012
  Test result:
  recall@10 : 0.0535    mrr@10 : 0.0383    ndcg@10 : 0.0328    hit@10 : 0.1029    precision@10 : 0.0117
  
  dropout_prob:0.1, learning_rate:0.0005, mlp_hidden_size:[64,32,16]
  Valid result:
  recall@10 : 0.0518    mrr@10 : 0.0363    ndcg@10 : 0.0312    hit@10 : 0.1001    precision@10 : 0.0113
  Test result:
  recall@10 : 0.051    mrr@10 : 0.0351    ndcg@10 : 0.0303    hit@10 : 0.0993    precision@10 : 0.0113
  
  dropout_prob:0.1, learning_rate:0.005, mlp_hidden_size:[64,32,16]
  Valid result:
  recall@10 : 0.0646    mrr@10 : 0.041    ndcg@10 : 0.0378    hit@10 : 0.1156    precision@10 : 0.0129
  Test result:
  recall@10 : 0.0659    mrr@10 : 0.0404    ndcg@10 : 0.0375    hit@10 : 0.1183    precision@10 : 0.0132
  
  dropout_prob:0.0, learning_rate:0.0005, mlp_hidden_size:[64,32,16]
  Valid result:
  recall@10 : 0.0496    mrr@10 : 0.0337    ndcg@10 : 0.0295    hit@10 : 0.0951    precision@10 : 0.0106
  Test result:
  recall@10 : 0.0494    mrr@10 : 0.0334    ndcg@10 : 0.0289    hit@10 : 0.0961    precision@10 : 0.0108
  
  dropout_prob:0.0, learning_rate:0.0005, mlp_hidden_size:[32,16,8]
  Valid result:
  recall@10 : 0.0494    mrr@10 : 0.0347    ndcg@10 : 0.0297    hit@10 : 0.0966    precision@10 : 0.0109
  Test result:
  recall@10 : 0.0495    mrr@10 : 0.0353    ndcg@10 : 0.03    hit@10 : 0.0968    precision@10 : 0.0109
  
  dropout_prob:0.1, learning_rate:0.001, mlp_hidden_size:[32,16,8]
  Valid result:
  recall@10 : 0.0508    mrr@10 : 0.0333    ndcg@10 : 0.0296    hit@10 : 0.0956    precision@10 : 0.0106
  Test result:
  recall@10 : 0.0518    mrr@10 : 0.0335    ndcg@10 : 0.0303    hit@10 : 0.0964    precision@10 : 0.0107
  ```
  
- **Logging Result**:

  ```yaml
  100%|██████████| 16/16 [44:57:32<00:00, 10115.78s/trial, best loss: -0.041]
  best params:  {'dropout_prob': 0.0, 'learning_rate': 0.005, 'mlp_hidden_size': '[64,32,16]'}
  best result: 
  {'model': 'NeuMF', 'best_valid_score': 0.041, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.0697), ('mrr@10', 0.0435), ('ndcg@10', 0.041), ('hit@10', 0.1221), ('precision@10', 0.0136)]), 'test_result': OrderedDict([('recall@10', 0.0706), ('mrr@10', 0.0427), ('ndcg@10', 0.0405), ('hit@10', 0.1225), ('precision@10', 0.0138)])}
  ```
