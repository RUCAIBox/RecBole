# General Recommendation

- **Dataset**: [Yelp2022](../../md/yelp_general.md)

- **Model**: [GCMC](https://recbole.io/docs/user_guide/model/general/gcmc.html)

- **Time cost**: 31194.12s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [5e-4,1e-3,0.01]
  accum choice ['stack','sum']
  dropout_prob choice [0.3,0.5]
  gcn_output_dim choice [500,1024]
  ```
  
- **Best parameters**:

  ```yaml
  learning_rate: 5e-4
  accum: 'stack'
  dropout_prob: 0.5
  gcn_output_dim: 500
  ```
  
- **Hyper-parameter logging**:

  ```yaml
  accum:sum, dropout_prob:0.3, gcn_output_dim:500, learning_rate:0.0005, num_basis_functions:2
  Valid result:
  recall@10 : 0.0605    mrr@10 : 0.0418    ndcg@10 : 0.0362    hit@10 : 0.1154    precision@10 : 0.0131
  Test result:
  recall@10 : 0.061    mrr@10 : 0.0422    ndcg@10 : 0.0365    hit@10 : 0.1165    precision@10 : 0.0132
  
  accum:stack, dropout_prob:0.3, gcn_output_dim:1024, learning_rate:0.0005, num_basis_functions:2
  Valid result:
  recall@10 : 0.0535    mrr@10 : 0.0366    ndcg@10 : 0.0319    hit@10 : 0.1027    precision@10 : 0.0115
  Test result:
  recall@10 : 0.0542    mrr@10 : 0.0372    ndcg@10 : 0.0324    hit@10 : 0.1035    precision@10 : 0.0115
  
  accum:stack, dropout_prob:0.3, gcn_output_dim:500, learning_rate:0.01, num_basis_functions:2
  Valid result:
  recall@10 : 0.0    mrr@10 : 0.0    ndcg@10 : 0.0    hit@10 : 0.0    precision@10 : 0.0
  Test result:
  recall@10 : 0.0    mrr@10 : 0.0    ndcg@10 : 0.0    hit@10 : 0.0    precision@10 : 0.0
  
  accum:sum, dropout_prob:0.5, gcn_output_dim:500, learning_rate:0.01, num_basis_functions:2
  Valid result:
  recall@10 : 0.0    mrr@10 : 0.0    ndcg@10 : 0.0    hit@10 : 0.0    precision@10 : 0.0
  Test result:
  recall@10 : 0.0    mrr@10 : 0.0    ndcg@10 : 0.0    hit@10 : 0.0    precision@10 : 0.0
  
  accum:sum, dropout_prob:0.5, gcn_output_dim:500, learning_rate:0.001, num_basis_functions:2
  Valid result:
  recall@10 : 0.0597    mrr@10 : 0.0432    ndcg@10 : 0.0364    hit@10 : 0.1168    precision@10 : 0.0132
  Test result:
  recall@10 : 0.062    mrr@10 : 0.0443    ndcg@10 : 0.0378    hit@10 : 0.1192    precision@10 : 0.0136
  
  accum:stack, dropout_prob:0.3, gcn_output_dim:500, learning_rate:0.001, num_basis_functions:2
  Valid result:
  recall@10 : 0.0626    mrr@10 : 0.0439    ndcg@10 : 0.038    hit@10 : 0.1192    precision@10 : 0.0135
  Test result:
  recall@10 : 0.0649    mrr@10 : 0.0445    ndcg@10 : 0.0387    hit@10 : 0.1216    precision@10 : 0.0138
  
  accum:sum, dropout_prob:0.3, gcn_output_dim:500, learning_rate:0.001, num_basis_functions:2
  Valid result:
  recall@10 : 0.0626    mrr@10 : 0.0439    ndcg@10 : 0.038    hit@10 : 0.1192    precision@10 : 0.0135
  Test result:
  recall@10 : 0.0649    mrr@10 : 0.0445    ndcg@10 : 0.0387    hit@10 : 0.1216    precision@10 : 0.0138
  
  accum:sum, dropout_prob:0.3, gcn_output_dim:1024, learning_rate:0.0005, num_basis_functions:2
  Valid result:
  recall@10 : 0.0535    mrr@10 : 0.0366    ndcg@10 : 0.0319    hit@10 : 0.1027    precision@10 : 0.0115
  Test result:
  recall@10 : 0.0542    mrr@10 : 0.0372    ndcg@10 : 0.0324    hit@10 : 0.1035    precision@10 : 0.0115
  
  accum:sum, dropout_prob:0.5, gcn_output_dim:1024, learning_rate:0.0005, num_basis_functions:2
  Valid result:
  recall@10 : 0.0551    mrr@10 : 0.039    ndcg@10 : 0.0332    hit@10 : 0.1078    precision@10 : 0.0122
  Test result:
  recall@10 : 0.0574    mrr@10 : 0.0407    ndcg@10 : 0.0348    hit@10 : 0.1113    precision@10 : 0.0126
  
  accum:stack, dropout_prob:0.3, gcn_output_dim:1024, learning_rate:0.001, num_basis_functions:2
  Valid result:
  recall@10 : 0.0503    mrr@10 : 0.0352    ndcg@10 : 0.0303    hit@10 : 0.0988    precision@10 : 0.011
  Test result:
  recall@10 : 0.051    mrr@10 : 0.036    ndcg@10 : 0.0306    hit@10 : 0.0999    precision@10 : 0.0112
  
  accum:sum, dropout_prob:0.3, gcn_output_dim:500, learning_rate:0.01, num_basis_functions:2
  Valid result:
  recall@10 : 0.0    mrr@10 : 0.0    ndcg@10 : 0.0    hit@10 : 0.0    precision@10 : 0.0
  Test result:
  recall@10 : 0.0    mrr@10 : 0.0    ndcg@10 : 0.0    hit@10 : 0.0    precision@10 : 0.0
  
  accum:sum, dropout_prob:0.5, gcn_output_dim:1024, learning_rate:0.01, num_basis_functions:2
  Valid result:
  recall@10 : 0.0436    mrr@10 : 0.0321    ndcg@10 : 0.0268    hit@10 : 0.0871    precision@10 : 0.0096
  Test result:
  recall@10 : 0.0445    mrr@10 : 0.0319    ndcg@10 : 0.0269    hit@10 : 0.0872    precision@10 : 0.0096
  
  accum:sum, dropout_prob:0.3, gcn_output_dim:1024, learning_rate:0.001, num_basis_functions:2
  Valid result:
  recall@10 : 0.0503    mrr@10 : 0.0352    ndcg@10 : 0.0303    hit@10 : 0.0988    precision@10 : 0.011
  Test result:
  recall@10 : 0.051    mrr@10 : 0.036    ndcg@10 : 0.0306    hit@10 : 0.0999    precision@10 : 0.0112
  
  accum:stack, dropout_prob:0.5, gcn_output_dim:500, learning_rate:0.01, num_basis_functions:2
  Valid result:
  recall@10 : 0.0    mrr@10 : 0.0    ndcg@10 : 0.0    hit@10 : 0.0    precision@10 : 0.0
  Test result:
  recall@10 : 0.0    mrr@10 : 0.0    ndcg@10 : 0.0    hit@10 : 0.0    precision@10 : 0.0
  
  accum:stack, dropout_prob:0.5, gcn_output_dim:500, learning_rate:0.001, num_basis_functions:2
  Valid result:
  recall@10 : 0.0597    mrr@10 : 0.0432    ndcg@10 : 0.0364    hit@10 : 0.1168    precision@10 : 0.0132
  Test result:
  recall@10 : 0.062    mrr@10 : 0.0443    ndcg@10 : 0.0378    hit@10 : 0.1192    precision@10 : 0.0136
  
  accum:stack, dropout_prob:0.3, gcn_output_dim:1024, learning_rate:0.01, num_basis_functions:2
  Valid result:
  recall@10 : 0.0486    mrr@10 : 0.0349    ndcg@10 : 0.0294    hit@10 : 0.0973    precision@10 : 0.0109
  Test result:
  recall@10 : 0.0503    mrr@10 : 0.036    ndcg@10 : 0.0305    hit@10 : 0.0995    precision@10 : 0.0112
  
  accum:stack, dropout_prob:0.5, gcn_output_dim:500, learning_rate:0.0005, num_basis_functions:2
  Valid result:
  recall@10 : 0.065    mrr@10 : 0.0463    ndcg@10 : 0.0396    hit@10 : 0.125    precision@10 : 0.0143
  Test result:
  recall@10 : 0.0673    mrr@10 : 0.048    ndcg@10 : 0.0411    hit@10 : 0.1277    precision@10 : 0.0147
  
  accum:stack, dropout_prob:0.5, gcn_output_dim:1024, learning_rate:0.001, num_basis_functions:2
  Valid result:
  recall@10 : 0.0486    mrr@10 : 0.0354    ndcg@10 : 0.0295    hit@10 : 0.0967    precision@10 : 0.0108
  Test result:
  recall@10 : 0.0497    mrr@10 : 0.0355    ndcg@10 : 0.0299    hit@10 : 0.098    precision@10 : 0.011
  
  accum:sum, dropout_prob:0.3, gcn_output_dim:1024, learning_rate:0.01, num_basis_functions:2
  Valid result:
  recall@10 : 0.0486    mrr@10 : 0.0349    ndcg@10 : 0.0294    hit@10 : 0.0973    precision@10 : 0.0109
  Test result:
  recall@10 : 0.0503    mrr@10 : 0.036    ndcg@10 : 0.0305    hit@10 : 0.0995    precision@10 : 0.0112
  
  accum:stack, dropout_prob:0.3, gcn_output_dim:500, learning_rate:0.0005, num_basis_functions:2
  Valid result:
  recall@10 : 0.0605    mrr@10 : 0.0418    ndcg@10 : 0.0362    hit@10 : 0.1154    precision@10 : 0.0131
  Test result:
  recall@10 : 0.061    mrr@10 : 0.0422    ndcg@10 : 0.0365    hit@10 : 0.1165    precision@10 : 0.0132
  
  accum:stack, dropout_prob:0.5, gcn_output_dim:1024, learning_rate:0.01, num_basis_functions:2
  Valid result:
  recall@10 : 0.0436    mrr@10 : 0.0321    ndcg@10 : 0.0268    hit@10 : 0.0871    precision@10 : 0.0096
  Test result:
  recall@10 : 0.0445    mrr@10 : 0.0319    ndcg@10 : 0.0269    hit@10 : 0.0872    precision@10 : 0.0096
  
  accum:stack, dropout_prob:0.5, gcn_output_dim:1024, learning_rate:0.0005, num_basis_functions:2
  Valid result:
  recall@10 : 0.0551    mrr@10 : 0.039    ndcg@10 : 0.0332    hit@10 : 0.1078    precision@10 : 0.0122
  Test result:
  recall@10 : 0.0574    mrr@10 : 0.0407    ndcg@10 : 0.0348    hit@10 : 0.1113    precision@10 : 0.0126
  
  accum:sum, dropout_prob:0.5, gcn_output_dim:1024, learning_rate:0.001, num_basis_functions:2
  Valid result:
  recall@10 : 0.0486    mrr@10 : 0.0354    ndcg@10 : 0.0295    hit@10 : 0.0967    precision@10 : 0.0108
  Test result:
  recall@10 : 0.0497    mrr@10 : 0.0355    ndcg@10 : 0.0299    hit@10 : 0.098    precision@10 : 0.011
  
  accum:sum, dropout_prob:0.5, gcn_output_dim:500, learning_rate:0.0005, num_basis_functions:2
  Valid result:
  recall@10 : 0.065    mrr@10 : 0.0463    ndcg@10 : 0.0396    hit@10 : 0.125    precision@10 : 0.0143
  Test result:
  recall@10 : 0.0673    mrr@10 : 0.048    ndcg@10 : 0.0411    hit@10 : 0.1277    precision@10 : 0.0147
  ```
  
- **Logging Result**:

  ```yaml
  100%|██████████| 24/24 [207:57:38<00:00, 31194.12s/trial, best loss: -0.0396]
  best params:  {'accum': 'stack', 'dropout_prob': 0.5, 'gcn_output_dim': 500, 'learning_rate': 0.0005, 'num_basis_functions': '2'}
  best result: 
  {'model': 'GCMC', 'best_valid_score': 0.0396, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.065), ('mrr@10', 0.0463), ('ndcg@10', 0.0396), ('hit@10', 0.125), ('precision@10', 0.0143)]), 'test_result': OrderedDict([('recall@10', 0.0673), ('mrr@10', 0.048), ('ndcg@10', 0.0411), ('hit@10', 0.1277), ('precision@10', 0.0147)])}
  ```
