# General Recommendation

- **Dataset**: [Yelp2022](../../md/yelp_general.md)

- **Model**: [LINE](https://recbole.io/docs/user_guide/model/general/line.html)

- **Time cost**: 981.19s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [5e-4,1e-3,2e-3]
  tranining_neg_sample_num choice [1,3]
  second_order_loss_weight choice [0.3,0.6,1]
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 0.001
  tranining_neg_sample_num: 3
  second_order_loss_weight: 1
  ```

- **Hyper-parameter logging**:

  ```yaml
  learning_rate:0.001, second_order_loss_weight:1, tranining_neg_sample_num:3
  Valid result:
  recall@10 : 0.0383    mrr@10 : 0.0298    ndcg@10 : 0.0244    hit@10 : 0.077    precision@10 : 0.0085
  Test result:
  recall@10 : 0.0392    mrr@10 : 0.0296    ndcg@10 : 0.0244    hit@10 : 0.0774    precision@10 : 0.0085
  
  learning_rate:0.001, second_order_loss_weight:1, tranining_neg_sample_num:1
  Valid result:
  recall@10 : 0.0383    mrr@10 : 0.0298    ndcg@10 : 0.0244    hit@10 : 0.077    precision@10 : 0.0085
  Test result:
  recall@10 : 0.0392    mrr@10 : 0.0296    ndcg@10 : 0.0244    hit@10 : 0.0774    precision@10 : 0.0085
  
  learning_rate:0.0005, second_order_loss_weight:0.6, tranining_neg_sample_num:1
  Valid result:
  recall@10 : 0.0384    mrr@10 : 0.0294    ndcg@10 : 0.0243    hit@10 : 0.0769    precision@10 : 0.0085
  Test result:
  recall@10 : 0.039    mrr@10 : 0.0294    ndcg@10 : 0.0244    hit@10 : 0.0769    precision@10 : 0.0084
  
  learning_rate:0.001, second_order_loss_weight:0.6, tranining_neg_sample_num:1
  Valid result:
  recall@10 : 0.0382    mrr@10 : 0.0298    ndcg@10 : 0.0244    hit@10 : 0.0769    precision@10 : 0.0085
  Test result:
  recall@10 : 0.039    mrr@10 : 0.0295    ndcg@10 : 0.0244    hit@10 : 0.0772    precision@10 : 0.0085
  
  learning_rate:0.001, second_order_loss_weight:0.3, tranining_neg_sample_num:1
  Valid result:
  recall@10 : 0.0383    mrr@10 : 0.0295    ndcg@10 : 0.0244    hit@10 : 0.0769    precision@10 : 0.0085
  Test result:
  recall@10 : 0.0392    mrr@10 : 0.0294    ndcg@10 : 0.0244    hit@10 : 0.0771    precision@10 : 0.0085
  
  learning_rate:0.002, second_order_loss_weight:0.3, tranining_neg_sample_num:3
  Valid result:
  recall@10 : 0.0382    mrr@10 : 0.0295    ndcg@10 : 0.0242    hit@10 : 0.0766    precision@10 : 0.0085
  Test result:
  recall@10 : 0.0389    mrr@10 : 0.0297    ndcg@10 : 0.0245    hit@10 : 0.077    precision@10 : 0.0085
  
  learning_rate:0.002, second_order_loss_weight:0.6, tranining_neg_sample_num:3
  Valid result:
  recall@10 : 0.0381    mrr@10 : 0.0294    ndcg@10 : 0.0241    hit@10 : 0.0763    precision@10 : 0.0084
  Test result:
  recall@10 : 0.039    mrr@10 : 0.0298    ndcg@10 : 0.0246    hit@10 : 0.0774    precision@10 : 0.0085
  
  learning_rate:0.0005, second_order_loss_weight:1, tranining_neg_sample_num:3
  Valid result:
  recall@10 : 0.0383    mrr@10 : 0.0292    ndcg@10 : 0.0242    hit@10 : 0.0764    precision@10 : 0.0084
  Test result:
  recall@10 : 0.039    mrr@10 : 0.0292    ndcg@10 : 0.0243    hit@10 : 0.0767    precision@10 : 0.0084
  
  learning_rate:0.0005, second_order_loss_weight:0.3, tranining_neg_sample_num:3
  Valid result:
  recall@10 : 0.0383    mrr@10 : 0.0294    ndcg@10 : 0.0243    hit@10 : 0.0767    precision@10 : 0.0085
  Test result:
  recall@10 : 0.0391    mrr@10 : 0.0295    ndcg@10 : 0.0244    hit@10 : 0.077    precision@10 : 0.0085
  
  learning_rate:0.002, second_order_loss_weight:1, tranining_neg_sample_num:1
  Valid result:
  recall@10 : 0.038    mrr@10 : 0.0296    ndcg@10 : 0.0241    hit@10 : 0.0766    precision@10 : 0.0085
  Test result:
  recall@10 : 0.0387    mrr@10 : 0.0297    ndcg@10 : 0.0244    hit@10 : 0.0767    precision@10 : 0.0085
  
  learning_rate:0.002, second_order_loss_weight:0.6, tranining_neg_sample_num:1
  Valid result:
  recall@10 : 0.0381    mrr@10 : 0.0294    ndcg@10 : 0.0241    hit@10 : 0.0763    precision@10 : 0.0084
  Test result:
  recall@10 : 0.039    mrr@10 : 0.0298    ndcg@10 : 0.0246    hit@10 : 0.0774    precision@10 : 0.0085
  
  learning_rate:0.002, second_order_loss_weight:0.3, tranining_neg_sample_num:1
  Valid result:
  recall@10 : 0.0382    mrr@10 : 0.0295    ndcg@10 : 0.0242    hit@10 : 0.0766    precision@10 : 0.0085
  Test result:
  recall@10 : 0.0389    mrr@10 : 0.0297    ndcg@10 : 0.0245    hit@10 : 0.077    precision@10 : 0.0085
  
  learning_rate:0.001, second_order_loss_weight:0.3, tranining_neg_sample_num:3
  Valid result:
  recall@10 : 0.0383    mrr@10 : 0.0295    ndcg@10 : 0.0244    hit@10 : 0.0769    precision@10 : 0.0085
  Test result:
  recall@10 : 0.0392    mrr@10 : 0.0294    ndcg@10 : 0.0244    hit@10 : 0.0771    precision@10 : 0.0085
  
  learning_rate:0.0005, second_order_loss_weight:0.3, tranining_neg_sample_num:1
  Valid result:
  recall@10 : 0.0383    mrr@10 : 0.0294    ndcg@10 : 0.0243    hit@10 : 0.0767    precision@10 : 0.0085
  Test result:
  recall@10 : 0.0391    mrr@10 : 0.0295    ndcg@10 : 0.0244    hit@10 : 0.077    precision@10 : 0.0085
  
  learning_rate:0.0005, second_order_loss_weight:1, tranining_neg_sample_num:1
  Valid result:
  recall@10 : 0.0383    mrr@10 : 0.0292    ndcg@10 : 0.0242    hit@10 : 0.0764    precision@10 : 0.0084
  Test result:
  recall@10 : 0.039    mrr@10 : 0.0292    ndcg@10 : 0.0243    hit@10 : 0.0767    precision@10 : 0.0084
  
  learning_rate:0.002, second_order_loss_weight:1, tranining_neg_sample_num:3
  Valid result:
  recall@10 : 0.038    mrr@10 : 0.0296    ndcg@10 : 0.0241    hit@10 : 0.0766    precision@10 : 0.0085
  Test result:
  recall@10 : 0.0387    mrr@10 : 0.0297    ndcg@10 : 0.0244    hit@10 : 0.0767    precision@10 : 0.0085
  
  learning_rate:0.001, second_order_loss_weight:0.6, tranining_neg_sample_num:3
  Valid result:
  recall@10 : 0.0382    mrr@10 : 0.0298    ndcg@10 : 0.0244    hit@10 : 0.0769    precision@10 : 0.0085
  Test result:
  recall@10 : 0.039    mrr@10 : 0.0295    ndcg@10 : 0.0244    hit@10 : 0.0772    precision@10 : 0.0085
  
  learning_rate:0.0005, second_order_loss_weight:0.6, tranining_neg_sample_num:3
  Valid result:
  recall@10 : 0.0384    mrr@10 : 0.0294    ndcg@10 : 0.0243    hit@10 : 0.0769    precision@10 : 0.0085
  Test result:
  recall@10 : 0.039    mrr@10 : 0.0294    ndcg@10 : 0.0244    hit@10 : 0.0769    precision@10 : 0.0084
  ```
  
- **Logging Result**:

  ```yaml
  100%|██████████| 18/18 [4:54:21<00:00, 981.19s/trial, best loss: -0.0244] 
  best params:  {'learning_rate': 0.001, 'second_order_loss_weight': 1, 'tranining_neg_sample_num': 3}
  best result: 
  {'model': 'LINE', 'best_valid_score': 0.0244, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.0383), ('mrr@10', 0.0298), ('ndcg@10', 0.0244), ('hit@10', 0.077), ('precision@10', 0.0085)]), 'test_result': OrderedDict([('recall@10', 0.0392), ('mrr@10', 0.0296), ('ndcg@10', 0.0244), ('hit@10', 0.0774), ('precision@10', 0.0085)])}
  ```
