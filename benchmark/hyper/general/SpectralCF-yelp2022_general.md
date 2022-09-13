# General Recommendation

- **Dataset**: [Yelp2022](../../md/yelp_general.md)

- **Model**: [SpectralCF](https://recbole.io/docs/user_guide/model/general/spectralcf.html)

- **Time cost**: 10167.87s/trial

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [1e-4,5e-4,1e-3,2e-3,5e-3,0.01]
  reg_weight choice [0.002,0.001,0.0005]
  n_layers choice [1,2,3]
  ```
  
- **Best parameters**:

  ```yaml
  learning_rate: 2e-3
  reg_weight: 0.0005
  n_layers: 3
  ```
  
- **Hyper-parameter logging**:

  ```yaml
  learning_rate:0.01, n_layers:2, reg_weight:0.002
  Valid result:
  recall@10 : 0.0395    mrr@10 : 0.0294    ndcg@10 : 0.0244    hit@10 : 0.081    precision@10 : 0.0089
  Test result:
  recall@10 : 0.0392    mrr@10 : 0.0294    ndcg@10 : 0.0241    hit@10 : 0.0806    precision@10 : 0.0089
  
  learning_rate:0.0005, n_layers:1, reg_weight:0.002
  Valid result:
  recall@10 : 0.0384    mrr@10 : 0.0283    ndcg@10 : 0.0231    hit@10 : 0.0789    precision@10 : 0.0087
  Test result:
  recall@10 : 0.0377    mrr@10 : 0.028    ndcg@10 : 0.0231    hit@10 : 0.078    precision@10 : 0.0087
  
  learning_rate:0.002, n_layers:2, reg_weight:0.002
  Valid result:
  recall@10 : 0.0574    mrr@10 : 0.0409    ndcg@10 : 0.0351    hit@10 : 0.1113    precision@10 : 0.0127
  Test result:
  recall@10 : 0.0585    mrr@10 : 0.0415    ndcg@10 : 0.0355    hit@10 : 0.1138    precision@10 : 0.0131
  
  learning_rate:0.001, n_layers:1, reg_weight:0.001
  Valid result:
  recall@10 : 0.0355    mrr@10 : 0.0264    ndcg@10 : 0.0218    hit@10 : 0.0734    precision@10 : 0.0081
  Test result:
  recall@10 : 0.0355    mrr@10 : 0.0261    ndcg@10 : 0.0216    hit@10 : 0.074    precision@10 : 0.0082
  
  learning_rate:0.01, n_layers:1, reg_weight:0.0005
  Valid result:
  recall@10 : 0.037    mrr@10 : 0.0269    ndcg@10 : 0.0226    hit@10 : 0.0743    precision@10 : 0.0081
  Test result:
  recall@10 : 0.0375    mrr@10 : 0.0272    ndcg@10 : 0.0227    hit@10 : 0.0759    precision@10 : 0.0083
  
  learning_rate:0.01, n_layers:3, reg_weight:0.002
  Valid result:
  recall@10 : 0.0347    mrr@10 : 0.026    ndcg@10 : 0.0214    hit@10 : 0.0713    precision@10 : 0.0078
  Test result:
  recall@10 : 0.0341    mrr@10 : 0.0258    ndcg@10 : 0.0211    hit@10 : 0.0707    precision@10 : 0.0078
  
  learning_rate:0.005, n_layers:2, reg_weight:0.002
  Valid result:
  recall@10 : 0.0467    mrr@10 : 0.0338    ndcg@10 : 0.0285    hit@10 : 0.0916    precision@10 : 0.0102
  Test result:
  recall@10 : 0.047    mrr@10 : 0.0337    ndcg@10 : 0.0286    hit@10 : 0.0921    precision@10 : 0.0103
  
  learning_rate:0.0005, n_layers:3, reg_weight:0.002
  Valid result:
  recall@10 : 0.0415    mrr@10 : 0.0307    ndcg@10 : 0.0254    hit@10 : 0.0856    precision@10 : 0.0095
  Test result:
  recall@10 : 0.0406    mrr@10 : 0.03    ndcg@10 : 0.0248    hit@10 : 0.0832    precision@10 : 0.0093
  
  learning_rate:0.001, n_layers:2, reg_weight:0.0005
  Valid result:
  recall@10 : 0.0426    mrr@10 : 0.0316    ndcg@10 : 0.0262    hit@10 : 0.0879    precision@10 : 0.0099
  Test result:
  recall@10 : 0.042    mrr@10 : 0.0308    ndcg@10 : 0.0255    hit@10 : 0.087    precision@10 : 0.0099
  
  learning_rate:0.0005, n_layers:1, reg_weight:0.0005
  Valid result:
  recall@10 : 0.0385    mrr@10 : 0.0283    ndcg@10 : 0.0232    hit@10 : 0.0789    precision@10 : 0.0087
  Test result:
  recall@10 : 0.0375    mrr@10 : 0.028    ndcg@10 : 0.023    hit@10 : 0.0774    precision@10 : 0.0087
  
  learning_rate:0.005, n_layers:1, reg_weight:0.001
  Valid result:
  recall@10 : 0.0464    mrr@10 : 0.0337    ndcg@10 : 0.0284    hit@10 : 0.092    precision@10 : 0.0103
  Test result:
  recall@10 : 0.0462    mrr@10 : 0.0333    ndcg@10 : 0.0281    hit@10 : 0.0916    precision@10 : 0.0103
  
  learning_rate:0.0001, n_layers:2, reg_weight:0.002
  Valid result:
  recall@10 : 0.0094    mrr@10 : 0.0067    ndcg@10 : 0.0057    hit@10 : 0.019    precision@10 : 0.002
  Test result:
  recall@10 : 0.0092    mrr@10 : 0.0064    ndcg@10 : 0.0054    hit@10 : 0.0186    precision@10 : 0.0019
  
  learning_rate:0.005, n_layers:3, reg_weight:0.001
  Valid result:
  recall@10 : 0.0462    mrr@10 : 0.0345    ndcg@10 : 0.0287    hit@10 : 0.0933    precision@10 : 0.0105
  Test result:
  recall@10 : 0.0464    mrr@10 : 0.0351    ndcg@10 : 0.029    hit@10 : 0.0938    precision@10 : 0.0105
  
  learning_rate:0.01, n_layers:3, reg_weight:0.001
  Valid result:
  recall@10 : 0.039    mrr@10 : 0.0297    ndcg@10 : 0.0243    hit@10 : 0.0804    precision@10 : 0.0089
  Test result:
  recall@10 : 0.0382    mrr@10 : 0.0297    ndcg@10 : 0.024    hit@10 : 0.0797    precision@10 : 0.0089
  
  learning_rate:0.0001, n_layers:1, reg_weight:0.0005
  Valid result:
  recall@10 : 0.0094    mrr@10 : 0.0068    ndcg@10 : 0.0057    hit@10 : 0.0185    precision@10 : 0.0019
  Test result:
  recall@10 : 0.0099    mrr@10 : 0.0067    ndcg@10 : 0.0057    hit@10 : 0.0191    precision@10 : 0.002
  
  learning_rate:0.002, n_layers:1, reg_weight:0.002
  Valid result:
  recall@10 : 0.0549    mrr@10 : 0.0397    ndcg@10 : 0.0336    hit@10 : 0.1079    precision@10 : 0.0123
  Test result:
  recall@10 : 0.056    mrr@10 : 0.0406    ndcg@10 : 0.0345    hit@10 : 0.1087    precision@10 : 0.0125
  
  learning_rate:0.01, n_layers:1, reg_weight:0.001
  Valid result:
  recall@10 : 0.037    mrr@10 : 0.0272    ndcg@10 : 0.0226    hit@10 : 0.0749    precision@10 : 0.0082
  Test result:
  recall@10 : 0.0358    mrr@10 : 0.0269    ndcg@10 : 0.022    hit@10 : 0.0734    precision@10 : 0.0081
  
  learning_rate:0.002, n_layers:2, reg_weight:0.0005
  Valid result:
  recall@10 : 0.0557    mrr@10 : 0.0402    ndcg@10 : 0.0342    hit@10 : 0.1084    precision@10 : 0.0124
  Test result:
  recall@10 : 0.0558    mrr@10 : 0.0406    ndcg@10 : 0.0342    hit@10 : 0.11    precision@10 : 0.0127
  
  learning_rate:0.001, n_layers:3, reg_weight:0.001
  Valid result:
  recall@10 : 0.0393    mrr@10 : 0.0291    ndcg@10 : 0.024    hit@10 : 0.0807    precision@10 : 0.0089
  Test result:
  recall@10 : 0.0389    mrr@10 : 0.0286    ndcg@10 : 0.0238    hit@10 : 0.0805    precision@10 : 0.009
  
  learning_rate:0.001, n_layers:3, reg_weight:0.002
  Valid result:
  recall@10 : 0.0369    mrr@10 : 0.028    ndcg@10 : 0.0228    hit@10 : 0.0771    precision@10 : 0.0085
  Test result:
  recall@10 : 0.0369    mrr@10 : 0.0275    ndcg@10 : 0.0226    hit@10 : 0.076    precision@10 : 0.0084
  
  learning_rate:0.0005, n_layers:3, reg_weight:0.001
  Valid result:
  recall@10 : 0.0384    mrr@10 : 0.029    ndcg@10 : 0.0238    hit@10 : 0.0796    precision@10 : 0.0088
  Test result:
  recall@10 : 0.0389    mrr@10 : 0.0296    ndcg@10 : 0.0242    hit@10 : 0.0797    precision@10 : 0.0088
  
  learning_rate:0.001, n_layers:3, reg_weight:0.0005
  Valid result:
  recall@10 : 0.0373    mrr@10 : 0.028    ndcg@10 : 0.0229    hit@10 : 0.0776    precision@10 : 0.0086
  Test result:
  recall@10 : 0.037    mrr@10 : 0.0275    ndcg@10 : 0.0227    hit@10 : 0.076    precision@10 : 0.0084
  
  learning_rate:0.0001, n_layers:1, reg_weight:0.002
  Valid result:
  recall@10 : 0.0094    mrr@10 : 0.0068    ndcg@10 : 0.0057    hit@10 : 0.0185    precision@10 : 0.0019
  Test result:
  recall@10 : 0.0099    mrr@10 : 0.0067    ndcg@10 : 0.0057    hit@10 : 0.0191    precision@10 : 0.002
  
  learning_rate:0.0001, n_layers:2, reg_weight:0.0005
  Valid result:
  recall@10 : 0.0094    mrr@10 : 0.0067    ndcg@10 : 0.0057    hit@10 : 0.019    precision@10 : 0.002
  Test result:
  recall@10 : 0.0092    mrr@10 : 0.0064    ndcg@10 : 0.0054    hit@10 : 0.0186    precision@10 : 0.0019
  
  learning_rate:0.005, n_layers:2, reg_weight:0.0005
  Valid result:
  recall@10 : 0.0478    mrr@10 : 0.0337    ndcg@10 : 0.029    hit@10 : 0.0929    precision@10 : 0.0104
  Test result:
  recall@10 : 0.0472    mrr@10 : 0.0333    ndcg@10 : 0.0286    hit@10 : 0.0927    precision@10 : 0.0104
  
  learning_rate:0.0001, n_layers:3, reg_weight:0.0005
  Valid result:
  recall@10 : 0.0095    mrr@10 : 0.0069    ndcg@10 : 0.0057    hit@10 : 0.019    precision@10 : 0.002
  Test result:
  recall@10 : 0.0096    mrr@10 : 0.007    ndcg@10 : 0.0057    hit@10 : 0.019    precision@10 : 0.002
  
  learning_rate:0.002, n_layers:2, reg_weight:0.001
  Valid result:
  recall@10 : 0.0542    mrr@10 : 0.0392    ndcg@10 : 0.0333    hit@10 : 0.1074    precision@10 : 0.0123
  Test result:
  recall@10 : 0.0549    mrr@10 : 0.0405    ndcg@10 : 0.034    hit@10 : 0.1082    precision@10 : 0.0124
  
  learning_rate:0.0005, n_layers:2, reg_weight:0.002
  Valid result:
  recall@10 : 0.0366    mrr@10 : 0.0277    ndcg@10 : 0.0226    hit@10 : 0.0756    precision@10 : 0.0083
  Test result:
  recall@10 : 0.0358    mrr@10 : 0.0281    ndcg@10 : 0.0224    hit@10 : 0.075    precision@10 : 0.0083
  
  learning_rate:0.002, n_layers:3, reg_weight:0.002
  Valid result:
  recall@10 : 0.0549    mrr@10 : 0.0405    ndcg@10 : 0.0341    hit@10 : 0.1083    precision@10 : 0.0124
  Test result:
  recall@10 : 0.0544    mrr@10 : 0.0397    ndcg@10 : 0.0335    hit@10 : 0.1076    precision@10 : 0.0123
  
  learning_rate:0.005, n_layers:1, reg_weight:0.0005
  Valid result:
  recall@10 : 0.0465    mrr@10 : 0.0334    ndcg@10 : 0.0282    hit@10 : 0.0922    precision@10 : 0.0103
  Test result:
  recall@10 : 0.0467    mrr@10 : 0.0333    ndcg@10 : 0.0283    hit@10 : 0.0926    precision@10 : 0.0104
  
  learning_rate:0.005, n_layers:2, reg_weight:0.001
  Valid result:
  recall@10 : 0.0467    mrr@10 : 0.0337    ndcg@10 : 0.0285    hit@10 : 0.0925    precision@10 : 0.0103
  Test result:
  recall@10 : 0.0468    mrr@10 : 0.0342    ndcg@10 : 0.0286    hit@10 : 0.0926    precision@10 : 0.0104
  
  learning_rate:0.0005, n_layers:2, reg_weight:0.001
  Valid result:
  recall@10 : 0.0365    mrr@10 : 0.0282    ndcg@10 : 0.0228    hit@10 : 0.0763    precision@10 : 0.0085
  Test result:
  recall@10 : 0.0359    mrr@10 : 0.028    ndcg@10 : 0.0224    hit@10 : 0.0762    precision@10 : 0.0084
  
  learning_rate:0.0005, n_layers:2, reg_weight:0.0005
  Valid result:
  recall@10 : 0.0411    mrr@10 : 0.0312    ndcg@10 : 0.0254    hit@10 : 0.0855    precision@10 : 0.0095
  Test result:
  recall@10 : 0.0419    mrr@10 : 0.0314    ndcg@10 : 0.0259    hit@10 : 0.0858    precision@10 : 0.0096
  
  learning_rate:0.001, n_layers:1, reg_weight:0.002
  Valid result:
  recall@10 : 0.0355    mrr@10 : 0.0264    ndcg@10 : 0.0218    hit@10 : 0.0735    precision@10 : 0.0081
  Test result:
  recall@10 : 0.0355    mrr@10 : 0.0262    ndcg@10 : 0.0216    hit@10 : 0.0741    precision@10 : 0.0082
  
  learning_rate:0.002, n_layers:3, reg_weight:0.001
  Valid result:
  recall@10 : 0.0586    mrr@10 : 0.0423    ndcg@10 : 0.0359    hit@10 : 0.1139    precision@10 : 0.0131
  Test result:
  recall@10 : 0.0572    mrr@10 : 0.0415    ndcg@10 : 0.0351    hit@10 : 0.1122    precision@10 : 0.0129
  
  learning_rate:0.01, n_layers:1, reg_weight:0.002
  Valid result:
  recall@10 : 0.0358    mrr@10 : 0.0266    ndcg@10 : 0.0222    hit@10 : 0.0733    precision@10 : 0.008
  Test result:
  recall@10 : 0.0365    mrr@10 : 0.027    ndcg@10 : 0.0223    hit@10 : 0.0744    precision@10 : 0.0081
  
  learning_rate:0.002, n_layers:1, reg_weight:0.0005
  Valid result:
  recall@10 : 0.0539    mrr@10 : 0.0398    ndcg@10 : 0.0334    hit@10 : 0.1062    precision@10 : 0.0121
  Test result:
  recall@10 : 0.0549    mrr@10 : 0.0402    ndcg@10 : 0.034    hit@10 : 0.1079    precision@10 : 0.0124
  
  learning_rate:0.01, n_layers:2, reg_weight:0.001
  Valid result:
  recall@10 : 0.0421    mrr@10 : 0.0322    ndcg@10 : 0.0263    hit@10 : 0.0861    precision@10 : 0.0096
  Test result:
  recall@10 : 0.0431    mrr@10 : 0.0326    ndcg@10 : 0.0269    hit@10 : 0.087    precision@10 : 0.0098
  
  learning_rate:0.005, n_layers:3, reg_weight:0.002
  Valid result:
  recall@10 : 0.0468    mrr@10 : 0.0341    ndcg@10 : 0.0287    hit@10 : 0.092    precision@10 : 0.0103
  Test result:
  recall@10 : 0.0466    mrr@10 : 0.0339    ndcg@10 : 0.0284    hit@10 : 0.0932    precision@10 : 0.0105
  
  learning_rate:0.002, n_layers:3, reg_weight:0.0005
  Valid result:
  recall@10 : 0.0595    mrr@10 : 0.0429    ndcg@10 : 0.0367    hit@10 : 0.1148    precision@10 : 0.0132
  Test result:
  recall@10 : 0.0587    mrr@10 : 0.0425    ndcg@10 : 0.036    hit@10 : 0.1145    precision@10 : 0.0132
  
  learning_rate:0.0001, n_layers:3, reg_weight:0.002
  Valid result:
  recall@10 : 0.0095    mrr@10 : 0.0069    ndcg@10 : 0.0057    hit@10 : 0.019    precision@10 : 0.002
  Test result:
  recall@10 : 0.0096    mrr@10 : 0.007    ndcg@10 : 0.0057    hit@10 : 0.019    precision@10 : 0.002
  
  learning_rate:0.01, n_layers:3, reg_weight:0.0005
  Valid result:
  recall@10 : 0.0364    mrr@10 : 0.0281    ndcg@10 : 0.0228    hit@10 : 0.0771    precision@10 : 0.0085
  Test result:
  recall@10 : 0.0364    mrr@10 : 0.0272    ndcg@10 : 0.0224    hit@10 : 0.075    precision@10 : 0.0083
  
  learning_rate:0.0001, n_layers:1, reg_weight:0.001
  Valid result:
  recall@10 : 0.0094    mrr@10 : 0.0068    ndcg@10 : 0.0057    hit@10 : 0.0185    precision@10 : 0.0019
  Test result:
  recall@10 : 0.0099    mrr@10 : 0.0067    ndcg@10 : 0.0057    hit@10 : 0.0191    precision@10 : 0.002
  
  learning_rate:0.001, n_layers:1, reg_weight:0.0005
  Valid result:
  recall@10 : 0.0354    mrr@10 : 0.0263    ndcg@10 : 0.0218    hit@10 : 0.0732    precision@10 : 0.0081
  Test result:
  recall@10 : 0.0356    mrr@10 : 0.0261    ndcg@10 : 0.0216    hit@10 : 0.0741    precision@10 : 0.0082
  
  learning_rate:0.01, n_layers:2, reg_weight:0.0005
  Valid result:
  recall@10 : 0.0377    mrr@10 : 0.0287    ndcg@10 : 0.0235    hit@10 : 0.077    precision@10 : 0.0085
  Test result:
  recall@10 : 0.0378    mrr@10 : 0.0282    ndcg@10 : 0.0233    hit@10 : 0.0766    precision@10 : 0.0084
  
  learning_rate:0.005, n_layers:1, reg_weight:0.002
  Valid result:
  recall@10 : 0.0477    mrr@10 : 0.0349    ndcg@10 : 0.0295    hit@10 : 0.0946    precision@10 : 0.0107
  Test result:
  recall@10 : 0.0486    mrr@10 : 0.0358    ndcg@10 : 0.0299    hit@10 : 0.0966    precision@10 : 0.0108
  
  learning_rate:0.005, n_layers:3, reg_weight:0.0005
  Valid result:
  recall@10 : 0.0465    mrr@10 : 0.0341    ndcg@10 : 0.0285    hit@10 : 0.0931    precision@10 : 0.0105
  Test result:
  recall@10 : 0.046    mrr@10 : 0.0339    ndcg@10 : 0.0283    hit@10 : 0.0919    precision@10 : 0.0103
  
  learning_rate:0.0005, n_layers:1, reg_weight:0.001
  Valid result:
  recall@10 : 0.0383    mrr@10 : 0.0283    ndcg@10 : 0.0232    hit@10 : 0.0788    precision@10 : 0.0087
  Test result:
  recall@10 : 0.0376    mrr@10 : 0.0281    ndcg@10 : 0.023    hit@10 : 0.0776    precision@10 : 0.0087
  
  learning_rate:0.0001, n_layers:3, reg_weight:0.001
  Valid result:
  recall@10 : 0.0095    mrr@10 : 0.0069    ndcg@10 : 0.0057    hit@10 : 0.019    precision@10 : 0.002
  Test result:
  recall@10 : 0.0096    mrr@10 : 0.007    ndcg@10 : 0.0057    hit@10 : 0.019    precision@10 : 0.002
  
  learning_rate:0.0005, n_layers:3, reg_weight:0.0005
  Valid result:
  recall@10 : 0.0385    mrr@10 : 0.029    ndcg@10 : 0.0238    hit@10 : 0.0796    precision@10 : 0.0088
  Test result:
  recall@10 : 0.0386    mrr@10 : 0.0287    ndcg@10 : 0.0236    hit@10 : 0.0799    precision@10 : 0.0089
  
  learning_rate:0.0001, n_layers:2, reg_weight:0.001
  Valid result:
  recall@10 : 0.0094    mrr@10 : 0.0067    ndcg@10 : 0.0057    hit@10 : 0.019    precision@10 : 0.002
  Test result:
  recall@10 : 0.0092    mrr@10 : 0.0064    ndcg@10 : 0.0054    hit@10 : 0.0186    precision@10 : 0.0019
  
  learning_rate:0.002, n_layers:1, reg_weight:0.001
  Valid result:
  recall@10 : 0.0541    mrr@10 : 0.0397    ndcg@10 : 0.0335    hit@10 : 0.1065    precision@10 : 0.0121
  Test result:
  recall@10 : 0.0551    mrr@10 : 0.0403    ndcg@10 : 0.034    hit@10 : 0.1083    precision@10 : 0.0124
  
  learning_rate:0.001, n_layers:2, reg_weight:0.002
  Valid result:
  recall@10 : 0.0419    mrr@10 : 0.0312    ndcg@10 : 0.0257    hit@10 : 0.0867    precision@10 : 0.0097
  Test result:
  recall@10 : 0.0418    mrr@10 : 0.0311    ndcg@10 : 0.0257    hit@10 : 0.0861    precision@10 : 0.0098
  
  learning_rate:0.001, n_layers:2, reg_weight:0.001
  Valid result:
  recall@10 : 0.0478    mrr@10 : 0.0355    ndcg@10 : 0.0295    hit@10 : 0.0968    precision@10 : 0.011
  Test result:
  recall@10 : 0.0487    mrr@10 : 0.0356    ndcg@10 : 0.0297    hit@10 : 0.0984    precision@10 : 0.0113
  ```
  
- **Logging Result**:

  ```yaml
  100%|██████████| 54/54 [152:31:04<00:00, 10167.87s/trial, best loss: -0.0367]
  best params:  {'learning_rate': 0.002, 'n_layers': 3, 'reg_weight': 0.0005}
  best result: 
  {'model': 'SpectralCF', 'best_valid_score': 0.0367, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.0595), ('mrr@10', 0.0429), ('ndcg@10', 0.0367), ('hit@10', 0.1148), ('precision@10', 0.0132)]), 'test_result': OrderedDict([('recall@10', 0.0587), ('mrr@10', 0.0425), ('ndcg@10', 0.036), ('hit@10', 0.1145), ('precision@10', 0.0132)])}
  ```
