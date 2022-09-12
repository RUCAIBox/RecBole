# General Recommendation

- **Dataset**: [Yelp2022](../../md/yelp_general.md)

- **Model**: [SimpleX](https://recbole.io/docs/user_guide/model/general/simplex.html)

- **Time cost**: 3031.18s/trial

- **Hyper-parameter searching** (hyper.test):s

  ```yaml
  learning_rate choice [1e-4,1e-3]
  gamma choice [0.3,0.5,0.7]
  margin choice [0,0.5,0.9]
  negative_weight choice [1,10,50]
  ```
  
- **Best parameters**:

  ```yaml
  learning_rate: 0.001
  gamma: 0.7
  margin: 0.5
  negative_weight: 50
  ```
  
- **Hyper-parameter logging**:

  ```yaml
  gamma:0.5, learning_rate:0.0001, margin:0, negative_weight:1
  Valid result:
  recall@10 : 0.0262    mrr@10 : 0.0194    ndcg@10 : 0.0159    hit@10 : 0.0556    precision@10 : 0.0061
  Test result:
  recall@10 : 0.0265    mrr@10 : 0.0196    ndcg@10 : 0.016    hit@10 : 0.0561    precision@10 : 0.0062
  
  gamma:0.3, learning_rate:0.0001, margin:0.9, negative_weight:1
  Valid result:
  recall@10 : 0.0279    mrr@10 : 0.0198    ndcg@10 : 0.0166    hit@10 : 0.0588    precision@10 : 0.0064
  Test result:
  recall@10 : 0.0292    mrr@10 : 0.0201    ndcg@10 : 0.0169    hit@10 : 0.0604    precision@10 : 0.0066
  
  gamma:0.7, learning_rate:0.001, margin:0.5, negative_weight:10
  Valid result:
  recall@10 : 0.0545    mrr@10 : 0.0405    ndcg@10 : 0.034    hit@10 : 0.1075    precision@10 : 0.0122
  Test result:
  recall@10 : 0.0548    mrr@10 : 0.0402    ndcg@10 : 0.0339    hit@10 : 0.1068    precision@10 : 0.0122
  
  gamma:0.7, learning_rate:0.001, margin:0.5, negative_weight:1
  Valid result:
  recall@10 : 0.0205    mrr@10 : 0.0155    ndcg@10 : 0.0124    hit@10 : 0.045    precision@10 : 0.0049
  Test result:
  recall@10 : 0.0208    mrr@10 : 0.0158    ndcg@10 : 0.0125    hit@10 : 0.0462    precision@10 : 0.005
  
  gamma:0.7, learning_rate:0.0001, margin:0.5, negative_weight:10
  Valid result:
  recall@10 : 0.0526    mrr@10 : 0.0382    ndcg@10 : 0.0325    hit@10 : 0.1032    precision@10 : 0.0116
  Test result:
  recall@10 : 0.0525    mrr@10 : 0.0388    ndcg@10 : 0.0325    hit@10 : 0.1035    precision@10 : 0.0117
  
  gamma:0.3, learning_rate:0.001, margin:0.9, negative_weight:50
  Valid result:
  recall@10 : 0.062    mrr@10 : 0.0462    ndcg@10 : 0.0389    hit@10 : 0.1204    precision@10 : 0.0137
  Test result:
  recall@10 : 0.0632    mrr@10 : 0.0467    ndcg@10 : 0.0393    hit@10 : 0.1224    precision@10 : 0.014
  
  gamma:0.5, learning_rate:0.001, margin:0.9, negative_weight:50
  Valid result:
  recall@10 : 0.0588    mrr@10 : 0.0435    ndcg@10 : 0.0365    hit@10 : 0.1149    precision@10 : 0.013
  Test result:
  recall@10 : 0.0595    mrr@10 : 0.0435    ndcg@10 : 0.0369    hit@10 : 0.1152    precision@10 : 0.013
  
  gamma:0.5, learning_rate:0.001, margin:0, negative_weight:50
  Valid result:
  recall@10 : 0.0845    mrr@10 : 0.063    ndcg@10 : 0.0537    hit@10 : 0.1546    precision@10 : 0.018
  Test result:
  recall@10 : 0.0857    mrr@10 : 0.0634    ndcg@10 : 0.0542    hit@10 : 0.156    precision@10 : 0.0182
  
  gamma:0.3, learning_rate:0.0001, margin:0, negative_weight:10
  Valid result:
  recall@10 : 0.0554    mrr@10 : 0.0402    ndcg@10 : 0.0339    hit@10 : 0.1071    precision@10 : 0.012
  Test result:
  recall@10 : 0.0556    mrr@10 : 0.0406    ndcg@10 : 0.0341    hit@10 : 0.1083    precision@10 : 0.0122
  
  gamma:0.7, learning_rate:0.0001, margin:0.9, negative_weight:10
  Valid result:
  recall@10 : 0.047    mrr@10 : 0.0331    ndcg@10 : 0.0283    hit@10 : 0.0934    precision@10 : 0.0103
  Test result:
  recall@10 : 0.0469    mrr@10 : 0.0331    ndcg@10 : 0.0283    hit@10 : 0.0935    precision@10 : 0.0105
  
  gamma:0.7, learning_rate:0.001, margin:0, negative_weight:1
  Valid result:
  recall@10 : 0.0181    mrr@10 : 0.0139    ndcg@10 : 0.0111    hit@10 : 0.0413    precision@10 : 0.0044
  Test result:
  recall@10 : 0.0186    mrr@10 : 0.0142    ndcg@10 : 0.0113    hit@10 : 0.0424    precision@10 : 0.0046
  
  gamma:0.7, learning_rate:0.0001, margin:0.9, negative_weight:1
  Valid result:
  recall@10 : 0.027    mrr@10 : 0.0191    ndcg@10 : 0.016    hit@10 : 0.0561    precision@10 : 0.006
  Test result:
  recall@10 : 0.0269    mrr@10 : 0.0188    ndcg@10 : 0.0158    hit@10 : 0.056    precision@10 : 0.006
  
  gamma:0.7, learning_rate:0.001, margin:0, negative_weight:10
  Valid result:
  recall@10 : 0.0582    mrr@10 : 0.042    ndcg@10 : 0.0357    hit@10 : 0.1128    precision@10 : 0.0127
  Test result:
  recall@10 : 0.0585    mrr@10 : 0.0425    ndcg@10 : 0.036    hit@10 : 0.1129    precision@10 : 0.0128
  
  gamma:0.5, learning_rate:0.001, margin:0, negative_weight:10
  Valid result:
  recall@10 : 0.058    mrr@10 : 0.0418    ndcg@10 : 0.0356    hit@10 : 0.1119    precision@10 : 0.0126
  Test result:
  recall@10 : 0.0577    mrr@10 : 0.0417    ndcg@10 : 0.0352    hit@10 : 0.1122    precision@10 : 0.0127
  
  gamma:0.7, learning_rate:0.0001, margin:0.9, negative_weight:50
  Valid result:
  recall@10 : 0.0518    mrr@10 : 0.0384    ndcg@10 : 0.032    hit@10 : 0.1023    precision@10 : 0.0114
  Test result:
  recall@10 : 0.0527    mrr@10 : 0.0381    ndcg@10 : 0.0323    hit@10 : 0.1032    precision@10 : 0.0116
  
  gamma:0.5, learning_rate:0.0001, margin:0, negative_weight:50
  Valid result:
  recall@10 : 0.0767    mrr@10 : 0.0579    ndcg@10 : 0.049    hit@10 : 0.1425    precision@10 : 0.0166
  Test result:
  recall@10 : 0.0779    mrr@10 : 0.0596    ndcg@10 : 0.0503    hit@10 : 0.1433    precision@10 : 0.0167
  
  gamma:0.3, learning_rate:0.001, margin:0, negative_weight:10
  Valid result:
  recall@10 : 0.0535    mrr@10 : 0.0383    ndcg@10 : 0.0326    hit@10 : 0.1049    precision@10 : 0.0117
  Test result:
  recall@10 : 0.0538    mrr@10 : 0.0393    ndcg@10 : 0.033    hit@10 : 0.105    precision@10 : 0.0118
  
  gamma:0.5, learning_rate:0.0001, margin:0.9, negative_weight:50
  Valid result:
  recall@10 : 0.0626    mrr@10 : 0.0449    ndcg@10 : 0.0384    hit@10 : 0.1203    precision@10 : 0.0137
  Test result:
  recall@10 : 0.062    mrr@10 : 0.0455    ndcg@10 : 0.0384    hit@10 : 0.1208    precision@10 : 0.0138
  
  gamma:0.5, learning_rate:0.0001, margin:0.9, negative_weight:1
  Valid result:
  recall@10 : 0.0263    mrr@10 : 0.0192    ndcg@10 : 0.0158    hit@10 : 0.0561    precision@10 : 0.0061
  Test result:
  recall@10 : 0.0264    mrr@10 : 0.0194    ndcg@10 : 0.0159    hit@10 : 0.056    precision@10 : 0.0061
  
  gamma:0.5, learning_rate:0.0001, margin:0.9, negative_weight:10
  Valid result:
  recall@10 : 0.0431    mrr@10 : 0.0298    ndcg@10 : 0.0255    hit@10 : 0.0852    precision@10 : 0.0094
  Test result:
  recall@10 : 0.0416    mrr@10 : 0.0298    ndcg@10 : 0.0251    hit@10 : 0.0842    precision@10 : 0.0093
  
  gamma:0.5, learning_rate:0.0001, margin:0, negative_weight:10
  Valid result:
  recall@10 : 0.0578    mrr@10 : 0.0415    ndcg@10 : 0.0352    hit@10 : 0.1115    precision@10 : 0.0126
  Test result:
  recall@10 : 0.0573    mrr@10 : 0.0415    ndcg@10 : 0.0352    hit@10 : 0.1111    precision@10 : 0.0126
  
  gamma:0.7, learning_rate:0.0001, margin:0, negative_weight:1
  Valid result:
  recall@10 : 0.0248    mrr@10 : 0.0192    ndcg@10 : 0.0154    hit@10 : 0.0534    precision@10 : 0.0059
  Test result:
  recall@10 : 0.024    mrr@10 : 0.0192    ndcg@10 : 0.0152    hit@10 : 0.0539    precision@10 : 0.006
  
  gamma:0.3, learning_rate:0.001, margin:0.5, negative_weight:10
  Valid result:
  recall@10 : 0.0514    mrr@10 : 0.0363    ndcg@10 : 0.0311    hit@10 : 0.1007    precision@10 : 0.0112
  Test result:
  recall@10 : 0.0515    mrr@10 : 0.037    ndcg@10 : 0.0314    hit@10 : 0.101    precision@10 : 0.0114
  
  gamma:0.7, learning_rate:0.001, margin:0.5, negative_weight:50
  Valid result:
  recall@10 : 0.0958    mrr@10 : 0.0764    ndcg@10 : 0.0636    hit@10 : 0.1752    precision@10 : 0.0211
  Test result:
  recall@10 : 0.0964    mrr@10 : 0.077    ndcg@10 : 0.064    hit@10 : 0.1777    precision@10 : 0.0216
  
  gamma:0.3, learning_rate:0.0001, margin:0.5, negative_weight:1
  Valid result:
  recall@10 : 0.0292    mrr@10 : 0.0205    ndcg@10 : 0.0173    hit@10 : 0.0609    precision@10 : 0.0066
  Test result:
  recall@10 : 0.0301    mrr@10 : 0.0209    ndcg@10 : 0.0176    hit@10 : 0.0615    precision@10 : 0.0068
  
  gamma:0.5, learning_rate:0.001, margin:0, negative_weight:1
  Valid result:
  recall@10 : 0.0174    mrr@10 : 0.0139    ndcg@10 : 0.0107    hit@10 : 0.0407    precision@10 : 0.0044
  Test result:
  recall@10 : 0.017    mrr@10 : 0.0128    ndcg@10 : 0.01    hit@10 : 0.0399    precision@10 : 0.0043
  
  gamma:0.7, learning_rate:0.0001, margin:0, negative_weight:50
  Valid result:
  recall@10 : 0.0782    mrr@10 : 0.0606    ndcg@10 : 0.0506    hit@10 : 0.145    precision@10 : 0.0169
  Test result:
  recall@10 : 0.0799    mrr@10 : 0.0615    ndcg@10 : 0.0516    hit@10 : 0.1472    precision@10 : 0.0173
  
  gamma:0.3, learning_rate:0.0001, margin:0, negative_weight:1
  Valid result:
  recall@10 : 0.0276    mrr@10 : 0.0199    ndcg@10 : 0.0166    hit@10 : 0.0582    precision@10 : 0.0063
  Test result:
  recall@10 : 0.0287    mrr@10 : 0.0204    ndcg@10 : 0.017    hit@10 : 0.0604    precision@10 : 0.0066
  
  gamma:0.5, learning_rate:0.001, margin:0.9, negative_weight:10
  Valid result:
  recall@10 : 0.0457    mrr@10 : 0.0329    ndcg@10 : 0.0279    hit@10 : 0.0907    precision@10 : 0.0101
  Test result:
  recall@10 : 0.0453    mrr@10 : 0.0327    ndcg@10 : 0.0273    hit@10 : 0.0917    precision@10 : 0.0102
  
  gamma:0.3, learning_rate:0.001, margin:0.5, negative_weight:50
  Valid result:
  recall@10 : 0.0937    mrr@10 : 0.0737    ndcg@10 : 0.0616    hit@10 : 0.172    precision@10 : 0.0207
  Test result:
  recall@10 : 0.096    mrr@10 : 0.0749    ndcg@10 : 0.0629    hit@10 : 0.1747    precision@10 : 0.0211
  
  gamma:0.3, learning_rate:0.001, margin:0.9, negative_weight:1
  Valid result:
  recall@10 : 0.0263    mrr@10 : 0.02    ndcg@10 : 0.016    hit@10 : 0.0561    precision@10 : 0.006
  Test result:
  recall@10 : 0.0266    mrr@10 : 0.0205    ndcg@10 : 0.0163    hit@10 : 0.056    precision@10 : 0.006
  
  gamma:0.7, learning_rate:0.0001, margin:0.5, negative_weight:1
  Valid result:
  recall@10 : 0.0267    mrr@10 : 0.0199    ndcg@10 : 0.0163    hit@10 : 0.0568    precision@10 : 0.0063
  Test result:
  recall@10 : 0.0267    mrr@10 : 0.0206    ndcg@10 : 0.0166    hit@10 : 0.058    precision@10 : 0.0064
  
  gamma:0.7, learning_rate:0.001, margin:0.9, negative_weight:50
  Valid result:
  recall@10 : 0.0623    mrr@10 : 0.0449    ndcg@10 : 0.0382    hit@10 : 0.1198    precision@10 : 0.0135
  Test result:
  recall@10 : 0.0619    mrr@10 : 0.0452    ndcg@10 : 0.0384    hit@10 : 0.1196    precision@10 : 0.0137
  
  gamma:0.3, learning_rate:0.0001, margin:0.9, negative_weight:50
  Valid result:
  recall@10 : 0.0621    mrr@10 : 0.0453    ndcg@10 : 0.0386    hit@10 : 0.12    precision@10 : 0.0136
  Test result:
  recall@10 : 0.0608    mrr@10 : 0.0446    ndcg@10 : 0.0378    hit@10 : 0.1166    precision@10 : 0.0134
  
  gamma:0.3, learning_rate:0.001, margin:0, negative_weight:50
  Valid result:
  recall@10 : 0.0812    mrr@10 : 0.0592    ndcg@10 : 0.0512    hit@10 : 0.1493    precision@10 : 0.0173
  Test result:
  recall@10 : 0.0815    mrr@10 : 0.0601    ndcg@10 : 0.0515    hit@10 : 0.1494    precision@10 : 0.0173
  
  gamma:0.7, learning_rate:0.0001, margin:0.5, negative_weight:50
  Valid result:
  recall@10 : 0.0886    mrr@10 : 0.0688    ndcg@10 : 0.0576    hit@10 : 0.1637    precision@10 : 0.0197
  Test result:
  recall@10 : 0.0908    mrr@10 : 0.0708    ndcg@10 : 0.0592    hit@10 : 0.1679    precision@10 : 0.0203
  
  gamma:0.7, learning_rate:0.0001, margin:0, negative_weight:10
  Valid result:
  recall@10 : 0.0568    mrr@10 : 0.0411    ndcg@10 : 0.0348    hit@10 : 0.1105    precision@10 : 0.0125
  Test result:
  recall@10 : 0.0576    mrr@10 : 0.0421    ndcg@10 : 0.0354    hit@10 : 0.1116    precision@10 : 0.0127
  
  gamma:0.5, learning_rate:0.001, margin:0.9, negative_weight:1
  Valid result:
  recall@10 : 0.0256    mrr@10 : 0.0194    ndcg@10 : 0.0156    hit@10 : 0.055    precision@10 : 0.0059
  Test result:
  recall@10 : 0.0257    mrr@10 : 0.0196    ndcg@10 : 0.0157    hit@10 : 0.0545    precision@10 : 0.0059
  
  gamma:0.5, learning_rate:0.0001, margin:0.5, negative_weight:10
  Valid result:
  recall@10 : 0.0551    mrr@10 : 0.0404    ndcg@10 : 0.0341    hit@10 : 0.1076    precision@10 : 0.0122
  Test result:
  recall@10 : 0.0547    mrr@10 : 0.0399    ndcg@10 : 0.0338    hit@10 : 0.107    precision@10 : 0.0122
  
  gamma:0.5, learning_rate:0.0001, margin:0.5, negative_weight:1
  Valid result:
  recall@10 : 0.0273    mrr@10 : 0.0198    ndcg@10 : 0.0164    hit@10 : 0.0578    precision@10 : 0.0063
  Test result:
  recall@10 : 0.0273    mrr@10 : 0.02    ndcg@10 : 0.0164    hit@10 : 0.0575    precision@10 : 0.0063
  
  gamma:0.3, learning_rate:0.0001, margin:0.9, negative_weight:10
  Valid result:
  recall@10 : 0.0428    mrr@10 : 0.0303    ndcg@10 : 0.0256    hit@10 : 0.0863    precision@10 : 0.0095
  Test result:
  recall@10 : 0.0423    mrr@10 : 0.0302    ndcg@10 : 0.0255    hit@10 : 0.0847    precision@10 : 0.0094
  
  gamma:0.5, learning_rate:0.001, margin:0.5, negative_weight:1
  Valid result:
  recall@10 : 0.0199    mrr@10 : 0.0154    ndcg@10 : 0.0123    hit@10 : 0.0434    precision@10 : 0.0047
  Test result:
  recall@10 : 0.0194    mrr@10 : 0.015    ndcg@10 : 0.0118    hit@10 : 0.0431    precision@10 : 0.0046
  
  gamma:0.7, learning_rate:0.001, margin:0.9, negative_weight:1
  Valid result:
  recall@10 : 0.0264    mrr@10 : 0.0204    ndcg@10 : 0.0162    hit@10 : 0.0568    precision@10 : 0.0062
  Test result:
  recall@10 : 0.0274    mrr@10 : 0.0207    ndcg@10 : 0.0167    hit@10 : 0.0579    precision@10 : 0.0063
  
  gamma:0.5, learning_rate:0.0001, margin:0.5, negative_weight:50
  Valid result:
  recall@10 : 0.0896    mrr@10 : 0.0689    ndcg@10 : 0.058    hit@10 : 0.1652    precision@10 : 0.0198
  Test result:
  recall@10 : 0.0907    mrr@10 : 0.0703    ndcg@10 : 0.0591    hit@10 : 0.1671    precision@10 : 0.0202
  
  gamma:0.3, learning_rate:0.001, margin:0, negative_weight:1
  Valid result:
  recall@10 : 0.0181    mrr@10 : 0.014    ndcg@10 : 0.011    hit@10 : 0.042    precision@10 : 0.0045
  Test result:
  recall@10 : 0.0174    mrr@10 : 0.0135    ndcg@10 : 0.0106    hit@10 : 0.0402    precision@10 : 0.0043
  
  gamma:0.3, learning_rate:0.0001, margin:0, negative_weight:50
  Valid result:
  recall@10 : 0.0776    mrr@10 : 0.058    ndcg@10 : 0.0494    hit@10 : 0.1438    precision@10 : 0.0167
  Test result:
  recall@10 : 0.0793    mrr@10 : 0.0594    ndcg@10 : 0.0505    hit@10 : 0.1452    precision@10 : 0.0169
  
  gamma:0.3, learning_rate:0.001, margin:0.5, negative_weight:1
  Valid result:
  recall@10 : 0.0204    mrr@10 : 0.0153    ndcg@10 : 0.0124    hit@10 : 0.0446    precision@10 : 0.0047
  Test result:
  recall@10 : 0.02    mrr@10 : 0.0154    ndcg@10 : 0.0121    hit@10 : 0.0438    precision@10 : 0.0047
  
  gamma:0.7, learning_rate:0.001, margin:0.9, negative_weight:10
  Valid result:
  recall@10 : 0.0467    mrr@10 : 0.0334    ndcg@10 : 0.0282    hit@10 : 0.0934    precision@10 : 0.0103
  Test result:
  recall@10 : 0.0469    mrr@10 : 0.034    ndcg@10 : 0.0286    hit@10 : 0.0941    precision@10 : 0.0105
  
  gamma:0.7, learning_rate:0.001, margin:0, negative_weight:50
  Valid result:
  recall@10 : 0.0799    mrr@10 : 0.0598    ndcg@10 : 0.0511    hit@10 : 0.148    precision@10 : 0.0173
  Test result:
  recall@10 : 0.0813    mrr@10 : 0.0614    ndcg@10 : 0.052    hit@10 : 0.1488    precision@10 : 0.0174
  
  gamma:0.3, learning_rate:0.0001, margin:0.5, negative_weight:50
  Valid result:
  recall@10 : 0.0884    mrr@10 : 0.0679    ndcg@10 : 0.0571    hit@10 : 0.1639    precision@10 : 0.0195
  Test result:
  recall@10 : 0.0884    mrr@10 : 0.068    ndcg@10 : 0.0571    hit@10 : 0.1643    precision@10 : 0.0198
  
  gamma:0.5, learning_rate:0.001, margin:0.5, negative_weight:50
  Valid result:
  recall@10 : 0.0865    mrr@10 : 0.066    ndcg@10 : 0.0558    hit@10 : 0.1598    precision@10 : 0.019
  Test result:
  recall@10 : 0.0864    mrr@10 : 0.0664    ndcg@10 : 0.056    hit@10 : 0.1596    precision@10 : 0.0192
  
  gamma:0.3, learning_rate:0.0001, margin:0.5, negative_weight:10
  Valid result:
  recall@10 : 0.0534    mrr@10 : 0.0383    ndcg@10 : 0.0326    hit@10 : 0.1048    precision@10 : 0.0118
  Test result:
  recall@10 : 0.0536    mrr@10 : 0.0389    ndcg@10 : 0.0329    hit@10 : 0.1046    precision@10 : 0.0118
  
  gamma:0.3, learning_rate:0.001, margin:0.9, negative_weight:10
  Valid result:
  recall@10 : 0.0471    mrr@10 : 0.0334    ndcg@10 : 0.0283    hit@10 : 0.094    precision@10 : 0.0104
  Test result:
  recall@10 : 0.0465    mrr@10 : 0.033    ndcg@10 : 0.028    hit@10 : 0.0922    precision@10 : 0.0103
  
  gamma:0.5, learning_rate:0.001, margin:0.5, negative_weight:10
  Valid result:
  recall@10 : 0.0529    mrr@10 : 0.0386    ndcg@10 : 0.0325    hit@10 : 0.1045    precision@10 : 0.0118
  Test result:
  recall@10 : 0.0531    mrr@10 : 0.0387    ndcg@10 : 0.0327    hit@10 : 0.1038    precision@10 : 0.0118
  ```
  
- **Logging Result**:

  ```yaml
  100%|██████████| 54/54 [56:15:19<00:00, 3750.36s/trial, best loss: -0.0636]
  best params:  {'gamma': 0.7, 'learning_rate': 0.001, 'margin': 0.5, 'negative_weight': 50}
  best result: 
  {'model': 'SimpleX', 'best_valid_score': 0.0636, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.0958), ('mrr@10', 0.0764), ('ndcg@10', 0.0636), ('hit@10', 0.1752), ('precision@10', 0.0211)]), 'test_result': OrderedDict([('recall@10', 0.0964), ('mrr@10', 0.077), ('ndcg@10', 0.064), ('hit@10', 0.1777), ('precision@10', 0.0216)])}
  ```
