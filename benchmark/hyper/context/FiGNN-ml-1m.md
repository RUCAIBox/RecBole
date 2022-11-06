# Context-aware Recommendation

- **Dataset**: [ml-1m](../../md/ml-1m_contxt.md)

- **Model**: [FiGNN](https://recbole.io/docs/user_guide/model/context/fignn.html)

- **Hyper-parameter searching** (hyper.test):

```yaml
  learning_rate choice [0.005,0.001,0.0005]
  attention_size choice [8,16,32]
  n_layers choice [2,3,4]
```

- **Best parameters**:

```
  attention_size: 32
  learning_rate: 0.005
  n_layers: 2
```

- **Hyper-parameter logging** (hyper.result):

```yaml
  attention_size:8, learning_rate:0.0005, n_layers:4
  Valid result:
  auc : 0.8921    logloss : 0.3389
  Test result:
  auc : 0.891    logloss : 0.3419

  attention_size:8, learning_rate:0.001, n_layers:4
  Valid result:
  auc : 0.8937    logloss : 0.3337
  Test result:
  auc : 0.8927    logloss : 0.3351

  attention_size:8, learning_rate:0.001, n_layers:3
  Valid result:
  auc : 0.8911    logloss : 0.3366
  Test result:
  auc : 0.8895    logloss : 0.3392

  attention_size:8, learning_rate:0.0005, n_layers:3
  Valid result:
  auc : 0.8898    logloss : 0.3433
  Test result:
  auc : 0.8889    logloss : 0.345

  attention_size:16, learning_rate:0.005, n_layers:3
  Valid result:
  auc : 0.8962    logloss : 0.323
  Test result:
  auc : 0.895    logloss : 0.3256

  attention_size:16, learning_rate:0.0005, n_layers:4
  Valid result:
  auc : 0.8913    logloss : 0.332
  Test result:
  auc : 0.8899    logloss : 0.3342

  attention_size:32, learning_rate:0.001, n_layers:2
  Valid result:
  auc : 0.8937    logloss : 0.3296
  Test result:
  auc : 0.8925    logloss : 0.3323

  attention_size:32, learning_rate:0.001, n_layers:4
  Valid result:
  auc : 0.8952    logloss : 0.325
  Test result:
  auc : 0.894    logloss : 0.3269

  attention_size:16, learning_rate:0.001, n_layers:3
  Valid result:
  auc : 0.8951    logloss : 0.3307
  Test result:
  auc : 0.8922    logloss : 0.3359

  attention_size:8, learning_rate:0.0005, n_layers:2
  Valid result:
  auc : 0.8934    logloss : 0.3292
  Test result:
  auc : 0.8913    logloss : 0.3326

  attention_size:32, learning_rate:0.005, n_layers:4
  Valid result:
  auc : 0.8968    logloss : 0.3239
  Test result:
  auc : 0.8952    logloss : 0.3276

  attention_size:16, learning_rate:0.0005, n_layers:2
  Valid result:
  auc : 0.8926    logloss : 0.3345
  Test result:
  auc : 0.8915    logloss : 0.3372

  attention_size:32, learning_rate:0.001, n_layers:3
  Valid result:
  auc : 0.8957    logloss : 0.3264
  Test result:
  auc : 0.8939    logloss : 0.3306

  attention_size:8, learning_rate:0.005, n_layers:4
  Valid result:
  auc : 0.8951    logloss : 0.3232
  Test result:
  auc : 0.893    logloss : 0.3272

  attention_size:32, learning_rate:0.005, n_layers:2
  Valid result:
  auc : 0.8982    logloss : 0.321
  Test result:
  auc : 0.8966    logloss : 0.3241

  attention_size:32, learning_rate:0.0005, n_layers:3
  Valid result:
  auc : 0.8938    logloss : 0.3351
  Test result:
  auc : 0.8924    logloss : 0.3382

  attention_size:32, learning_rate:0.005, n_layers:3
  Valid result:
  auc : 0.8974    logloss : 0.3237
  Test result:
  auc : 0.8959    logloss : 0.3268

  attention_size:16, learning_rate:0.001, n_layers:4
  Valid result:
  auc : 0.8938    logloss : 0.328
  Test result:
  auc : 0.8928    logloss : 0.3303

  attention_size:16, learning_rate:0.005, n_layers:2
  Valid result:
  auc : 0.8977    logloss : 0.3192
  Test result:
  auc : 0.8963    logloss : 0.3219

  attention_size:8, learning_rate:0.005, n_layers:3
  Valid result:
  auc : 0.8956    logloss : 0.3196
  Test result:
  auc : 0.8941    logloss : 0.3216

  attention_size:16, learning_rate:0.001, n_layers:2
  Valid result:
  auc : 0.8938    logloss : 0.333
  Test result:
  auc : 0.8928    logloss : 0.3365

  attention_size:32, learning_rate:0.0005, n_layers:2
  Valid result:
  auc : 0.8921    logloss : 0.3315
  Test result:
  auc : 0.8894    logloss : 0.3358

  attention_size:16, learning_rate:0.0005, n_layers:3
  Valid result:
  auc : 0.893    logloss : 0.334
  Test result:
  auc : 0.891    logloss : 0.3379

  attention_size:8, learning_rate:0.005, n_layers:2
  Valid result:
  auc : 0.8956    logloss : 0.3265
  Test result:
  auc : 0.894    logloss : 0.3297

  attention_size:32, learning_rate:0.0005, n_layers:4
  Valid result:
  auc : 0.8926    logloss : 0.3368
  Test result:
  auc : 0.8922    logloss : 0.3376
```

- **Logging Result**:

```yaml
  93% 25/27 [49:57<03:59, 119.89s/trial, best loss: -0.8982]
  best params:  {'attention_size': 32, 'learning_rate': 0.005, 'n_layers': 2}
  best result: 
  {'model': 'FiGNN', 'best_valid_score': 0.8982, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('auc', 0.8982), ('logloss', 0.321)]), 'test_result': OrderedDict([('auc', 0.8966), ('logloss', 0.3241)])}
```

