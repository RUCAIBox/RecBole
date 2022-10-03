# Knowledge-aware Recommendation

- **Dataset**: [Lastfm-track](../../md/lastfm-track_kg.md)

- **Model**: [MCCLK](https://recbole.io/docs/user_guide/model/knowledge/mcclk.html)

- **Time cost**: 

- **Hyper-parameter searching** (hyper.test):

  ```yaml
  learning_rate choice [1e-3]
  node_dropout_rate choice [0.1,0.3,0.5]
  mess_dropout_rate choice [0.1]
  build_graph_separately choice [True, False]
  loss_type choice ['BPR']
  lightgcn_layer choice [3]
  item_agg_layer choice [2]
  ```

- **Best parameters**:

  ```yaml
  learning_rate: 1e-3
  node_dropout_rate: 0.1,0.3,0.5
  mess_dropout_rate: 0.1
  build_graph_separately: True, False
  loss_type: 'BPR'
  lightgcn_layer: 3
  item_agg_layer: 2
  ```

- **Hyper-parameter logging** (hyper.result):

  ```yaml
  
  ```


- **Logging Result**:

  ```yaml
  ```

  

