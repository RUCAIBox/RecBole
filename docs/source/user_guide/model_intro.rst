Model Introduction
=====================
We implement 94 recommendation models covering general recommendation, sequential recommendation,
context-aware recommendation and knowledge-based recommendation. A brief introduction to these models are as follows:


General Recommendation
--------------------------
In the class of general recommendation, the interaction of users and items(.inter file) is the only data
that can be used by model. Usually, the models are trained on implicit feedback data and evaluated under the
task of top-n recommendation. All the collaborative filter(CF) based models are classified in this class.

.. toctree::
   :maxdepth: 1

   model/general/asymknn
   model/general/pop
   model/general/itemknn
   model/general/bpr
   model/general/neumf
   model/general/convncf
   model/general/dmf
   model/general/fism
   model/general/nais
   model/general/spectralcf
   model/general/gcmc
   model/general/ngcf
   model/general/lightgcn
   model/general/dgcf
   model/general/line
   model/general/multivae
   model/general/multidae
   model/general/macridvae
   model/general/cdae
   model/general/enmf
   model/general/nncf
   model/general/ract
   model/general/recvae
   model/general/ease
   model/general/slimelastic
   model/general/sgl
   model/general/admmslim
   model/general/nceplrec
   model/general/simplex
   model/general/ncl
   model/general/random
   model/general/diffrec
   model/general/ldiffrec


Context-aware Recommendation
-------------------------------
Context-aware recommendation can be seen as an extension of click-through rate prediction. All the model in this
class can be used for CTR prediction. Usually, the dataset is explicit and contains label field. Other feature fields
are also support for these models. And evaluation is always conducted in the way of binary classification.

.. toctree::
   :maxdepth: 1

   model/context/lr
   model/context/fm
   model/context/nfm
   model/context/deepfm
   model/context/xdeepfm
   model/context/afm
   model/context/ffm
   model/context/fwfm
   model/context/fnn
   model/context/pnn
   model/context/dssm
   model/context/widedeep
   model/context/din
   model/context/dien
   model/context/dcn
   model/context/dcnv2
   model/context/autoint
   model/context/xgboost
   model/context/lightgbm
   model/context/kd_dagfm
   model/context/fignn
   model/context/eulernet


Sequential Recommendation
---------------------------------
The task of sequential recommendation(next-item recommendation) is the same as general recommendation which sorts a list of items according
to preference. While the history interactions are organized in sequences and the model tends to characterize
the sequential data. The models of session-based recommendation are also included in this class.

.. toctree::
   :maxdepth: 1

   model/sequential/fpmc
   model/sequential/gru4rec
   model/sequential/narm
   model/sequential/stamp
   model/sequential/caser
   model/sequential/nextitnet
   model/sequential/transrec
   model/sequential/sasrec
   model/sequential/bert4rec
   model/sequential/srgnn
   model/sequential/gcsan
   model/sequential/gru4recf
   model/sequential/sasrecf
   model/sequential/fdsa
   model/sequential/s3rec
   model/sequential/gru4reckg
   model/sequential/ksr
   model/sequential/fossil
   model/sequential/shan
   model/sequential/repeatnet
   model/sequential/hgn
   model/sequential/hrm
   model/sequential/npe
   model/sequential/lightsans
   model/sequential/sine
   model/sequential/core
   model/sequential/fearec
   model/sequential/sasreccpr
   model/sequential/gru4reccpr


Knowledge-based Recommendation
---------------------------------
Knowledge-based recommendation introduces an external knowledge graph to enhance general or sequential recommendation.

.. toctree::
   :maxdepth: 1

   model/knowledge/cke
   model/knowledge/cfkg
   model/knowledge/ktup
   model/knowledge/kgat
   model/knowledge/kgin
   model/knowledge/ripplenet
   model/knowledge/mcclk
   model/knowledge/mkr
   model/knowledge/kgcn
   model/knowledge/kgnnls


