Dataset Download
================================

In RecBole, we have collected and released 28 commonly-used publiced dataset (detailed as `Dataset List </dataset_list.html>`_).
Users can freely download these datasets in the following three ways:

1. Automatically downloading
-----------------------------
For the convenience of users, we implement automatically downloading module in RecBole and now we support to download the :doc:`atomic_files` of 28 commonly-used
publiced datasets (detailed as `Dataset List </dataset_list.html>`_). If you want to run models on a dataset, you just need to set the
`dataset` and then the data files will be automatically downloaded.

For example, if you want to run BPR model on the ml-1m dataset but you don't prepare the :doc:`atomic_files` of ml-1m dataset,
you can use our automatically downloading module to download the data. 
All you need is to run the model as normal, and RecBole will automatically check if you have the data files, if not, it will begin to download the data files
and you will get the output like this:

.. code:: none

    23 Aug 10:02    INFO  Prepare to download dataset [ml-1m] from [https://recbole.s3-accelerate.amazonaws.com/ProcessedDatasets/MovieLens/ml-1m.zip].
    23 Aug 10:02    INFO  Downloading https://recbole.s3-accelerate.amazonaws.com/ProcessedDatasets/MovieLens/ml-1m.zip
    Downloaded 0.01 GB: 100%|█████████████████████████████████████████████████████████████████████████████| 7/7 [04:16<00:00, 36.65s/it]
    23 Aug 10:06    INFO  Extracting dataset/ml-1m/ml-1m.zip
    23 Aug 10:06    INFO  Downloading done.

And next time you can directly run other models on ml-1m dataset.

2. Download from cloud disk
-----------------------------
Besides automatically downloading, we also upload our collected and converted atomic files of 28 datasets in `Google Drive <https://drive.google.com/drive/folders/1so0lckI6N6_niVEYaBu-LIcpOdZf99kj?usp=sharing>`_ and `Baidu Wangpan <https://pan.baidu.com/s/1p51sWMgVFbAaHQmL4aD_-g>`_ (Password: e272).
You can also download the data from these two resources by yourself.

3. Covert the raw data 
-----------------------------
If you have already download the raw data, you can also covert them into atomic files format by yourself. 
And we have already publiced some converting scripts in `RecDatasets <https://github.com/RUCAIBox/RecDatasets>`_.



