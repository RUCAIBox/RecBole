# @Time   : 2021/03/20
# @Author : Yushuo Chen
# @Email  : chenyushuo@ruc.edu.cn

"""
save and load example
========================
Here is the sample code for the save and load in RecBole.

The path to saved data or model can be found in the output of RecBole.
"""


from recbole.quick_start import run_recbole, load_data_and_model


def save_example():
    # configurations initialization
    config_dict = {
        "checkpoint_dir": "../saved",
        "save_dataset": True,
        "save_dataloaders": True,
    }
    run_recbole(model="BPR", dataset="ml-100k", config_dict=config_dict)


def load_example():
    # Filtered dataset and split dataloaders are created according to 'config'.
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file="../saved/BPR-Aug-20-2021_03-32-13.pth",
    )

    # Filtered dataset is loaded from file, and split dataloaders are created according to 'config'.
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file="../saved/BPR-Aug-20-2021_03-32-13.pth",
        dataset_file="../saved/ml-100k-dataset.pth",
    )

    # Dataset is neither created nor loaded, and split dataloaders are loaded from file.
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file="../saved/BPR-Aug-20-2021_03-32-13.pth",
        dataloader_file="../saved/ml-100k-for-BPR-dataloader.pth",
    )
    assert dataset is None

    # Filtered dataset and split dataloaders are loaded from file.
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file="../saved/BPR-Aug-20-2021_03-32-13.pth",
        dataset_file="../saved/ml-100k-dataset.pth",
        dataloader_file="../saved/ml-100k-for-BPR-dataloader.pth",
    )


if __name__ == "__main__":
    save_example()
    # load_example()
