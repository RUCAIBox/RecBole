# -*- coding: utf-8 -*-
# @Time   : 2020/10/24
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn


import unittest

from recbole.quick_start import objective_function


class TestGeneralRecommender(unittest.TestCase):

    def test_bpr(self):
        config_dict = {
            'model': 'BPR',
        }
        objective_function(config_dict=config_dict,
                           config_file_list=['test_model.yaml'], saved=False)

    def test_neumf(self):
        config_dict = {
            'model': 'NeuMF',
        }
        objective_function(config_dict=config_dict,
                           config_file_list=['test_model.yaml'], saved=False)

    def test_dmf(self):
        config_dict = {
            'model': 'DMF',
        }
        objective_function(config_dict=config_dict,
                           config_file_list=['test_model.yaml'], saved=False)

    def test_nais(self):
        config_dict = {
            'model': 'NAIS',
        }
        objective_function(config_dict=config_dict,
                           config_file_list=['test_model.yaml'], saved=False)

    def test_gcmc(self):
        config_dict = {
            'model': 'GCMC',
        }
        objective_function(config_dict=config_dict,
                           config_file_list=['test_model.yaml'], saved=False)

    def test_ngcf(self):
        config_dict = {
            'model': 'NGCF',
        }
        objective_function(config_dict=config_dict,
                           config_file_list=['test_model.yaml'], saved=False)

    def test_lightgcn(self):
        config_dict = {
            'model': 'LightGCN',
        }
        objective_function(config_dict=config_dict,
                           config_file_list=['test_model.yaml'], saved=False)

    def test_dgcf(self):
        config_dict = {
            'model': 'DGCF',
        }
        objective_function(config_dict=config_dict,
                           config_file_list=['test_model.yaml'], saved=False)

    def test_fism(self):
        config_dict = {
            'model': 'FISM',
        }
        objective_function(config_dict=config_dict,
                           config_file_list=['test_model.yaml'], saved=False)

    def test_spectralcf(self):
        config_dict = {
            'model': 'SpectralCF',
        }
        objective_function(config_dict=config_dict,
                           config_file_list=['test_model.yaml'], saved=False)

    def test_pop(self):
        config_dict = {
            'model': 'Pop',
        }
        objective_function(config_dict=config_dict,
                           config_file_list=['test_model.yaml'], saved=False)

    def test_itemknn(self):
        config_dict = {
            'model': 'ItemKNN',
        }
        objective_function(config_dict=config_dict,
                           config_file_list=['test_model.yaml'], saved=False)

    def test_convncf(self):
        config_dict = {
            'model': 'ConvNCF',
        }
        objective_function(config_dict=config_dict,
                           config_file_list=['test_model.yaml'], saved=False)


class TestContextRecommender(unittest.TestCase):
    # todo: more complex context information should be test, such as criteo dataset

    def test_fm(self):
        config_dict = {
            'model': 'FM',
        }
        objective_function(config_dict=config_dict,
                           config_file_list=['test_model.yaml'], saved=False)

    def test_dcn(self):
        config_dict = {
            'model': 'DCN',
        }
        objective_function(config_dict=config_dict,
                           config_file_list=['test_model.yaml'], saved=False)

    def test_xdeepfm(self):
        config_dict = {
            'model': 'xDeepFM',
        }
        objective_function(config_dict=config_dict,
                           config_file_list=['test_model.yaml'], saved=False)

    def test_afm(self):
        config_dict = {
            'model': 'AFM',
        }
        objective_function(config_dict=config_dict,
                           config_file_list=['test_model.yaml'], saved=False)

    def test_autoint(self):
        config_dict = {
            'model': 'AutoInt',
        }
        objective_function(config_dict=config_dict,
                           config_file_list=['test_model.yaml'], saved=False)

    def test_deepfm(self):
        config_dict = {
            'model': 'DeepFM',
        }
        objective_function(config_dict=config_dict,
                           config_file_list=['test_model.yaml'], saved=False)

    def test_dssm(self):
        config_dict = {
            'model': 'DSSM',
        }
        objective_function(config_dict=config_dict,
                           config_file_list=['test_model.yaml'], saved=False)

    def test_ffm(self):
        config_dict = {
            'model': 'FFM',
        }
        objective_function(config_dict=config_dict,
                           config_file_list=['test_model.yaml'], saved=False)

    def test_fnn(self):
        config_dict = {
            'model': 'FNN',
        }
        objective_function(config_dict=config_dict,
                           config_file_list=['test_model.yaml'], saved=False)

    def test_fwfm(self):
        config_dict = {
            'model': 'FwFM',
        }
        objective_function(config_dict=config_dict,
                           config_file_list=['test_model.yaml'], saved=False)

    def test_lr(self):
        config_dict = {
            'model': 'LR',
        }
        objective_function(config_dict=config_dict,
                           config_file_list=['test_model.yaml'], saved=False)

    def test_nfm(self):
        config_dict = {
            'model': 'NFM',
        }
        objective_function(config_dict=config_dict,
                           config_file_list=['test_model.yaml'], saved=False)

    def test_pnn(self):
        config_dict = {
            'model': 'PNN',
        }
        objective_function(config_dict=config_dict,
                           config_file_list=['test_model.yaml'], saved=False)

    def test_widedeep(self):
        config_dict = {
            'model': 'WideDeep',
        }
        objective_function(config_dict=config_dict,
                           config_file_list=['test_model.yaml'], saved=False)


class TestSequentialRecommender(unittest.TestCase):

    def test_fpmc(self):
        config_dict = {
            'model': 'FPMC',
        }
        objective_function(config_dict=config_dict,
                           config_file_list=['test_model.yaml'], saved=False)

    def test_transrec(self):
        config_dict = {
            'model': 'TransRec',
        }
        objective_function(config_dict=config_dict,
                           config_file_list=['test_model.yaml'], saved=False)

    def test_gru4rec(self):
        config_dict = {
            'model': 'GRU4Rec',
        }
        objective_function(config_dict=config_dict,
                           config_file_list=['test_model.yaml'], saved=False)

    def test_caser(self):
        config_dict = {
            'model': 'Caser',
            'MAX_ITEM_LIST_LENGTH': 10,
        }
        objective_function(config_dict=config_dict,
                           config_file_list=['test_model.yaml'], saved=False)

    def test_sasrec(self):
        config_dict = {
            'model': 'SASRec',
        }
        objective_function(config_dict=config_dict,
                           config_file_list=['test_model.yaml'], saved=False)

    def test_bert4rec(self):
        config_dict = {
            'model': 'BERT4Rec',
        }
        objective_function(config_dict=config_dict,
                           config_file_list=['test_model.yaml'], saved=False)

    def test_stamp(self):
        config_dict = {
            'model': 'STAMP',
        }
        objective_function(config_dict=config_dict,
                           config_file_list=['test_model.yaml'], saved=False)

    def test_narm(self):
        config_dict = {
            'model': 'NARM',
        }
        objective_function(config_dict=config_dict,
                           config_file_list=['test_model.yaml'], saved=False)

    def test_nextitnet(self):
        config_dict = {
            'model': 'NextItNet',
            'reproducibility': False,
        }
        objective_function(config_dict=config_dict,
                           config_file_list=['test_model.yaml'], saved=False)

    def test_srgnn(self):
        config_dict = {
            'model': 'SRGNN',
            'MAX_ITEM_LIST_LENGTH': 3,
        }
        objective_function(config_dict=config_dict,
                           config_file_list=['test_model.yaml'], saved=False)

    def test_gcsan(self):
        config_dict = {
            'model': 'GCSAN',
            'MAX_ITEM_LIST_LENGTH': 3,
        }
        objective_function(config_dict=config_dict,
                           config_file_list=['test_model.yaml'], saved=False)

    def test_gru4recf(self):
        config_dict = {
            'model': 'GRU4RecF',
            'load_col': {'inter': ['user_id', 'item_id', 'rating', 'timestamp'],
                         'item': ['item_id', 'class']},
            'selected_features': ['class'],
        }
        objective_function(config_dict=config_dict,
                           config_file_list=['test_model.yaml'], saved=False)

    def test_sasrecf(self):
        config_dict = {
            'model': 'SASRecF',
            'load_col': {'inter': ['user_id', 'item_id', 'rating', 'timestamp'],
                         'item': ['item_id', 'class']},
            'selected_features': ['class'],
        }
        objective_function(config_dict=config_dict,
                           config_file_list=['test_model.yaml'], saved=False)

    def test_fdsa(self):
        config_dict = {
            'model': 'FDSA',
            'load_col': {'inter': ['user_id', 'item_id', 'rating', 'timestamp'],
                         'item': ['item_id', 'class']},
            'selected_features': ['class'],
        }
        objective_function(config_dict=config_dict,
                           config_file_list=['test_model.yaml'], saved=False)

    # def test_gru4reckg(self):
    #     config_dict = {
    #         'model': 'GRU4RecKG',
    #     }
    #     objective_function(config_dict=config_dict,
    #                        config_file_list=['test_model.yaml'], saved=False)

    # def test_din(self):
    #     config_dict = {
    #         'model': 'DIN',
    #     }
    #     objective_function(config_dict=config_dict,
    #                        config_file_list=['test_model.yaml'], saved=False)

    # def test_s3rec(self):
    #     config_dict = {
    #         'model': 'S3Rec',
    #         'train_stage': 'pretrain',
    #         'save_step': 1,
    #         'load_col': 'inter: {}'
    #     }
    #     objective_function(config_dict=config_dict,
    #                        config_file_list=['test_model.yaml'], saved=False)
    #     config_dict['train_stage'] = 'finetune'
    #     config_dict['pre_model_path'] = 'saved/S3Rec-ml-100k-1.pth'
    #     objective_function(config_dict=config_dict,
    #                        config_file_list=['test_model.yaml'], saved=False)


class TestKnowledgeRecommender(unittest.TestCase):

    def test_cke(self):
        config_dict = {
            'model': 'CKE',
        }
        objective_function(config_dict=config_dict,
                           config_file_list=['test_model.yaml'], saved=False)

    def test_ktup(self):
        config_dict = {
            'model': 'KTUP',
            'train_rec_step': 1,
            'train_kg_step': 1,
            'epochs': 2,
        }
        objective_function(config_dict=config_dict,
                           config_file_list=['test_model.yaml'], saved=False)

    def test_cfkg(self):
        config_dict = {
            'model': 'CFKG',
        }
        objective_function(config_dict=config_dict,
                           config_file_list=['test_model.yaml'], saved=False)

    def test_kgat(self):
        config_dict = {
            'model': 'KGAT',
        }
        objective_function(config_dict=config_dict,
                           config_file_list=['test_model.yaml'], saved=False)

    def test_ripplenet(self):
        config_dict = {
            'model': 'RippleNet',
        }
        objective_function(config_dict=config_dict,
                           config_file_list=['test_model.yaml'], saved=False)

    def test_mkr(self):
        config_dict = {
            'model': 'MKR',
        }
        objective_function(config_dict=config_dict,
                           config_file_list=['test_model.yaml'], saved=False)

    def test_kgcn(self):
        config_dict = {
            'model': 'KGCN',
        }
        objective_function(config_dict=config_dict,
                           config_file_list=['test_model.yaml'], saved=False)

    def test_kgnnls(self):
        config_dict = {
            'model': 'KGNNLS',
        }
        objective_function(config_dict=config_dict,
                           config_file_list=['test_model.yaml'], saved=False)


class Test(unittest.TestCase):

    def test_dmf(self):
        pass

    def test_gcmc(self):
        pass

    def test_bert4rec(self):
        pass

    def test_cfkg(self):
        pass

    def test_din(self):
        pass

    def test_fdsa(self):
        pass

    def test_gcsan(self):
        pass

    def test_gru4recf(self):
        pass

    def test_kgat(self):
        pass

    def test_kgcn(self):
        pass

    def test_kgnnls(self):
        pass

    def test_ktup(self):
        pass

    def test_mkr(self):
        pass

    def test_nais(self):
        pass

    def test_nextitnet(self):
        pass

    def test_s3rec(self):
        pass

    def test_sasrec(self):
        pass

    def test_sasrecf(self):
        pass


if __name__ == '__main__':
    unittest.main()
