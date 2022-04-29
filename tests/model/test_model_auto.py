# -*- coding: utf-8 -*-
# @Time   : 2020/10/24
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE
# @Time    :   2020/11/17
# @Author  :   Xingyu Pan
# @email   :   panxy@ruc.edu.cn

import os
import unittest

from recbole.quick_start import objective_function

current_path = os.path.dirname(os.path.realpath(__file__))
config_file_list = [os.path.join(current_path, 'test_model.yaml')]


def quick_test(config_dict):
    objective_function(config_dict=config_dict, config_file_list=config_file_list, saved=False)


class TestGeneralRecommender(unittest.TestCase):

    def test_pop(self):
        config_dict = {
            'model': 'Pop',
        }
        quick_test(config_dict)

    def test_itemknn(self):
        config_dict = {
            'model': 'ItemKNN',
        }
        quick_test(config_dict)

    def test_bpr(self):
        config_dict = {
            'model': 'BPR',
        }
        quick_test(config_dict)

    def test_bpr_with_dns(self):
        config_dict = {
            'model': 'BPR',
            'neg_sampling': {
                'uniform': 1,
                'dynamic': 2
            }
        }
        quick_test(config_dict)

    def test_neumf(self):
        config_dict = {
            'model': 'NeuMF',
        }
        quick_test(config_dict)

    def test_convncf(self):
        config_dict = {
            'model': 'ConvNCF',
        }
        quick_test(config_dict)

    def test_dmf(self):
        config_dict = {
            'model': 'DMF',
        }
        quick_test(config_dict)

    def test_dmf_with_rating(self):
        config_dict = {
            'model': 'DMF',
            'inter_matrix_type': 'rating',
        }
        quick_test(config_dict)

    def test_fism(self):
        config_dict = {
            'model': 'FISM',
        }
        quick_test(config_dict)

    def test_fism_with_split_to_and_alpha(self):
        config_dict = {
            'model': 'FISM',
            'split_to': 10,
            'alpha': 0.5,
        }
        quick_test(config_dict)

    def test_nais(self):
        config_dict = {
            'model': 'NAIS',
        }
        quick_test(config_dict)

    def test_nais_with_concat(self):
        config_dict = {
            'model': 'NAIS',
            'algorithm': 'concat',
            'split_to': 10,
            'alpha': 0.5,
            'beta': 0.1,
        }
        quick_test(config_dict)

    def test_spectralcf(self):
        config_dict = {
            'model': 'SpectralCF',
        }
        quick_test(config_dict)

    def test_gcmc(self):
        config_dict = {
            'model': 'GCMC',
        }
        quick_test(config_dict)

    def test_gcmc_with_stack(self):
        config_dict = {
            'model': 'GCMC',
            'accum': 'stack',
            'sparse_feature': False,
        }
        quick_test(config_dict)

    def test_ngcf(self):
        config_dict = {
            'model': 'NGCF',
        }
        quick_test(config_dict)

    def test_lightgcn(self):
        config_dict = {
            'model': 'LightGCN',
        }
        quick_test(config_dict)

    def test_dgcf(self):
        config_dict = {
            'model': 'DGCF',
        }
        quick_test(config_dict)

    def test_line(self):
        config_dict = {
            'model': 'LINE',
        }
        quick_test(config_dict)

    def test_ease(self):
        config_dict = {
            'model': 'EASE',
        }
        quick_test(config_dict)

    def test_MultiDAE(self):
        config_dict = {
            'model': 'MultiDAE',
            'neg_sampling': None
        }
        quick_test(config_dict)

    def test_MultiVAE(self):
        config_dict = {
            'model': 'MultiVAE',
            'neg_sampling': None
        }
        quick_test(config_dict)

    def test_enmf(self):
        config_dict = {
            'model': 'ENMF',
            'neg_sampling': None,
        }
        quick_test(config_dict)

    def test_MacridVAE(self):
        config_dict = {
            'model': 'MacridVAE',
            'neg_sampling': None
        }
        quick_test(config_dict)

    def test_CDAE(self):
        config_dict = {
            'model': 'CDAE',
            'neg_sampling': None
        }
        quick_test(config_dict)

    def test_NNCF(self):
        config_dict = {
            'model': 'NNCF',
        }
        quick_test(config_dict)

    def test_RecVAE(self):
        config_dict = {
            'model': 'RecVAE',
            'neg_sampling': None
        }
        quick_test(config_dict)

    def test_slimelastic(self):
        config_dict = {
            'model': 'SLIMElastic',
        }
        quick_test(config_dict)
    
    def test_SGL(self):
        config_dict = {
            'model': 'SGL',
        }
        quick_test(config_dict)
    
    def test_ADMMSLIM(self):
        config_dict = {
            'model': 'ADMMSLIM',
        }
        quick_test(config_dict)

    def test_NCEPLRec(self):
        config_dict = {
            'model': 'NCEPLRec',
        }
        quick_test(config_dict)

    def test_NCL(self):
        config_dict = {
            'model': 'NCL',
            'num_clusters': 100
        }
        quick_test(config_dict)


class TestContextRecommender(unittest.TestCase):
    # todo: more complex context information should be test, such as criteo dataset

    def test_lr(self):
        config_dict = {
            'model': 'LR',
            'threshold': {'rating': 4},
        }
        quick_test(config_dict)

    def test_fm(self):
        config_dict = {
            'model': 'FM',
            'threshold': {'rating': 4},
        }
        quick_test(config_dict)

    def test_nfm(self):
        config_dict = {
            'model': 'NFM',
            'threshold': {'rating': 4},
        }
        quick_test(config_dict)

    def test_deepfm(self):
        config_dict = {
            'model': 'DeepFM',
            'threshold': {'rating': 4},
        }
        quick_test(config_dict)

    def test_xdeepfm(self):
        config_dict = {
            'model': 'xDeepFM',
            'threshold': {'rating': 4},
        }
        quick_test(config_dict)

    def test_xdeepfm_with_direct(self):
        config_dict = {
            'model': 'xDeepFM',
            'threshold': {'rating': 4},
            'direct': True,
        }
        quick_test(config_dict)

    def test_afm(self):
        config_dict = {
            'model': 'AFM',
            'threshold': {'rating': 4},
        }
        quick_test(config_dict)

    def test_fnn(self):
        config_dict = {
            'model': 'FNN',
            'threshold': {'rating': 4},
        }
        quick_test(config_dict)

    def test_pnn(self):
        config_dict = {
            'model': 'PNN',
            'threshold': {'rating': 4},
        }
        quick_test(config_dict)

    def test_pnn_with_use_inner_and_use_outer(self):
        config_dict = {
            'model': 'PNN',
            'threshold': {'rating': 4},
            'use_inner': True,
            'use_outer': True,
        }
        quick_test(config_dict)

    def test_pnn_without_use_inner_and_use_outer(self):
        config_dict = {
            'model': 'PNN',
            'threshold': {'rating': 4},
            'use_inner': False,
            'use_outer': False,
        }
        quick_test(config_dict)

    def test_dssm(self):
        config_dict = {
            'model': 'DSSM',
            'threshold': {'rating': 4},
        }
        quick_test(config_dict)

    def test_widedeep(self):
        config_dict = {
            'model': 'WideDeep',
            'threshold': {'rating': 4},
        }
        quick_test(config_dict)

    def test_autoint(self):
        config_dict = {
            'model': 'AutoInt',
            'threshold': {'rating': 4},
        }
        quick_test(config_dict)

    def test_ffm(self):
        config_dict = {
            'model': 'FFM',
            'threshold': {'rating': 4},
        }
        quick_test(config_dict)

    def test_fwfm(self):
        config_dict = {
            'model': 'FwFM',
            'threshold': {'rating': 4},
        }
        quick_test(config_dict)

    def test_dcn(self):
        config_dict = {
            'model': 'DCN',
            'threshold': {'rating': 4},
        }
        quick_test(config_dict)

    def test_xgboost(self):
        config_dict = {
            'model': 'xgboost',
            'threshold': {'rating': 4},
            'xgb_params': {
                'booster': 'gbtree',
                'objective': 'binary:logistic',
                'eval_metric': ['auc', 'logloss']
            },
            'xgb_num_boost_round': 1,
        }
        quick_test(config_dict)

    def test_lightgbm(self):
        config_dict = {
            'model': 'lightgbm',
            'threshold': {'rating': 4},
            'lgb_params': {
                'boosting': 'gbdt',
                'objective': 'binary',
                'metric': ['auc', 'binary_logloss']
            },
            'lgb_num_boost_round': 1,
        }
        quick_test(config_dict)


class TestSequentialRecommender(unittest.TestCase):

    def test_din(self):
        config_dict = {
            'model': 'DIN',
        }
        quick_test(config_dict)

    def test_dien(self):
        config_dict = {
            'model': 'DIEN',
        }
        quick_test(config_dict)

    def test_fpmc(self):
        config_dict = {
            'model': 'FPMC',
        }
        quick_test(config_dict)

    def test_gru4rec(self):
        config_dict = {
            'model': 'GRU4Rec',
            'neg_sampling': None
        }
        quick_test(config_dict)

    def test_gru4rec_with_BPR_loss(self):
        config_dict = {
            'model': 'GRU4Rec',
            'loss_type': 'BPR',
        }
        quick_test(config_dict)

    def test_narm(self):
        config_dict = {
            'model': 'NARM',
            'neg_sampling': None
        }
        quick_test(config_dict)

    def test_narm_with_BPR_loss(self):
        config_dict = {
            'model': 'NARM',
            'loss_type': 'BPR',
        }
        quick_test(config_dict)

    def test_stamp(self):
        config_dict = {
            'model': 'STAMP',
            'neg_sampling': None
        }
        quick_test(config_dict)

    def test_stamp_with_BPR_loss(self):
        config_dict = {
            'model': 'STAMP',
            'loss_type': 'BPR',
        }
        quick_test(config_dict)

    def test_caser(self):
        config_dict = {
            'model': 'Caser',
            'MAX_ITEM_LIST_LENGTH': 10,
            'reproducibility': False,
            'neg_sampling': None
        }
        quick_test(config_dict)

    def test_caser_with_BPR_loss(self):
        config_dict = {
            'model': 'Caser',
            'loss_type': 'BPR',
            'MAX_ITEM_LIST_LENGTH': 10,
            'reproducibility': False,
        }
        quick_test(config_dict)

    def test_nextitnet(self):
        config_dict = {
            'model': 'NextItNet',
            'reproducibility': False,
            'neg_sampling': None
        }
        quick_test(config_dict)

    def test_nextitnet_with_BPR_loss(self):
        config_dict = {
            'model': 'NextItNet',
            'loss_type': 'BPR',
            'reproducibility': False,
        }
        quick_test(config_dict)

    def test_transrec(self):
        config_dict = {
            'model': 'TransRec',
        }
        quick_test(config_dict)

    def test_sasrec(self):
        config_dict = {
            'model': 'SASRec',
            'neg_sampling': None
        }
        quick_test(config_dict)

    def test_sasrec_with_BPR_loss_and_relu(self):
        config_dict = {
            'model': 'SASRec',
            'loss_type': 'BPR',
            'hidden_act': 'relu'
        }
        quick_test(config_dict)

    def test_sasrec_with_BPR_loss_and_sigmoid(self):
        config_dict = {
            'model': 'SASRec',
            'loss_type': 'BPR',
            'hidden_act': 'sigmoid'
        }
        quick_test(config_dict)

    def test_srgnn(self):
        config_dict = {
            'model': 'SRGNN',
            'MAX_ITEM_LIST_LENGTH': 3,
            'neg_sampling': None
        }
        quick_test(config_dict)

    def test_srgnn_with_BPR_loss(self):
        config_dict = {
            'model': 'SRGNN',
            'loss_type': 'BPR',
            'MAX_ITEM_LIST_LENGTH': 3,
        }
        quick_test(config_dict)

    def test_gcsan(self):
        config_dict = {
            'model': 'GCSAN',
            'MAX_ITEM_LIST_LENGTH': 3,
            'neg_sampling': None
        }
        quick_test(config_dict)

    def test_gcsan_with_BPR_loss_and_tanh(self):
        config_dict = {
            'model': 'GCSAN',
            'loss_type': 'BPR',
            'hidden_act': 'tanh',
            'MAX_ITEM_LIST_LENGTH': 3,
        }
        quick_test(config_dict)

    def test_gru4recf(self):
        config_dict = {
            'model': 'GRU4RecF',
            'neg_sampling': None
        }
        quick_test(config_dict)

    def test_gru4recf_with_max_pooling(self):
        config_dict = {
            'model': 'GRU4RecF',
            'pooling_mode': 'max',
            'neg_sampling': None
        }
        quick_test(config_dict)

    def test_gru4recf_with_sum_pooling(self):
        config_dict = {
            'model': 'GRU4RecF',
            'pooling_mode': 'sum',
            'neg_sampling': None
        }
        quick_test(config_dict)

    def test_sasrecf(self):
        config_dict = {
            'model': 'SASRecF',
            'neg_sampling': None
        }
        quick_test(config_dict)

    def test_sasrecf_with_max_pooling(self):
        config_dict = {
            'model': 'SASRecF',
            'pooling_mode': 'max',
            'neg_sampling': None
        }
        quick_test(config_dict)

    def test_sasrecf_with_sum_pooling(self):
        config_dict = {
            'model': 'SASRecF',
            'pooling_mode': 'sum',
            'neg_sampling': None
        }
        quick_test(config_dict)

    def test_hrm(self):
        config_dict = {
            'model': 'HRM',
            'neg_sampling': None
        }
        quick_test(config_dict)

    def test_hrm_with_BPR_loss(self):
        config_dict = {
            'model': 'HRM',
            'loss_type': 'BPR',
        }
        quick_test(config_dict)

    def test_npe(self):
        config_dict = {
            'model': 'NPE',
            'neg_sampling': None
        }
        quick_test(config_dict)

    def test_npe_with_BPR_loss(self):
        config_dict = {
            'model': 'NPE',
            'loss_type': 'BPR',
        }
        quick_test(config_dict)

    def test_shan(self):
        config_dict = {
            'model': 'SHAN',
            'neg_sampling': None
        }
        quick_test(config_dict)

    def test_shan_with_BPR_loss(self):
        config_dict = {
            'model': 'SHAN',
            'loss_type': 'BPR',
        }
        quick_test(config_dict)

    def test_hgn(self):
        config_dict = {
            'model': 'HGN',
        }
        quick_test(config_dict)

    def test_hgn_with_CE_loss(self):
        config_dict = {
            'model': 'HGN',
            'loss_type': 'CE',
            'neg_sampling': None
        }
        quick_test(config_dict)

    def test_fossil(self):
        config_dict = {
            'model': 'FOSSIL',
            'neg_sampling': None
        }
        quick_test(config_dict)

    def test_repeat_net(self):
        config_dict = {
            'model': 'RepeatNet',
        }
        quick_test(config_dict)

    def test_fdsa(self):
        config_dict = {
            'model': 'FDSA',
            'neg_sampling': None
        }
        quick_test(config_dict)

    def test_fdsa_with_max_pooling(self):
        config_dict = {
            'model': 'FDSA',
            'pooling_mode': 'max',
            'neg_sampling': None
        }
        quick_test(config_dict)

    def test_fdsa_with_sum_pooling(self):
        config_dict = {
            'model': 'FDSA',
            'pooling_mode': 'sum',
            'neg_sampling': None
        }
        quick_test(config_dict)

    def test_bert4rec(self):
        config_dict = {
            'model': 'BERT4Rec',
            'neg_sampling': None
        }
        quick_test(config_dict)

    def test_bert4rec_with_BPR_loss_and_swish(self):
        config_dict = {
            'model': 'BERT4Rec',
            'loss_type': 'BPR',
            'hidden_act': 'swish'
        }
        quick_test(config_dict)

    def test_lightsans(self):
        config_dict = {
            'model': 'LightSANs',
            'neg_sampling': None
        }
        quick_test(config_dict)

    def test_lightsans_with_BPR_loss(self):
        config_dict = {
            'model': 'LightSANs',
            'loss_type': 'BPR',
        }
        quick_test(config_dict)

    def test_sine(self):
        config_dict = {
            'model': 'SINE',
            'neg_sampling': None
        }
        quick_test(config_dict)

    def test_sine_with_BPR_loss(self):
        config_dict = {
            'model': 'SINE',
            'loss_type': 'BPR',
        }
        quick_test(config_dict)

    def test_sine_with_NLL_loss(self):
        config_dict = {
            'model': 'SINE',
            'neg_sampling': None,
            'loss_type': 'NLL',
        }
        quick_test(config_dict)

    def test_core_trm(self):
        config_dict = {
            'model': 'CORE',
            'neg_sampling': None,
            'dnn_type': 'trm'
        }
        quick_test(config_dict)

    def test_core_ave(self):
        config_dict = {
            'model': 'CORE',
            'neg_sampling': None,
            'dnn_type': 'ave'
        }
        quick_test(config_dict)

    # def test_gru4reckg(self):
    #     config_dict = {
    #         'model': 'GRU4RecKG',
    #     }
    #     quick_test(config_dict)

    # def test_s3rec(self):
    #     config_dict = {
    #         'model': 'S3Rec',
    #         'train_stage': 'pretrain',
    #         'save_step': 1,
    #     }
    #     quick_test(config_dict)
    #
    #     config_dict = {
    #         'model': 'S3Rec',
    #         'train_stage': 'finetune',
    #         'pre_model_path': './saved/S3Rec-test-1.pth',
    #     }
    #     quick_test(config_dict)


class TestKnowledgeRecommender(unittest.TestCase):

    def test_cke(self):
        config_dict = {
            'model': 'CKE',
        }
        quick_test(config_dict)

    def test_cfkg(self):
        config_dict = {
            'model': 'CFKG',
        }
        quick_test(config_dict)

    def test_cfkg_with_transe(self):
        config_dict = {
            'model': 'CFKG',
            'loss_function': 'transe',
        }
        quick_test(config_dict)

    def test_ktup(self):
        config_dict = {
            'model': 'KTUP',
            'train_rec_step': 1,
            'train_kg_step': 1,
            'epochs': 2,
        }
        quick_test(config_dict)

    def test_ktup_with_L1_flag(self):
        config_dict = {
            'model': 'KTUP',
            'use_st_gumbel': False,
            'L1_flag': True,
        }
        quick_test(config_dict)

    def test_kgat(self):
        config_dict = {
            'model': 'KGAT',
        }
        quick_test(config_dict)

    def test_kgat_with_gcn(self):
        config_dict = {
            'model': 'KGAT',
            'aggregator_type': 'gcn',
        }
        quick_test(config_dict)

    def test_kgat_with_graphsage(self):
        config_dict = {
            'model': 'KGAT',
            'aggregator_type': 'graphsage',
        }
        quick_test(config_dict)

    def test_ripplenet(self):
        config_dict = {
            'model': 'RippleNet',
        }
        quick_test(config_dict)

    def test_mkr(self):
        config_dict = {
            'model': 'MKR',
        }
        quick_test(config_dict)

    def test_mkr_without_use_inner_product(self):
        config_dict = {
            'model': 'MKR',
            'use_inner_product': False,
        }
        quick_test(config_dict)

    def test_kgcn(self):
        config_dict = {
            'model': 'KGCN',
        }
        quick_test(config_dict)

    def test_kgcn_with_neighbor(self):
        config_dict = {
            'model': 'KGCN',
            'aggregator': 'neighbor',
        }
        quick_test(config_dict)

    def test_kgcn_with_concat(self):
        config_dict = {
            'model': 'KGCN',
            'aggregator': 'concat',
        }
        quick_test(config_dict)

    def test_kgnnls(self):
        config_dict = {
            'model': 'KGNNLS',
        }
        quick_test(config_dict)

    def test_kgnnls_with_neighbor(self):
        config_dict = {
            'model': 'KGNNLS',
            'aggregator': 'neighbor',
        }
        quick_test(config_dict)

    def test_kgnnls_with_concat(self):
        config_dict = {
            'model': 'KGNNLS',
            'aggregator': 'concat',
        }
        quick_test(config_dict)


if __name__ == '__main__':
    unittest.main()
