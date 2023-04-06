# coding=utf-8

import torch
import torch.nn.functional as F
from PreHash_temp.src.models.DeepModel import DeepModel
from PreHash_temp.src.utils import utils
from PreHash_temp.src.utils.global_p import *


class PreHashWideDeep(DeepModel):
    include_id = False
    include_user_features = True
    include_item_features = True
    include_context_features = True
    data_loader = 'DataLoader'  # 默认data_loader
    data_processor = 'PreHashDP'  # 默认data_processor
    runner = 'BaseRunner'  # 默认runner

    @staticmethod
    def parse_model_args(parser, model_name='PreHash'):
        parser.add_argument('--hash_u_num', type=int, default=128,
                            help='Size of user hash.')
        parser.add_argument('--sample_max_n', type=int, default=128,
                            help='Sample top-n when learn hash.')
        parser.add_argument('--sample_r_n', type=int, default=128,
                            help='Sample random-n when learn hash.')
        parser.add_argument('--hash_layers', type=str, default='[32]',
                            help='MLP layer sizes of hash')
        parser.add_argument('--tree_layers', type=str, default='[64]',
                            help='Number of branches in each level of the hash tree')
        parser.add_argument('--transfer_att_size', type=int, default=16,
                            help='Size of attention layer of transfer layer (combine the hash and cf vector)')
        parser.add_argument('--cs_ratio', type=float, default=0.1,
                            help='Cold-Sampling ratio of each batch.')
        return DeepModel.parse_model_args(parser, model_name)

    def __init__(self, item_num, hash_u_num, hash_layers, tree_layers, transfer_att_size,
                 cs_ratio, sample_max_n, sample_r_n,
                 *args, **kwargs):
        self.item_num = item_num
        self.hash_u_num = hash_u_num
        self.hash_layers = hash_layers if type(hash_layers) == list else eval(hash_layers)
        self.tree_layers = tree_layers if type(tree_layers) == list else eval(tree_layers)
        self.transfer_att_size = transfer_att_size
        self.sample_max_n, self.sample_r_n = sample_max_n, sample_r_n
        self.cs_ratio = cs_ratio
        DeepModel.__init__(self, *args, **kwargs)

    def _init_weights(self):
        self.feature_embeddings = torch.nn.Embedding(self.feature_dims, self.f_vector_size)
        self.cross_bias = torch.nn.Embedding(self.feature_dims, 1)
        self.uid_embeddings = torch.nn.Embedding(1497021, self.f_vector_size)
        self.iid_embeddings = torch.nn.Embedding(self.item_num, self.f_vector_size)
        self.item_bias = torch.nn.Embedding(self.item_num, 1)
        self.l2_embeddings = ['feature_embeddings', 'cross_bias', 'uid_embeddings', 'iid_embeddings', 'item_bias']

        pre_size = self.f_vector_size
        for i, layer_size in enumerate(self.hash_layers):
            setattr(self, 'u_hash_%d' % i, torch.nn.Linear(pre_size, layer_size))
            pre_size = layer_size
        self.u_hash_predict = torch.nn.Linear(pre_size, self.hash_u_num + sum(self.tree_layers), bias=False)
        self.transfer_att_layer = torch.nn.Linear(self.f_vector_size, self.transfer_att_size)
        self.transfer_att_pre = torch.nn.Linear(self.transfer_att_size, 1, bias=False)

        pre_size = self.f_vector_size * (self.feature_num + 3)
        for i, layer_size in enumerate(self.layers):
            setattr(self, 'layer_%d' % i, torch.nn.Linear(pre_size, layer_size))
            setattr(self, 'bn_%d' % i, torch.nn.BatchNorm1d(layer_size))
            pre_size = layer_size
        self.prediction = torch.nn.Linear(pre_size, 1)

    def predict(self, feed_dict):
        check_list, embedding_l2 = [], []


        u_ids = feed_dict[UID]
        i_ids = feed_dict[IID]
        i_bias = self.item_bias(i_ids).view([-1])
        cf_i_vectors = self.iid_embeddings(i_ids.view([-1, 1]))
        embedding_l2.extend([cf_i_vectors, i_bias])

        total_batch_size = feed_dict[TOTAL_BATCH_SIZE]
        real_batch_size = feed_dict[REAL_BATCH_SIZE]

        # # history to hash_uid
        history = feed_dict[C_HISTORY]

        # print(type(history))
        # print(i_ids.shape)# [16384]
        # print(self.item_num, self.f_vector_size)# 166050, 64
        # assert 0
        # assert None------><class 'torch.Tensor'> <class 'numpy.int64'> <class 'int'>

        his_i_vectors = self.iid_embeddings(i_ids)

        if 'sparse' in str(history.type()):
            all_his_vector = history.mm(self.iid_embeddings.weight)
            if feed_dict[TRAIN]:
                # remove item i from history vectors
                if real_batch_size != total_batch_size:
                    padding_zeros = torch.zeros(size=[total_batch_size - real_batch_size, self.f_vector_size],
                                                dtype=torch.float32)
                    padding_zeros = utils.tensor_to_gpu(padding_zeros)
                    tmp_his_i_vectors = torch.cat([his_i_vectors[:real_batch_size], padding_zeros])
                else:
                    tmp_his_i_vectors = his_i_vectors
                his_vector = all_his_vector - tmp_his_i_vectors
                his_length = feed_dict[C_HISTORY_LENGTH] - 1
            else:
                his_vector = all_his_vector
                his_length = feed_dict[C_HISTORY_LENGTH]
            embedding_l2.append(his_vector)
            # normalize alpha = 0.5
            valid_his = his_length.gt(0).float()
            tmp_length = his_length.float() * valid_his + (1 - valid_his) * 1
            his_vector = his_vector / tmp_length.sqrt().view([-1, 1])
        else:
            valid_his = history.gt(0).long()  # Batch * His
            if feed_dict[TRAIN]:
                if_target_item = (history != i_ids.view([-1, 1])).long()
                valid_his = if_target_item * valid_his
            his_length = valid_his.sum(dim=1, keepdim=True)

            his_vectors = self.iid_embeddings(history * valid_his)  # Batch * His * v
            valid_his = valid_his.view([total_batch_size, -1, 1]).float()  # Batch * His * 1
            his_vectors = his_vectors * valid_his  # Batch * His * v
            his_att = (his_vectors * cf_i_vectors).sum(dim=-1, keepdim=True).exp() * valid_his  # Batch * His * 1
            his_att_sum = his_att.sum(dim=1, keepdim=True)  # Batch * 1 * 1
            his_att_weight = his_att / (his_att_sum + 1e-8)
            all_his_vector = (his_vectors * his_att_weight).sum(dim=1)  # Batch * 64
            his_vector = all_his_vector
            embedding_l2.append(his_vector)
            # normalize alpha = 0.5
            his_vector = his_vector * his_length.float().sqrt().view([-1, 1])


        # u_transfer_vectors=list[his_vector,u_embedings]
        u_embedings = self.uid_embeddings(u_ids)
        u_transfer_vectors = torch.cat([his_vector, u_embedings], dim=-1)       # (b,64)  (b,64)
        # 把u_transfer_vectors的size弄成大小为2维，第二个是-1

        if self.feature_num > 0:
            nonzero_embeddings = self.feature_embeddings(feed_dict[X])  # 输入特征：(b,21)   输出特征：(b,21,64)
            embedding_l2.append(nonzero_embeddings)
            pre_layer = nonzero_embeddings.view([-1, self.feature_num * self.f_vector_size])
            pre_layer = torch.cat([u_transfer_vectors, cf_i_vectors.view([-1, self.f_vector_size]), pre_layer], dim=1)      # 16384x1536    error:16384x1536 and 1472x64
        else:
            pre_layer = torch.cat([u_transfer_vectors, cf_i_vectors.view([-1, self.f_vector_size])], dim=1)



        for i in range(0, len(self.layers)):
            pre_layer = getattr(self, 'layer_%d' % i)(pre_layer)
            pre_layer = getattr(self, 'bn_%d' % i)(pre_layer)
            pre_layer = F.relu(pre_layer)
            pre_layer = torch.nn.Dropout(p=feed_dict[DROPOUT])(pre_layer)
        deep_prediction = self.prediction(pre_layer).view([-1])
        if self.feature_num > 0:
            cross_bias = self.cross_bias(feed_dict[X]).sum(dim=1).view([-1])
            embedding_l2.append(cross_bias)
        else:
            cross_bias = 0
        prediction = deep_prediction + cross_bias + i_bias
        out_dict = {PREDICTION: prediction, CHECK: check_list, EMBEDDING_L2: embedding_l2}
        return out_dict
