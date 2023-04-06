# coding=utf-8

import torch
import torch.nn.functional as F
from PreHash_temp.src.models.DeepModel import DeepModel
from PreHash_temp.src.utils import utils
from PreHash_temp.src.utils.global_p import *


class LSTMWideDeep(DeepModel):
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
        self.lstm_layer = torch.nn.LSTM(input_size=self.f_vector_size, hidden_size=self.f_vector_size, num_layers=1,
                                   bias=True, batch_first=True, dropout=0.3, bidirectional=False)

    def predict(self, feed_dict):
        check_list, embedding_l2 = [], []



        u_ids = feed_dict[UID]
        i_ids = feed_dict[IID]
        # print(type(i_ids))
        # assert 0----><class 'torch.Tensor'>
        i_bias = self.item_bias(i_ids).view([-1])
        cf_i_vectors = self.iid_embeddings(i_ids.view([-1, 1]))
        embedding_l2.extend([cf_i_vectors, i_bias])

        total_batch_size = feed_dict[TOTAL_BATCH_SIZE]
        real_batch_size = feed_dict[REAL_BATCH_SIZE]


        # # history to hash_uid
        history = feed_dict[C_HISTORY]
        # print(type(history))#tensor
        #
        # (indices=tensor([[    0,     0,     0,  ..., 16382, 16382, 16383],
        #                        [   38,  1120, 24811,  ..., 11910, 34577,  2189]]),
        #        values=tensor([1., 1., 1.,  ..., 1., 1., 1.]),
        #        size=(16384, 166050), nnz=173664, layout=torch.sparse_coo)

        # print(type(self.item_num), type(self.f_vector_size))<class 'torch.Tensor'> <class 'numpy.int64'> <class 'int'>

        # # print(i_ids.shape)
        # print("-----------------------")
        # print(self.item_num, self.f_vector_size)#166050 64
        # print(history.shape)#[16384,166050]
        # print(type(history))<class 'torch.Tensor'>
        # print("-----------------------")
        # assert 0

        history = history.mm(self.iid_embeddings.weight)
        history = torch.tensor(history).to(torch.int64)#[16384, 64]

        h_0 = torch.zeros(real_batch_size, 1, self.f_vector_size)
        c_0 = torch.zeros(real_batch_size, 1, self.f_vector_size)
        #(16384, 1, 64)
        history = history.unsqueeze(1)

        his_seq_vectors, (h_n, c_n) = self.lstm_layer(history.float())
        his_vector = his_seq_vectors.squeeze(1)






        # u_transfer_vectors=list[his_vector,u_embedings]
        u_embedings = self.uid_embeddings(u_ids)
        u_transfer_vectors = torch.cat([his_vector, u_embedings], dim=-1)
        # 把u_transfer_vectors的size弄成大小为2维，第二个是-1
        if self.feature_num > 0:
            nonzero_embeddings = self.feature_embeddings(feed_dict[X].to(torch.int64))
            embedding_l2.append(nonzero_embeddings)
            pre_layer = nonzero_embeddings.view([-1, self.feature_num * self.f_vector_size])
            pre_layer = torch.cat([u_transfer_vectors, cf_i_vectors.view([-1, self.f_vector_size]), pre_layer], dim=1)
        else:
            pre_layer = torch.cat([u_transfer_vectors, cf_i_vectors.view([-1, self.f_vector_size])], dim=1)





        for i in range(0, len(self.layers)):
            pre_layer = getattr(self, 'layer_%d' % i)(pre_layer)
            pre_layer = getattr(self, 'bn_%d' % i)(pre_layer)
            pre_layer = F.relu(pre_layer)
            pre_layer = torch.nn.Dropout(p=feed_dict[DROPOUT])(pre_layer)#[16384.64]
        deep_prediction = self.prediction(pre_layer).view([-1])#16384
        if self.feature_num > 0:
            # print((feed_dict[X]).sum(dim=1).view([-1]).to(torch.int64).size)
            # print(type((feed_dict[X]).sum(dim=1).view([-1]).to(torch.int64)))#(feed_dict[X])[16384,21]
            cross_bias = self.cross_bias(feed_dict[X].to(torch.int64)).sum(dim=1).view([-1])
            embedding_l2.append(cross_bias)
        else:
            cross_bias = 0
        prediction = deep_prediction + cross_bias + i_bias
        out_dict = {PREDICTION: prediction,
                    CHECK: check_list, EMBEDDING_L2: embedding_l2}
        return out_dict
