import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import pdb

class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, dim_Inner, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)

        self.fc_o_0 = nn.Linear(dim_V, dim_V)
        self.fc_o = nn.Linear(dim_V, dim_Inner)
        self.fc_o_2 = nn.Linear(dim_Inner, dim_V)

    def forward(self, Q, K, mask_pos=None, knn_masks=None):
        forward = self.forward_op
        return forward(Q, K, mask_pos, knn_masks)

    def forward_op(self, Q, K, mask_pos=None, knn_masks=None):

        self.fuck_mask = None
        assert mask_pos is not None

        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)


        # given, masked, all
        K_given = K_[:, :mask_pos, :]
        A_logits_given = Q_.bmm(K_given.transpose(1,2))/math.sqrt(self.dim_V)

        # Q_masked, K_masked = Q_[:, mask_pos:, :], K_[:, mask_pos:, :]
        # fb_size, p_num, e_size = Q_masked.shape
        # Q_masked, K_masked = Q_masked.reshape((-1, 1, e_size)), K_masked.reshape((-1, e_size, 1))

        if knn_masks is not None:
            # _knn_masks = knn_masks.repeat(self.num_heads, 1, 1).transpose(1,2)
            A_logits_given += knn_masks

        A_logits_given_ = torch.softmax(A_logits_given, 2)

        # A_given_given, A_masked_given = A_logits_given[:, :mask_pos,:], A_logits_given[:, mask_pos:,:]
        # A_given_given = torch.softmax(A_given_given, 2)

        # V_given  = A_given_given.bmm(V_[:, :mask_pos, :])


        # A_masked_ = A_masked_given
        # A_masked_ = torch.softmax(A_masked_, 2)
        # V_masked = A_masked_.bmm(V_[:, :mask_pos, :])

        # self.fuck_attn = A_masked_
        # A_masked_ = self.dropout_2(A_masked_)

        # V__ = torch.cat([V_given, V_masked], axis=1)

        V__ = A_logits_given_.bmm(V_[:, :mask_pos, :])

        V__ = torch.cat((V__).split(Q.size(0), 0), 2)
        V__ = self.fc_o_0(V__)
        # pdb.set_trace()

        O = Q + V__

        # V__ = self.dropout_1(V__)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)

        fc_O = self.fc_o_2(F.relu(self.fc_o(O)))
        # fc_O = self.dropout_2(fc_O)
        O = O + fc_O
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O


class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, d_inner, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, d_inner, num_heads, ln=ln)

    def forward(self, X, mask_pos=None, knn_masks=None):
        return self.mab(X, X, mask_pos=mask_pos, knn_masks=knn_masks)


class MLPWeight(nn.Module):
    def __init__(self, dim_in, dim_hidden, d_inner, dim_out):
        super(MLPWeight, self).__init__()

        self.linear0 = nn.Linear(dim_in, dim_hidden)
        self.linear1 = nn.Linear(dim_hidden, dim_hidden)
        self.linear2 = nn.Linear(dim_hidden, dim_out)


    def forward(self, in_v):

        in_v = F.gelu(self.linear0(in_v))
        in_v = F.gelu(self.linear1(in_v))
        in_v = self.linear2(in_v)
        return in_v


class NIERT(nn.Module):
    def __init__ (
        self,
        cfg_dim_input=3,
        cfg_dim_output=-1,
        d_model=128,
        d_inner=512,
        n_layers=2,
        # n_head=8,
        n_head=4,
        d_k=128,
        d_v=128,
    ):
        super().__init__()

        if cfg_dim_output == -1:
            cfg_dim_output = 1
        
        self.cfg_dim_input = cfg_dim_input
        self.cfg_dim_output = cfg_dim_output

        self.linearl = nn.Linear(cfg_dim_input-cfg_dim_output,16*(cfg_dim_input-cfg_dim_output))
        self.linearr = nn.Linear(cfg_dim_output,16*cfg_dim_output)

        self.mask_embedding = nn.Parameter(torch.zeros(cfg_dim_output*16))
        self.selfatt = nn.ModuleList()
        self.selfatt1 = SAB(16*cfg_dim_input, d_model, d_inner, n_head, ln=True)

        for i in range(n_layers-1):
            self.selfatt.append(SAB(d_model, d_model, d_inner, n_head, ln=True))

        self.outatt = MLPWeight(d_model, d_model, d_model, cfg_dim_output)


    def forward(self, o_x, o_y, t_x, knn_masks, placeholder_):

        mask_pos = o_x.size(1)

        given_x, given_y = self.linearl(o_x), self.linearr(o_y)
        given_xy = torch.cat((given_x, given_y), dim=-1)

        target_x = self.linearl(t_x)
        target_y = self.mask_embedding.view((1,1,self.cfg_dim_output * 16)).expand((target_x.size(0), target_x.size(1), self.cfg_dim_output * 16))
        target_xy = torch.cat((target_x, target_y), dim=-1)
        given_target_xy = torch.cat((given_xy, target_xy), dim=1)

        given_target_xy = self.selfatt1(given_target_xy, mask_pos=mask_pos, knn_masks=knn_masks)

        for layer in self.selfatt:
            given_target_xy = layer(given_target_xy, mask_pos=mask_pos, knn_masks=knn_masks)

        predict_y = self.outatt(given_target_xy)

        # print(given_target_xy.shape)
        # return predict_y, target_y, mask_pos
        # return predict_y.unsqueeze(-1), label_y.unsqueeze(-1), None


        if len(predict_y.shape) == 2:
            return predict_y.unsqueeze(-1)
        elif len(predict_y.shape) == 3:
            return predict_y
        else:
            assert False



class HINT(nn.Module):
    def __init__ (
        self,
        cfg_dim_input=3,            # dx + dy
        cfg_dim_output=-1,          # dy
        d_model=128,
        d_inner=512,
        n_layers=2,                 # # attention layers of main block
        n_head=4,
        d_k=128,
        d_v=128,
        K_0_inv=None,                   # K0 = n / K0_inv
        K_min_inv=None,                 # K_min = n / K_min_inv
        K_min=None,                     # default K_min
        n_blocks=4                  # # interpolation blocks
    ):
        super().__init__()

        self.L = n_blocks
        self.K_0_inv = K_0_inv
        self.K_min_inv = K_min_inv
        self.K_min = K_min

        self.neirt_layers = nn.ModuleList([
            NIERT(
                cfg_dim_input,
                cfg_dim_output,
                d_model,
                d_inner,
                n_layers if l == 0 else 2,
                n_head,
                d_k,
                d_v
            )
            for l in range(self.L)
        ])

        self.num_heads = n_head


    def forward(self, o_x, o_y, t_x, t_y, placeholder=None):

        # import pdb; pdb.set_trace()
        batch_size, o_num = o_x.size(0), o_x.size(1)

        o_y_residuals = o_y

        ret = {
            "o": [],
            "o_predict": [],
        }

        KNN_MASKS = [None for _ in range(self.L)]

        K = o_x.size(1) // self.K_0_inv

        ot_x = torch.cat((o_x, t_x), dim=1)

        for layer_id in range(self.L):

            dist_mat = torch.cdist(o_x, ot_x, p=2)
            _, indices = torch.topk(dist_mat, K, dim=1, largest=False, sorted=True)
            knn_mask = -math.inf * torch.ones(batch_size, o_x.size(1), ot_x.size(1), device=indices.device)
            knn_mask = knn_mask.scatter(1, indices, 0)

            _knn_mask = knn_mask.repeat(self.num_heads, 1, 1).transpose(1,2)

            KNN_MASKS[layer_id] = _knn_mask

            if layer_id == 0:
                if self.K_min != -1:
                    K = max(K // 2, self.K_min)
                else:
                    K = max(K // 2, o_x.size(1) // self.K_min_inv)


        for layer_id in range(self.L):

            niert_layer = self.neirt_layers[layer_id]

            o_t_predict_y = niert_layer(o_x, o_y_residuals, t_x, KNN_MASKS[layer_id], None)

            ret["o"].append(o_y_residuals)
            ret["o_predict"].append(o_t_predict_y[:,:o_num])

            o_y_residuals = o_y_residuals - o_t_predict_y[:, -o_num:]

            if layer_id == 0:
                interpolation = o_t_predict_y[:, o_num:-o_num]
            else:
                interpolation += o_t_predict_y[:, o_num:-o_num]

        return interpolation, t_y[:,:-o_num], ret

