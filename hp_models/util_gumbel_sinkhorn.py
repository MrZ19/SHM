import torch
import numpy as np
import torch.nn as nn
from hp_models import my_sinkhorn_ops
from scipy.optimize import linear_sum_assignment


def Gumbel_sinkhorn_assignment(Matrix):
    # input: matching probability matrix ; [batchsize, N, N] ; dtype: torch.floattensor
    # output: assgnment matrix ; [batchsize, N, N]
    # Matrix = Matrix.detach()

    temperature, noise_factor = 1.0, 1.0
    samples_per_num = 1
    n_iter_sinkhorn = 5
    soft_perms_inf, log_alpha_w_noise = my_sinkhorn_ops.my_gumbel_sinkhorn(
                                            Matrix, temperature, samples_per_num,
                                                noise_factor, n_iter_sinkhorn, squeeze=False)
    # matching_pro = log_alpha_w_noise.squeeze(1)
    matching_pro = soft_perms_inf.squeeze(1)

    return matching_pro

    # matching_cost = -matching_pro
    # assign_matrix = []
    # [batchsize, N, N] = matching_pro.size()
    # for i in range(batchsize):
    #     matching_pro_temp = matching_pro[i].cpu().numpy()
    #     row_index, col_index = linear_sum_assignment(matching_pro_temp)
    #     M_temp = np.zeros((N,N))
    #     M_temp[row_index,col_index] = 1.0
    #     assign_matrix.append(np.expand_dims(M_temp,axis=0))
    # assign_matrix = np.concatenate(assign_matrix, axis=0) # numpy; [batchsize, N, N]
    # assign_matrix = torch.from_numpy(assign_matrix).cuda()
    #
    # return matching_pro, assign_matrix


def local_svd(pc1, pc2, M):
    # input:
    #   pc1: b, n, 3
    #   pc2: b, n, 3
    #   M: weight or matching matrix, b, n, n
    # output:
    #   R:b, 3, 3
    #   t:b, 3
    R, t = [], []
    W = torch.sum(M, dim=2)  # b,n
    pc2 = torch.matmul(M, pc2.transpose(2, 1)).transpose(2, 1)  # b,n,3

    for _pc1, _pc2, _W in zip(pc1, pc2, W):
        _pc1.requires_grad = False
        _pc2.requires_grad = False

        _R, _t = weighted_procrustes(X=_pc1, Y=_pc2, w=_W, eps=np.finfo(np.float32).eps)
        R.append(_R)
        t.append(_t)

    R = torch.stack(R, 0)
    t = torch.stack(t, 0)
    return R, t


def weighted_procrustes(X, Y, w, eps):
    """
    # https://ieeexplore.ieee.org/document/88573
    X: torch tensor N x 3
    Y: torch tensor N x 3
    w: torch tensor N
    """
    X = X.transpose(1, 0)
    Y = Y.transpose(1, 0)

    assert len(X) == len(Y)

    W1 = torch.abs(w).sum()
    w_norm = w / (W1 + eps)
    w_norm = w_norm.unsqueeze(1).repeat(1, 3)
    mux = (w_norm * X).sum(0, keepdim=True)
    muy = (w_norm * Y).sum(0, keepdim=True)

    # Use CPU for small arrays
    Sxy = (Y - muy).t().mm(w_norm * (X - mux)).cpu().double()
    U, D, V = Sxy.svd()
    S = torch.eye(3).double()
    if U.det() * V.det() < 0:
        S[-1, -1] = -1

    R = U.mm(S.mm(V.t())).float()
    t = (muy.cpu().squeeze() - R.mm(mux.cpu().t()).squeeze()).float()
    return R, t


def partial_weighted_sinkhorn(M, w_src, w_tgt, n_iters=5, eps=-1.0):
    # input:
    #   M:initial similarity matrix, b,n,n
    # output:
    #   pdsm: partial dsm b,n,n

    prev_alpha = None

    zero_pad = nn.ZeroPad2d((0, 1, 0, 1))
    log_alpha_padded = zero_pad(M[:, None, :, :])
    log_alpha_padded = torch.squeeze(log_alpha_padded, dim=1)
    w_src = w_src.unsqueeze(2).repeat(1, 1, M.shape[2]).cuda()
    w_tgt = w_tgt.unsqueeze(1).repeat(1, M.shape[1], 1).cuda()

    ones_pad = nn.ConstantPad2d((0, 1, 0, 1), 1.0)
    w_src_padded = ones_pad(w_src[:, None, :, :])
    w_src_padded = torch.squeeze(w_src_padded, dim=1)

    w_tgt_padded = ones_pad(w_tgt[:, None, :, :])
    w_tgt_padded = torch.squeeze(w_tgt_padded, dim=1)

    for i in range(n_iters):
        # Row normalization
        log_alpha_padded = torch.cat((
            (log_alpha_padded[:, :-1, :] - (torch.logsumexp(log_alpha_padded[:, :-1, :], dim=2, keepdim=True))).mul(
                w_src_padded[:, :-1, :]),
            log_alpha_padded[:, -1, None, :]),  # Don't normalize last row
            dim=1)

        # Column normalization
        log_alpha_padded = torch.cat((
            (log_alpha_padded[:, :, :-1] - (torch.logsumexp(log_alpha_padded[:, :, :-1], dim=1, keepdim=True))).mul(
                w_tgt_padded[:, :, :-1]),
            log_alpha_padded[:, :, -1, None]),  # Don't normalize last column
            dim=2)

        if eps > 0:
            if prev_alpha is not None:
                abs_dev = torch.abs(torch.exp(log_alpha_padded[:, :-1, :-1]) - prev_alpha)
                if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
                    break
            prev_alpha = torch.exp(log_alpha_padded[:, :-1, :-1]).clone()
    log_alpha = log_alpha_padded[:, :-1, :-1]

    return log_alpha


def partial_sinkhorn(log_alpha, n_iters=5, slack=True, eps=-1.0):
    """
    Args:
        log_alpha: log of positive matrix to apply sinkhorn normalization (B, J, K)
        n_iters (int): Number of normalization iterations
        slack (bool): Whether to include slack row and column
        eps: eps for early termination (Used only for handcrafted RPM). Set to negative to disable.

    Returns:
        log(perm_matrix): Doubly stochastic matrix (B, J, K)

    Modified from original source taken from:
        Learning Latent Permutations with Gumbel-Sinkhorn Networks
        https://github.com/HeddaCohenIndelman/Learning-Gumbel-Sinkhorn-Permutations-w-Pytorch
    """

    # Sinkhorn iterations
    prev_alpha = None
    if slack:
        zero_pad = nn.ZeroPad2d((0, 1, 0, 1))
        log_alpha_padded = zero_pad(log_alpha[:, None, :, :])

        log_alpha_padded = torch.squeeze(log_alpha_padded, dim=1)

        for i in range(n_iters):
            # Row normalization
            log_alpha_padded = torch.cat((
                log_alpha_padded[:, :-1, :] - (torch.logsumexp(log_alpha_padded[:, :-1, :], dim=2, keepdim=True)),
                log_alpha_padded[:, -1, None, :]),  # Don't normalize last row
                dim=1)

            # Column normalization
            log_alpha_padded = torch.cat((
                log_alpha_padded[:, :, :-1] - (torch.logsumexp(log_alpha_padded[:, :, :-1], dim=1, keepdim=True)),
                log_alpha_padded[:, :, -1, None]),  # Don't normalize last column
                dim=2)

            if eps > 0:
                if prev_alpha is not None:
                    abs_dev = torch.abs(torch.exp(log_alpha_padded[:, :-1, :-1]) - prev_alpha)
                    if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
                        break
                prev_alpha = torch.exp(log_alpha_padded[:, :-1, :-1]).clone()

        log_alpha = log_alpha_padded[:, :-1, :-1]
    else:
        for i in range(n_iters):
            # Row normalization (i.e. each row sum to 1)
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=2, keepdim=True))

            # Column normalization (i.e. each column sum to 1)
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=1, keepdim=True))

            if eps > 0:
                if prev_alpha is not None:
                    abs_dev = torch.abs(torch.exp(log_alpha) - prev_alpha)
                    if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
                        break
                prev_alpha = torch.exp(log_alpha).clone()

    return log_alpha


def DSM_to_permutation(partial_DSM):
    # input: partial_DSM
    # output: PPM partial_permutation_matrix
    # framework function
    M, m, n = threshold_concate(partial_DSM)
    partial_permutation_matrix = linear_assignment(M, m, n)

    return partial_permutation_matrix


def linear_assignment(Input_M, m, n):
    # input:
    #  Input_M: B, N+M, N+M
    # output:
    #  PPM: patical permutation matrix; B,N,N
    assign_matrix = []
    matching_cost = -Input_M
    batch_size, N, _ = matching_cost.size()

    for i in range(batch_size):
        matching_pro_temp = matching_cost[i].detach().cpu().numpy()
        row_index, col_index = linear_sum_assignment(matching_pro_temp)
        M_temp = np.zeros((N, N))
        M_temp[row_index, col_index] = 1.0
        assign_matrix.append(np.expand_dims(M_temp, axis=0))

    assign_matrix = np.concatenate(assign_matrix, axis=0)  # numpy; [batchsize, N, N]
    assign_matrix = torch.from_numpy(assign_matrix).cuda()

    PPM = assign_matrix[:, :m, :n]

    return PPM.float()


def threshold_concate(partial_DSM):
    # input:
    #  PDSM: patical DSM ; B,N,N
    # output:
    #  additive matrix
    #  m,n

    b, m, n = partial_DSM.shape
    std1 = torch.std(partial_DSM, dim=2)  # b, m
    std2 = torch.std(partial_DSM, dim=1)  # b, n

    std1 = threshold_set(std1, 1.0)
    std2 = threshold_set(std2, 1.0)

    ## make additive 
    # right up
    ru_I = torch.eye(m).unsqueeze(0).repeat(b, 1, 1).cuda()
    ru = torch.matmul(ru_I, std1.unsqueeze(2).repeat(1, 1, m))  # b,m,m

    # left down
    ld_I = torch.eye(n).unsqueeze(0).repeat(b, 1, 1).cuda()  # b,n,n
    ld = torch.matmul(ld_I, std2.unsqueeze(2).repeat(1, 1, n))  # b,n,n

    # right down
    rd = torch.zeros(b, n, m).cuda()

    ## concate
    M1 = torch.cat([partial_DSM, ru], dim=2)  # b,m,m+n
    M2 = torch.cat([ld, rd], dim=2)  # b,n,m+n
    M = torch.cat([M1, M2], dim=1)  # b,m+n,m+n

    return M, m, n


def threshold_set(std, k):
    # input:
    #  std: b,m
    # output:
    #  threshold: b,m
    ##k = k.cuda().float()
    temp = k / std  # b,m
    threshold = torch.tanh(temp)
    threshold = torch.ones_like(threshold) * 0.00005

    return threshold


def main():
    matching_matrix = np.random.rand(5, 3, 3)
    print("matching matrix : {}".format(matching_matrix))
    temperature = 1.0
    noise_factor = 1.0
    samples_per_num = 1
    n_iter_sinkhorn = 10
    matching_matrix = torch.from_numpy(matching_matrix).cuda().float()
    assign_matrix = Gumbel_sinkhorn_assignment(matching_matrix)
    print(assign_matrix.shape)
    print(assign_matrix)


if __name__ == "__main__":
    main()
