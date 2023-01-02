import torch
import numpy as np
import torch.nn.functional as F


def siamese_global_loss(src, tgt, Num, R_gt=None, t_gt=None):
    """
    Args:
        src: point cloud; B,3,num_points
        tgt: corresponding point cloud; B,3,num_points
        Num: the number of siamese structure; int
        R_gt:
        t_gt:

    Returns:
        loss: difference loss; float
    """
    src = src.cuda()
    tgt = tgt.cuda()

    R_gt = R_gt.cuda()
    t_gt = t_gt.cuda()

    batch_size = src.shape[0]
    num_points = src.shape[2]
    n = np.floor(num_points / Num).astype(int)
    print("n: {}".format(n))

    loss = 0
    identity = torch.eye(3).cuda()
    for i in range(batch_size):
        loss_i = 0
        src_one_batch = src[i]
        tgt_one_batch = tgt[i]  # 3,num_points
        R_gb = R_gt[i]
        t_gb = t_gt[i]
        # R_gb, t_gb = motion_estimate(src_one_batch, tgt_one_batch) #3,3;3
        s = torch.randperm(src_one_batch.shape[1])
        src_one_batch_perm = src_one_batch.transpose(1, 0)[s, :]  # 3,n -> n,3 -> ...
        tgt_one_batch_perm = tgt_one_batch.transpose(1, 0)[s, :]  # 3,n -> n,3 -> ...
        for j in range(Num):
            src_local = src_one_batch_perm[j * n:(j + 1) * n, :]  # n,3
            tgt_local = tgt_one_batch_perm[j * n:(j + 1) * n, :]  # n,3
            R_local, t_local = motion_estimate(src_local.transpose(1, 0), tgt_local.transpose(1, 0))
            loss_j = F.mse_loss(torch.matmul(R_local.transpose(1, 0), R_gb), identity) + F.mse_loss(t_local, t_gb)
            loss_i = loss_i + loss_j

        loss = loss + loss_i

    return loss


def motion_estimate(src, tgt):
    """
    Args:
        src: point cloud; 3,num_points
        tgt: corresponding point cloud; 3,num_points

    Returns:
        R: Rotation matrix; 3,3
        t: translation matrix; 3
    """
    src_centered = src - src.mean(dim=1, keepdim=True)  # 3,n
    tgt_centered = tgt - tgt.mean(dim=1, keepdim=True)  # 3,n
    H = torch.matmul(src_centered, tgt_centered.transpose(1, 0).contiguous()).cpu()

    u, s, v = torch.svd(H)
    r = torch.matmul(v, u.transpose(1, 0)).contiguous()
    r_det = torch.det(r).item()
    diag = torch.from_numpy(np.array([[1.0, 0, 0],
                                      [0, 1.0, 0],
                                      [0, 0, r_det]]).astype('float32')).to(v.device)
    R = torch.matmul(torch.matmul(v, diag), u.transpose(1, 0)).contiguous().cuda()
    t = torch.matmul(-R, src.mean(dim=1, keepdim=True)) + tgt.mean(dim=1, keepdim=True)

    return R, t


def matching_loss(pre, gt):
    """
    Args:
        pre: prediction matching matrix ; b,M,N
        gt: ground truth matching matrix; b,M,N
    Returns:
        loss:
    """
    pre = pre.cuda()
    gt = gt.cuda()
    loss_up = torch.sum(torch.mul(gt, pre))
    loss_down = torch.sum(gt)
    loss = -1.0 * loss_up / loss_down

    return loss


def mini_threshold(x, k):
    # input :
    #   pc: 3,n
    #   k: times
    x = torch.from_numpy(x).cuda()
    n = x.shape[1]
    inner = -2 * torch.matmul(x.transpose(1, 0).contiguous(), x)
    xx = torch.sum(x ** 2, dim=0, keepdim=True)
    distance_2 = xx + inner + xx.transpose(1, 0).contiguous()
    distance = torch.sqrt(torch.abs(distance_2).float())
    distance = distance + torch.eye(n).cuda() * 0.5

    distance_min = torch.min(distance)
    distance_max = torch.max(distance)
    print("max distance: {}, min distance: {}".format(distance_max, distance_min))
    threshold = distance_min * k
    return threshold


def correct_point_number(pc1, pc2, threshold):
    # input:
    #   pc1: n,3
    #   pc2: n,3 (pc1 and pc2 are aligned)
    #   threshold:
    pc1 = torch.from_numpy(pc1).cuda()
    pc2 = torch.from_numpy(pc2).cuda()
    distance = torch.sum((pc1 - pc2) ** 2, dim=1)  # n
    flag = distance < threshold
    number = torch.sum(flag)

    return number


def unsupervised_loss(src, tgt, scores, M, R, t):
    # input:
    #   src: b,n,3 
    #   target:b,n,3 
    #   scores:b,iter_n,n,n
    # output:
    #   loss
    R = torch.tensor(R).cuda()
    t = torch.tensor(t).cuda()
    scores = torch.tensor(scores).cuda()
    loss1 = inliers_loss(scores)
    loss2 = pcconsistent(src, tgt, scores, R, t)
    loss3 = Rtconsistent(src, tgt, 10)

    loss = loss1 + loss2 + loss3
    return loss


def inliers_loss(scores):
    loss1 = -torch.sum(scores)
    return loss1


def pcconsistent(src, tgt, scores, R, t):
    src = (torch.matmul(R, src) + t.unsqueeze(2).repeat(1, 1, src.shape[2])).transpose(2, 1)  # b,n,3
    W = torch.sum(scores, dim=2)  # b,n
    b = src.shape[0]
    tgt_perm = torch.matmul(scores, tgt.transpose(2, 1))  # b,n,3

    mse = torch.sum((src - tgt_perm) ** 2, dim=2)  # b,n
    mse_filter = mse.mul(W)
    number = torch.sum(W > 0.9)
    loss2 = torch.sum(mse_filter) * 1.0 / number
    return loss2


def Rtconsistent(src, tgt, Num):
    # input:
    #   scr: point cloud; B,3,num_points
    #   tgt: corresponding point cloud; B,3,num_points
    #   Num: the number of siamese structure; int
    # output:
    #   loss: difference loss; float

    src = src.cuda()
    tgt = tgt.cuda()

    batch_size = src.shape[0]
    num_points = src.shape[2]
    n = np.floor(num_points / Num).astype(int)
    print("n: {}".format(n))
    loss = 0
    identity = torch.eye(3).cuda()
    for i in range(batch_size):
        loss_i = 0
        src_one_batch = src[i]
        tgt_one_batch = tgt[i]  # 3,num_points
        R_global, t_global = motion_estimate(src_one_batch, tgt_one_batch)
        # R_gb, t_gb = motion_estimate(src_one_batch, tgt_one_batch) #3,3;3
        s = torch.randperm(src_one_batch.shape[1])
        src_one_batch_perm = src_one_batch.transpose(1, 0)[s, :]  # 3,n -> n,3 -> ...
        tgt_one_batch_perm = tgt_one_batch.transpose(1, 0)[s, :]  # 3,n -> n,3 -> ...
        for j in range(Num):
            src_local = src_one_batch_perm[j * n:(j + 1) * n, :]  # n,3
            tgt_local = tgt_one_batch_perm[j * n:(j + 1) * n, :]  # n,3
            R_local, t_local = motion_estimate(src_local.transpose(1, 0), tgt_local.transpose(1, 0))
            loss_j = F.mse_loss(torch.matmul(R_local.transpose(1, 0), R_global), identity) + F.mse_loss(t_local,
                                                                                                        t_global)
            loss_i = loss_i + loss_j

        loss = loss + loss_i

    return loss

