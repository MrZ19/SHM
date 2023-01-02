#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function
import torch
import numpy as np
from scipy.spatial.transform import Rotation


# Part of the code is referred from: https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/inverse_warp.py

def quat2mat(quat):
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,
                          2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
                          2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat


def transform_point_cloud(point_cloud, rotation, translation):
    if len(rotation.size()) == 2:
        rot_mat = quat2mat(rotation)
    else:
        rot_mat = rotation
    return torch.matmul(rot_mat, point_cloud) + translation.unsqueeze(2)


def npmat2euler(mats, seq='zyx'):
    eulers = []
    for i in range(mats.shape[0]):
        r = Rotation.from_dcm(mats[i])
        eulers.append(r.as_euler(seq, degrees=True))
    return np.asarray(eulers, dtype='float32')


def textio_cprint_and_boadio_add_scalar(
        boardio, textio, epoch, best_epoch, train_loss, test_loss, best_test_loss,
        train_mse, train_rmse, train_mae, train_r_mse, train_r_rmse, train_r_mae, train_t_mse, train_t_rmse,
        train_t_mae,
        test_mse, test_rmse, test_mae, test_r_mse, test_r_rmse, test_r_mae, test_t_mse, test_t_rmse, test_t_mae,
        best_test_mse, best_test_rmse, best_test_mae, best_test_r_mse, best_test_r_rmse, best_test_r_mae,
        best_test_t_mse, best_test_t_rmse, best_test_t_mae):
    textio.cprint('==TRAIN==')
    textio.cprint('EPOCH:: %d, Loss: %f, MSE: %f, RMSE: %f, MAE: %f, rot_MSE: %f, rot_RMSE: %f, '
                  'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
                  % (epoch, train_loss, train_mse, train_rmse, train_mae, train_r_mse,
                     train_r_rmse, train_r_mae, train_t_mse, train_t_rmse, train_t_mae))

    textio.cprint('==TEST==')
    textio.cprint('EPOCH:: %d, Loss: %f, MSE: %f, RMSE: %f, MAE: %f, rot_MSE: %f, rot_RMSE: %f, '
                  'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
                  % (epoch, test_loss, test_mse, test_rmse, test_mae, test_r_mse,
                     test_r_rmse, test_r_mae, test_t_mse, test_t_rmse, test_t_mae))

    textio.cprint('==BEST TEST==')
    textio.cprint('EPOCH:: %d, Loss: %f, MSE: %f, RMSE: %f, MAE: %f, rot_MSE: %f, rot_RMSE: %f, '
                  'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
                  % (best_epoch, best_test_loss, best_test_mse, best_test_rmse, best_test_mae, best_test_r_mse,
                     best_test_r_rmse,
                     best_test_r_mae, best_test_t_mse, best_test_t_rmse, best_test_t_mae))

    # Train
    boardio.add_scalar('train/loss', train_loss, epoch)
    boardio.add_scalar('train/MSE', train_mse, epoch)
    boardio.add_scalar('train/RMSE', train_rmse, epoch)
    boardio.add_scalar('train/MAE', train_mae, epoch)
    boardio.add_scalar('train/rotation/MSE', train_r_mse, epoch)
    boardio.add_scalar('train/rotation/RMSE', train_r_rmse, epoch)
    boardio.add_scalar('train/rotation/MAE', train_r_mae, epoch)
    boardio.add_scalar('train/translation/MSE', train_t_mse, epoch)
    boardio.add_scalar('train/translation/RMSE', train_t_rmse, epoch)
    boardio.add_scalar('train/translation/MAE', train_t_mae, epoch)

    # TEST
    boardio.add_scalar('test/loss', test_loss, epoch)
    boardio.add_scalar('test/MSE', test_mse, epoch)
    boardio.add_scalar('test/RMSE', test_rmse, epoch)
    boardio.add_scalar('test/MAE', test_mae, epoch)
    boardio.add_scalar('test/rotation/MSE', test_r_mse, epoch)
    boardio.add_scalar('test/rotation/RMSE', test_r_rmse, epoch)
    boardio.add_scalar('test/rotation/MAE', test_r_mae, epoch)
    boardio.add_scalar('test/translation/MSE', test_t_mse, epoch)
    boardio.add_scalar('test/translation/RMSE', test_t_rmse, epoch)
    boardio.add_scalar('test/translation/MAE', test_t_mae, epoch)

    # BEST TEST
    boardio.add_scalar('best_test/loss', best_test_loss, epoch)
    boardio.add_scalar('best_test/MSE', best_test_mse, epoch)
    boardio.add_scalar('best_test/RMSE', best_test_rmse, epoch)
    boardio.add_scalar('best_test/MAE', best_test_mae, epoch)
    boardio.add_scalar('best_test/rotation/MSE', best_test_r_mse, epoch)
    boardio.add_scalar('best_test/rotation/RMSE', best_test_r_rmse, epoch)
    boardio.add_scalar('best_test/rotation/MAE', best_test_r_mae, epoch)
    boardio.add_scalar('best_test/translation/MSE', best_test_t_mse, epoch)
    boardio.add_scalar('best_test/translation/RMSE', best_test_t_rmse, epoch)
    boardio.add_scalar('best_test/translation/MAE', best_test_t_mae, epoch)
