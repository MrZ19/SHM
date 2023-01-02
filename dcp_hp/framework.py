"""
framework: train and test function
@Time    :
@Author  : Zhiyuan Zhang & Jiadai Sun
"""

import os
import gc
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
from util import transform_point_cloud, npmat2euler, textio_cprint_and_boadio_add_scalar  # , visualize_trans_result
from loss_function import matching_loss


def train(args, net, train_loader, test_loader, boardio, textio):
    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(net.parameters(), lr=args.lr * 100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = MultiStepLR(opt, milestones=[75, 150, 200], gamma=0.1)

    # Some variables for comparison
    best_test_loss, best_test_mse, best_test_rmse, best_test_mae = np.inf, np.inf, np.inf, np.inf
    best_test_r_mse, best_test_r_rmse, best_test_r_mae = np.inf, np.inf, np.inf
    best_test_t_mse, best_test_t_rmse, best_test_t_mae = np.inf, np.inf, np.inf
    # best_test_mse_ba, best_test_rmse_ba, best_test_mae_ba = np.inf, np.inf, np.inf
    best_epoch = -1

    for epoch in range(args.epochs):
        scheduler.step()

        train_loss, train_mse, train_mae, \
        train_rotations, train_translations, \
        train_rotations_pred, train_translations_pred, \
        train_eulers = train_one_epoch(args, net, train_loader, opt)

        test_loss, test_mse, test_mae, \
        test_rotations, test_translations, \
        test_rotations_pred, test_translations_pred, \
        test_eulers = test_one_epoch(args, net, test_loader)

        train_rmse = np.sqrt(train_mse)
        test_rmse = np.sqrt(test_mse)

        # Train `mse mas rmse` about rotation and translation
        train_rotations_pred_euler = npmat2euler(train_rotations_pred)
        train_r_mse = np.mean((train_rotations_pred_euler - np.degrees(train_eulers)) ** 2)
        train_r_mae = np.mean(np.abs(train_rotations_pred_euler - np.degrees(train_eulers)))
        train_r_rmse = np.sqrt(train_r_mse)
        train_t_mse = np.mean((train_translations - train_translations_pred) ** 2)
        train_t_mae = np.mean(np.abs(train_translations - train_translations_pred))
        train_t_rmse = np.sqrt(train_t_mse)

        # Test `mse mas rmse` about rotation and translation
        test_rotations_pred_euler = npmat2euler(test_rotations_pred)
        test_r_mse = np.mean((test_rotations_pred_euler - np.degrees(test_eulers)) ** 2)
        test_r_mae = np.mean(np.abs(test_rotations_pred_euler - np.degrees(test_eulers)))
        test_r_rmse = np.sqrt(test_r_mse)
        test_t_mse = np.mean((test_translations - test_translations_pred) ** 2)
        test_t_mae = np.mean(np.abs(test_translations - test_translations_pred))
        test_t_rmse = np.sqrt(test_t_mse)

        # Save the model for each epoch
        torch.save(net.module.state_dict() if torch.cuda.device_count() > 1 else net.state_dict(),
                   os.path.join(args.checkpoint_dir, args.exp_name, 'models/model.%d.t7' % (epoch)))

        # Compare the loss of testing to save better model
        if best_test_loss >= test_loss:
            best_test_loss = test_loss
            best_test_mse, best_test_rmse, best_test_mae = test_mse, test_rmse, test_mae
            best_test_r_mse, best_test_r_rmse, best_test_r_mae = test_r_mse, test_r_rmse, test_r_mae
            best_test_t_mse, best_test_t_rmse, best_test_t_mae = test_t_mse, test_t_rmse, test_t_mae
            best_epoch = epoch
            # Save the current best model
            torch.save(net.module.state_dict() if torch.cuda.device_count() > 1 else net.state_dict(),
                       os.path.join(args.checkpoint_dir, args.exp_name, 'models/model.best.t7'))

        # Save running log by text print and tensorboard
        textio_cprint_and_boadio_add_scalar(boardio, textio, epoch, best_epoch, train_loss, test_loss, best_test_loss,
                                            train_mse, train_rmse, train_mae, train_r_mse, train_r_rmse, train_r_mae,
                                            train_t_mse, train_t_rmse, train_t_mae,
                                            test_mse, test_rmse, test_mae, test_r_mse, test_r_rmse, test_r_mae,
                                            test_t_mse, test_t_rmse, test_t_mae,
                                            best_test_mse, best_test_rmse, best_test_mae, best_test_r_mse,
                                            best_test_r_rmse, best_test_r_mae, best_test_t_mse, best_test_t_rmse,
                                            best_test_t_mae)

        gc.collect()


def test(args, net, test_loader, boardio, textio):
    with torch.no_grad():
        test_loss, test_mse, test_mae, \
        test_rotations, test_translations, \
        test_rotations_pred, test_translations_pred, \
        test_eulers = test_one_epoch(args, net, test_loader)

    test_rmse = np.sqrt(test_mse)

    test_rotations_pred_euler = npmat2euler(test_rotations_pred)
    test_r_mse = np.mean((test_rotations_pred_euler - np.degrees(test_eulers)) ** 2)
    test_r_mae = np.mean(np.abs(test_rotations_pred_euler - np.degrees(test_eulers)))
    test_r_rmse = np.sqrt(test_r_mse)

    test_t_mse = np.mean((test_translations - test_translations_pred) ** 2)
    test_t_mae = np.mean(np.abs(test_translations - test_translations_pred))
    test_t_rmse = np.sqrt(test_t_mse)

    textio.cprint('==FINAL TEST==')
    textio.cprint('EPOCH:: %d, Loss: %f, MSE: %f, RMSE: %f, MAE: %f, rot_MSE: %f, rot_RMSE: %f, '
                  'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
                  % (-1, test_loss, test_mse, test_rmse, test_mae, test_r_mse, test_r_rmse, test_r_mae, test_t_mse,
                     test_t_rmse, test_t_mae))


def train_one_epoch(args, net, train_loader, opt):
    net.train()

    total_loss, mse, mae, num_examples = 0, 0, 0, 0
    rotations, translations, eulers, = [], [], []
    rotations_pred, translations_pred = [], []

    for src, target, rotation, translation, euler, I_gt in tqdm(train_loader):
        src = src.cuda()
        target = target.cuda()
        rotation = rotation.cuda()
        translation = translation.cuda()
        I_gt = I_gt.cuda()
        batch_size = src.size(0)
        num_examples += batch_size

        opt.zero_grad()

        rotation_pred, translation_pred, scores_pred = net(src, target)

        # save rotation and translation
        rotations.append(rotation.detach().cpu().numpy())
        translations.append(translation.detach().cpu().numpy())
        eulers.append(euler.numpy())
        rotations_pred.append(rotation_pred.detach().cpu().numpy())
        translations_pred.append(translation_pred.detach().cpu().numpy())

        transformed_src = transform_point_cloud(src, rotation_pred, translation_pred)
        # transformed_target = transform_point_cloud(target, rotation_pred, translation_pred)

        ######### Rot. and trans. ######
        # identity = torch.eye(3).cuda().unsqueeze(0).repeat(batch_size, 1, 1)
        # loss = F.mse_loss(torch.matmul(rotation_pred.transpose(2, 1), rotation), identity) \
        #       + F.mse_loss(translation_pred, translation)
        ######### matching loss #####
        mat_loss = matching_loss(scores_pred, I_gt)
        # mse_loss = F.mse_loss(scores_pred, I_gt)

        ######### Final loss ########
        loss_final = mat_loss - torch.mean(scores_pred)
        loss_final.backward()

        opt.step()

        total_loss += loss_final.item() * batch_size
        mse += torch.mean((transformed_src - target) ** 2, dim=[0, 1, 2]).item() * batch_size
        mae += torch.mean(torch.abs(transformed_src - target), dim=[0, 1, 2]).item() * batch_size

    # Aggregate the results of all batches to return
    rotations = np.concatenate(rotations, axis=0)
    translations = np.concatenate(translations, axis=0)
    eulers = np.concatenate(eulers, axis=0)
    rotations_pred = np.concatenate(rotations_pred, axis=0)
    translations_pred = np.concatenate(translations_pred, axis=0)

    return total_loss * 1.0 / num_examples, \
           mse * 1.0 / num_examples, \
           mae * 1.0 / num_examples, \
           rotations, translations, rotations_pred, translations_pred, eulers


def test_one_epoch(args, net, test_loader):
    net.eval()

    total_loss, mse, mae, num_examples = 0, 0, 0, 0
    rotations, translations, eulers, = [], [], []
    rotations_pred, translations_pred = [], []

    for src, target, rotation, translation, euler, I_gt in tqdm(test_loader):
        src = src.cuda()
        target = target.cuda()
        rotation = rotation.cuda()
        translation = translation.cuda()
        I_gt = I_gt.cuda()
        batch_size = src.size(0)
        num_examples += batch_size

        rotation_pred, translation_pred, scores_pred = net(src, target)

        # save rotation and translation
        rotations.append(rotation.detach().cpu().numpy())
        translations.append(translation.detach().cpu().numpy())
        rotations_pred.append(rotation_pred.detach().cpu().numpy())
        translations_pred.append(translation_pred.detach().cpu().numpy())
        eulers.append(euler.numpy())

        transformed_src = transform_point_cloud(src, rotation_pred, translation_pred)
        # transformed_target = transform_point_cloud(target, rotation_pred, translation_pred)
        # if args.eval:
        #    visualize_trans_result(src, transformed_src, target, virtual_corr_points_ab, scores_ab_pred, \
        #        rotation_ab_pred, translation_ab_pred)

        ######### Rot. and trans. ######
        # identity = torch.eye(3).cuda().unsqueeze(0).repeat(batch_size, 1, 1)
        # loss = F.mse_loss(torch.matmul(rotation_pred.transpose(2, 1), rotation), identity) \
        #       + F.mse_loss(translation_pred, translation)
        ######### Final loss ########
        ######### matching loss #####
        mat_loss = matching_loss(scores_pred, I_gt) - torch.mean(scores_pred)
        # mse_loss = F.mse_loss(scores_pred, I_gt)
        loss_final = mat_loss

        total_loss += loss_final.item() * batch_size
        mse += torch.mean((transformed_src - target) ** 2, dim=[0, 1, 2]).item() * batch_size
        mae += torch.mean(torch.abs(transformed_src - target), dim=[0, 1, 2]).item() * batch_size

    # Aggregate the results of all batches to return
    rotations = np.concatenate(rotations, axis=0)
    translations = np.concatenate(translations, axis=0)
    eulers = np.concatenate(eulers, axis=0)
    rotations_pred = np.concatenate(rotations_pred, axis=0)
    translations_pred = np.concatenate(translations_pred, axis=0)

    return total_loss * 1.0 / num_examples, \
           mse * 1.0 / num_examples, \
           mae * 1.0 / num_examples, \
           rotations, translations, rotations_pred, translations_pred, eulers
