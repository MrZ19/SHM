'''
@Time    : 10/12/20 4:28 PM
@Author  : Jiadai Sun
'''
import argparse
import os
from datetime import datetime

class IOStream:
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def init_args(args):
    # Add time suffix to exp name
    args.exp_name += datetime.now().strftime("_%Y%m%d_%H%M%S")

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    if not os.path.exists(os.path.join(args.checkpoint_dir, args.exp_name)):
        os.makedirs(os.path.join(args.checkpoint_dir, args.exp_name))

    if not os.path.exists(os.path.join(args.checkpoint_dir, args.exp_name, 'models')):
        os.makedirs(os.path.join(args.checkpoint_dir, args.exp_name, 'models'))

    os.system('cp main.py {}'.format(os.path.join(args.checkpoint_dir, args.exp_name,'main.py.backup' )))
    os.system('cp model.py {}'.format(os.path.join(args.checkpoint_dir, args.exp_name,'model.py.backup' )))
    os.system('cp data.py {}'.format(os.path.join(args.checkpoint_dir, args.exp_name,'data.py.backup' )))


def params_init():
    parser = argparse.ArgumentParser(description='Hard Partial Point Cloud Registration')

    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', metavar='N',
                        help='where to save the result of experiment')
    parser.add_argument('--exp_name', type=str, default='tmpexp', metavar='N',
                        help='Name of the experiment')

    parser.add_argument('--model', type=str, default='HPNet', metavar='N',
                        choices=['HPNet'], help='Model to use, [HPNet]')
    parser.add_argument('--emb_nn', type=str, default='dgcnn', metavar='N',
                        choices=['pointnet', 'dgcnn'], help='Embedding to use, [pointnet, dgcnn]')
    parser.add_argument('--pointer', type=str, default='transformer', metavar='N',
                        choices=['identity', 'transformer'], help='Head to use, [identity, transformer]')
    parser.add_argument('--head', type=str, default='svd', metavar='N',
                        choices=['mlp', 'svd'], help='Head to use, [mlp, svd]')
    parser.add_argument('--emb_dims', type=int, default=512, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--n_blocks', type=int, default=1, metavar='N',
                        help='Num of blocks of encoder&decoder')
    parser.add_argument('--n_heads', type=int, default=4, metavar='N',
                        help='Num of heads in multiheadedattention')
    parser.add_argument('--n_iters', type=int, default=3, metavar='N',
                        help='Num of iters to run inference')
    parser.add_argument('--discount_factor', type=float, default=0.9, metavar='N',
                        help='Discount factor to compute the loss')
    parser.add_argument('--ff_dims', type=int, default=1024, metavar='N',
                        help='Num of dimensions of fc in transformer')
    parser.add_argument('--n_keypoints', type=int, default=512, metavar='N',
                        help='Num of keypoints to use')
    parser.add_argument('--temp_factor', type=float, default=100, metavar='N',
                        help='Factor to control the softmax precision')
    parser.add_argument('--cat_sampler', type=str, default='gumbel_softmax', metavar='N',
                        choices=['softmax', 'gumbel_softmax'], help='use gumbel_softmax to get the categorical sample')
    parser.add_argument('--dropout', type=float, default=0.0, metavar='N',
                        help='Dropout ratio in transformer')
    parser.add_argument('--batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=8, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=False,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1234, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='evaluate the model')
    parser.add_argument('--num_workers', type=int, default=6, metavar='N',
                        help='number of works to DataLoader')

    parser.add_argument('--feature_alignment_loss', type=float, default=0.1, metavar='N',
                        help='feature alignment loss')
    parser.add_argument('--gaussian_noise', type=bool, default=False, metavar='N',
                        help='Wheter to add gaussian noise')
    parser.add_argument('--unseen', type=bool, default=False, metavar='N',
                        help='Wheter to test on unseen category')
    parser.add_argument('--n_points', type=int, default=1024, metavar='N',
                        help='Num of points to use')
    parser.add_argument('--n_subsampled_points', type=int, default=768, metavar='N',
                        help='Num of subsampled points to use')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40'], help='dataset to use')
    parser.add_argument('--rot_factor', type=float, default=4, metavar='N',
                        help='Divided factor of rotation')

    parser.add_argument('--resume', action='store_true', default=False,
                        help='continue training')
    parser.add_argument('--model_path', type=str, default='./checkpoints/exp/models/model.best.t7', metavar='N',
                        help='Pretrained model path')

    parser.add_argument('--finetune', action='store_true', default=False,
                        help='Whether to finetune')
    parser.add_argument('--tune_path', type=str, default='./train_models/pdsm.t7',
                        help='Model to use, finetune model path')

    args = parser.parse_args()

    return args
