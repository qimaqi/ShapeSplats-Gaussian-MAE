import os
import argparse
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='yaml config file')
    parser.add_argument(
        '--launcher', choices=['none', 'pytorch'], default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--exp_name', type=str,
                        default='default', help='experiment name')
    parser.add_argument('--loss', type=str, default='cd1', help='loss name')
    parser.add_argument('--start_ckpts', type=str,
                        default=None, help='reload used ckpt path')
    parser.add_argument('--ckpts', type=str, default=None,
                        help='test used ckpt path')
    parser.add_argument('--val_freq', type=int, default=1, help='test freq')
    parser.add_argument('--soft_knn', action='store_true',
                        default=True, help='use soft knn in Encoder')
    parser.add_argument('--data_path', type=str, default=None)  # dataset path
    parser.add_argument('--gs_path', type=str,
                        default=None)    # gs dataset path
    parser.add_argument('--total_bs', type=int, default=0)
    parser.add_argument('--way', type=int, default=-1)
    parser.add_argument('--shot', type=int, default=-1)
    parser.add_argument('--fold', type=int, default=-1)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--sync_bn',
        action='store_true',
        default=False,
        help='whether to use sync bn')
    parser.add_argument(
        '--vote',
        action='store_true',
        default=False,
        help='vote acc')
    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='autoresume training (interrupted by accident)')
    parser.add_argument(
        '--test',
        action='store_true',
        default=False,
        help='test mode for certain ckpt')
    parser.add_argument(
        '--finetune_model',
        action='store_true',
        default=False,
        help='finetune modelnet with pretrained weight')
    parser.add_argument(
        '--scratch_model',
        action='store_true',
        default=False,
        help='training modelnet from scratch')
    parser.add_argument(
        '--mode',
        choices=['easy', 'median', 'hard', None],
        default=None,
        help='difficulty mode for shapenet')
    args = parser.parse_args()

    if args.test and args.resume:
        raise ValueError(
            '--test and --resume cannot be both activate')

    if args.resume and args.start_ckpts is not None:
        raise ValueError(
            '--resume and --start_ckpts cannot be both activate')

    if args.test and args.ckpts is None:
        raise ValueError(
            'ckpts shouldnt be None while test mode')

    if args.finetune_model and args.ckpts is None:
        print(
            'training from scratch')

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.test:
        args.exp_name = 'test_' + args.exp_name
    if args.mode is not None:
        args.exp_name = args.exp_name + '_' + args.mode
    output_path = args.output_path if args.output_path is not None else './experiments'
    args.experiment_path = os.path.join(output_path, Path(
        args.config).stem, Path(args.config).parent.stem, args.exp_name)
    args.tfboard_path = os.path.join(output_path, Path(args.config).stem, Path(
        args.config).parent.stem, 'TFBoard', args.exp_name)
    args.log_name = Path(args.config).stem
    create_experiment_dir(args)
    return args


def create_experiment_dir(args):
    if not os.path.exists(args.experiment_path):
        os.makedirs(args.experiment_path)
        print('Create experiment path successfully at %s' %
              args.experiment_path)
    if not os.path.exists(args.tfboard_path):
        os.makedirs(args.tfboard_path)
        print('Create TFBoard path successfully at %s' % args.tfboard_path)
