import argparse
import os.path as osp
import random
import numpy as np
import time
from datetime import timedelta

import torch
from torch import nn
from torch.backends import cudnn

from reid import models
from reid.server import FedDomainMemoTrainer
from reid.evaluators import Evaluator
from reid.utils.serialization import load_checkpoint, save_checkpoint
from reid.utils.tools import get_test_loader, get_data
from reid import datasets

start_epoch = best_mAP = 0


def create_model(args, num_cls=0):
    # we only use triplet loss, remember to turn off 'norm'
    # model = models.create(
    #     args.arch, num_features=args.features, norm=False,
    #     dropout=args.dropout, num_classes=num_cls
    # )
    model = models.make_model(
        args=args, num_class=num_cls, 
        index_num=None
    )
    # use CUDA
    model = model.cuda()
    model = nn.DataParallel(model) if args.is_parallel else model
    return model


def main():
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    main_worker(args)


def main_worker(args):
    global start_epoch, best_mAP
    start_time = time.monotonic()

    cudnn.benchmark = True
    all_datasets = datasets.names()
    test_set_name = args.test_dataset
    all_datasets.remove(test_set_name)
    
    if args.exclude_dataset is not '':
        exclude_set_name = args.exclude_dataset.split(',')
        [all_datasets.remove(name) for name in exclude_set_name]
    train_sets_name = sorted(all_datasets)

    # Create datasets
    print("==> Building Datasets")
    test_set = get_data(args)
    test_loader = get_test_loader(test_set, args.height, args.width, 
                                  args.batch_size, args.workers)
    train_sets = get_data(args, train_sets_name)
    num_users = len(train_sets)

    # Create model
    model = create_model(args)  
    # sub local models
    sub_models = [create_model(args) for key in range(num_users)]
    
    # Evaluator
    evaluator = Evaluator(model)
    if args.evaluate:
        evaluator.evaluate(test_loader, test_set.query, test_set.gallery, cmc_flag=True)
        return

    # trainer = FedDomainMemoTrainer(args, train_sets, model)
    trainer = FedDomainMemoTrainer(args, train_sets, model, feature_dim=768)
    
    # if args.resume:
    #     checkpoint = load_checkpoint(args.resume)
    #     start_epoch = checkpoint['epoch'] - 1
    #     model.load_state_dict(checkpoint['state_dict'], strict=False)
    #     evaluator.evaluate(test_loader, test_set.query, test_set.gallery, cmc_flag=True)
    #     pass
        
    # start training
    for epoch in range(args.epochs):  # number of epochs
        w_locals = []
        torch.cuda.empty_cache()
        for index in range(num_users):  # client index
            w = trainer.train_reid(sub_models[index], model, epoch, index)
            w_locals.append(w)
        # update global weight
        w_global = trainer.fed_avg(w_locals)
        model.load_state_dict(w_global) # the averaged model
        
        if epoch % args.eval_step == 0:
            cur_map, rank1 = evaluator.evaluate(test_loader, test_set.query,
                                                test_set.gallery, cmc_flag=True)
            # save
            if cur_map > best_mAP:
                print('best model saved!')
                save_checkpoint({
                    'state_dict': w_global,
                    'epoch': epoch + 1,
                    'best_mAP': best_mAP,
                }, 1, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))
            best_mAP = cur_map


    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Domain-level Fed Learning")
    # data
    parser.add_argument('-td', '--test-dataset', type=str, default='market1501',
                        choices=datasets.names())
    parser.add_argument('-ed', '--exclude-dataset', type=str, default='')
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    # model
    # parser.add_argument('-a', '--arch', type=str, default='resnet50',
    #                     choices=models.names())
    # parser.add_argument('--resume', type=str, default='')
    parser.add_argument('-a', '--arch', type=str, default='transformer',
                        choices=models.names())
    parser.add_argument('--pretrain-choice', type=str, default='imagenet')
    parser.add_argument('--neck-feat', type=str, default='before')
    parser.add_argument('--neck', type=str, default='bnneck')
    parser.add_argument('--transformer-type', type=str, default='vit_base_patch16_224_TransReID')
    parser.add_argument('--stride-size', type=int, default=12)
    parser.add_argument('--shift-num', type=int, default=5)
    parser.add_argument('--shuffle-group', type=int, default=2)
    parser.add_argument('--devide-length', type=int, default=4)
    parser.add_argument('--re-arrange', type=bool, default=True)
    parser.add_argument('--last-stride', type=bool, default=True)
    parser.add_argument('--resume', type=str, default='./checkpoints/jx_vit_base_p16_224-80ecf9dd.pth')
    
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    # optimizer
    parser.add_argument('--temp', type=float, default=3, help="temperature")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--milestones', nargs='+', type=int, 
                        default=[20, 30], help='milestones for the learning rate decay')
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--warmup-step', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3,
                        help="learning rate")
    parser.add_argument('--epochs', type=int, default=41)
    parser.add_argument('--max-iter', type=int, default=200)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 4")
    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--eval-step', type=int, default=10)
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--is_parallel', type=int, default=1)
    main()
