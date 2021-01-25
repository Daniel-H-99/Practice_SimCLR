import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision
import torchvision.datasets
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import TensorDataset, DataLoader 
from utils import *
from model import Res18
from _model import resnet18
from tqdm import tqdm
import logging
import time
from PIL import Image
from data_loader import DataSetWrapper
        
def main(args):
    
    # 0. initial setting
    
    # set environmet
    cudnn.benchmark = True
    
    if not os.path.isdir(os.path.join(args.path, './ckpt')):
        os.mkdir(os.path.join(args.path,'./ckpt'))
    if not os.path.isdir(os.path.join(args.path,'./results')):
        os.mkdir(os.path.join(args.path,'./results'))    
    if not os.path.isdir(os.path.join(args.path, './ckpt', args.name)):
        os.mkdir(os.path.join(args.path, './ckpt', args.name))
    if not os.path.isdir(os.path.join(args.path, './results', args.name)):
        os.mkdir(os.path.join(args.path, './results', args.name))
    if not os.path.isdir(os.path.join(args.path, './results', args.name, "log")):
        os.mkdir(os.path.join(args.path, './results', args.name, "log"))

    # set logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    handler = logging.FileHandler(os.path.join(args.path, "results/{}/log/{}.log".format(args.name, time.strftime('%c', time.localtime(time.time())))))
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler())
    args.logger = logger
    
    # set cuda
    if torch.cuda.is_available():
        args.logger.info("running on cuda")
        args.device = torch.device("cuda")
        args.use_cuda = True
    else:
        args.logger.info("running on cpu")
        args.device = torch.device("cpu")
        args.use_cuda = False
        
    # Hyperparameters setting #
    epochs = args.epochs
    batch_size = args.batch_size
    T = args.temperature
    proj_dim = args.out_dim
    
    args.logger.info("[{}] starts".format(args.name))

    ### DataLoader ###
    args.logger.info("loading data...")
    dataset = DataSetWrapper(args, args.batch_size , args.num_workers , args.valid_size, input_shape = (96, 96, 3))
    train_loader , valid_loader = dataset.get_data_loaders()
    finetune_loader, test_loader = dataset.get_finetune_data_loaders()

    ### Setting ###
    model = Res18(args).to(args.device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)
    
    if args.load:
        args.logger.info("loading checkpoint...")
        loaded_data = load(args, args.ckpt)
        model.load_state_dict(loaded_data['model'])
        args.logger.info("continue from step {}".format(args.start_from_step))
    
    if not args.test:
        if not args.finetune:
            # pretrain
            args.logger.info("starting pretraining")
            val_loss_meter = AverageMeter(args, name="Val_Loss (%)", save_all=True, x_label="epoch")
            train_loss_meter = AverageMeter(args, name="Loss", save_all=True, x_label="epoch")
            early_stop_cnt = 0
            val_loss = 1e9
            steps = 0
            model.pretrain()

            for epoch in range(1, 1 + epochs):
                # train
                steps += 1
                if args.start_from_step is not None:
                    if steps < args.start_from_step:
                        continue
                spent_time = time.time()
                model.train()
                train_loss_tmp_meter = AverageMeter(args)
                for (xi, xj), _ in tqdm(train_loader):
                    # xi: (batch x channels x width x height)
                    optimizer.zero_grad()
                    batch = xi.shape
                    input = torch.cat((xi, xj), dim=1).view(2 * batch[0], batch[1], batch[2], batch[3]).to(args.device)
                    target = torch.cat([torch.arange(batch[0]).unsqueeze(1)] * 2, dim=1).flatten(0) * 2
                    z = model(input)
                    loss = loss_fn(z, target.to(args.device)) 
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    train_loss_tmp_meter.update(loss.cpu().detach().numpy(), weight=batch[0])

                train_loss_meter.update(train_loss_tmp_meter.avg)
                spent_time = time.time() - spent_time
                args.logger.info("[{}] train loss: {:.3f} took {:.1f} seconds".format(epoch, train_loss_tmp_meter.avg, spent_time))

                # validation
                model.eval()
                val_loss_tmp_meter = AverageMeter(args)
                spent_time = time.time()
                with torch.no_grad():
                    for (val_xi, val_xj), _ in valid_loader:
                        batch = val_xi.shape
                        input = torch.cat((val_xi, val_xj), dim=1).view(2 * batch[0], batch[1], batch[2], batch[3]).to(args.device)
                        target = torch.cat([torch.arange(batch[0]).unsqueeze(1)] * 2, dim=1).flatten(0) * 2
                        z = model(input)
                        loss = loss_fn(z, target.to(args.device))
                        val_loss_tmp_meter.update(loss.cpu().detach().numpy(), weight=batch[0])

                spent_time = time.time() - spent_time
                args.logger.info("[{}] validation loss: {:.3f}, took {} seconds".format(steps, val_loss_tmp_meter.avg, spent_time))
                val_loss_meter.update(val_loss_tmp_meter.avg)
                if val_loss_meter.val > val_loss:
                    early_stop_cnt += 1
                else:
                    early_stop_cnt = 0
                val_loss = val_loss_meter.val
                if (steps % args.save_period == 0) or (early_stop_cnt >= args.early_stop_threshold):
                    save(args, "epoch_{}".format(epoch), {'model': model.state_dict()})
                    val_loss_meter.save()
                    val_loss_meter.plot()
                    train_loss_meter.save()
                    train_loss_meter.plot()
                    args.logger.info("[{}] saved".format(steps))
                if early_stop_cnt >= args.early_stop_threshold:
                    break
        else:
            # finetune
            args.logger.info("starting finetuning")
            train_loss_meter = AverageMeter(args, name="Finetune_Loss", save_all=True, x_label="epoch")
            steps = 0
            if args.lin_eval:
                model.lin_eval()
            else:
                model.finetune()
            for epoch in range(1 , 1 + epochs):
                steps += 1
                if args.start_from_step is not None:
                    if steps < args.start_from_step:
                        continue
                spent_time = time.time()
                model.train()
                train_loss_tmp_meter = AverageMeter(args)
                for input, target in tqdm(finetune_loader):
                    # input: (batch x channels x width x height)
                    optimizer.zero_grad()
                    batch = input.shape
                    input = input.to(args.device)
                    pred = model(input)
                    loss = loss_fn(pred, target.to(args.device)) 
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    train_loss_tmp_meter.update(loss)

                train_loss_meter.update(train_loss_tmp_meter.avg)
                spent_time = time.time() - spent_time
                args.logger.info("[{}] finetune loss: {:.3f} took {:.1f} seconds".format(epoch, train_loss_tmp_meter.avg, spent_time))
            
                if epoch % args.save_period == 0:
                    save(args, "epoch_{}".format(epoch), {'model': model.state_dict()})
                    train_loss_meter.save()
                    train_loss_meter.plot()
                    args.logger.info("[{}] saved".format(steps))
            
    else:
        # test
        args.logger.info("starting test")
        model.eval()
        model.finetune()
        test_acc_meter = AverageMeter(args, name="Test-Acc (%)")
        with torch.no_grad():
            spent_time = time.time()
            for input, target in tqdm(test_loader):
                # input: (batch x channels x width x height)
                batch = input.shape
                input = input.to(args.device)
                pred = model(input)
                test_acc_meter.update(top_n_accuracy_score(pred, target.to(args.device), n=1), weight=batch[0])

            spent_time = time.time() - spent_time
            args.logger.info("Test accuracy: {:.2f} took {:.1f} seconds".format(test_acc_meter.avg, spent_time))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "SimCLR implementation")

    parser.add_argument(
        '--temperature',
        type=float,
        default=0.5)
    parser.add_argument(
        '--out_dim',
        type=int,
        default=256)
    parser.add_argument(
        '--valid_size',
        type=float,
        default=0.05)
    parser.add_argument(
        '--data_dir',
        type=str,
        default='dataset')
    parser.add_argument(
        '--result_dir',
        type=str,
        default='results')
    parser.add_argument(
        '--ckpt_dir',
        type=str,
        default="ckpt")
    parser.add_argument(
        '--path',
        type=str,
        default='.')
    parser.add_argument(
        '--epochs',
        type=int,
        default=100)
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-5)
    parser.add_argument(
    	'--warmup',
    	type=int,
    	default=5),
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64)
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4)
    parser.add_argument(
        '--test',
        action='store_true')
    parser.add_argument(
        '--save_period',
        type=int,
        default=5)
    parser.add_argument(
        '--name',
        type=str,
        default="train")
    parser.add_argument(
        '--ckpt',
        type=str,
        default='_')
    parser.add_argument(
        '--load',
        action='store_true')
    parser.add_argument(
        '--channels',
        type=int,
        default=3)
    parser.add_argument(
        '--width',
        type=int,
        default=256)
    parser.add_argument(
        '--height',
        type=int,
        default=256)
    parser.add_argument(
        '--start_from_step',
        type=int,
        default=None)
    parser.add_argument(
        '--classes',
        type=int,
        default=10)
    parser.add_argument(
        '--resize',
        type=int,
        default=112)
    parser.add_argument(
        '--imgsize',
        type=int,
        default=96)
    parser.add_argument(
        '--finetune',
        action='store_true')
    parser.add_argument(
        '--lin_eval',
        action='store_true')
    parser.add_argument(
        '--use_z',
        action='store_true')
    parser.add_argument(
        '--triple_layer_projection',
        action='store_true')
    parser.add_argument(
        '--early_stop_threshold',
        type=int,
        default=2)

    args = parser.parse_args()
    main(args)




