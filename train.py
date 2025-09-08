from __future__ import division
from __future__ import print_function

import sys
sys.path.append('/data1/ziyan/RFN2')# Please change accordingly!

import datetime
import json
import logging
from optim import RiemannianAdam, RiemannianSGD
import os
import pickle
import time
import contextlib

import numpy as np
import torch
from config import parser, add_flags_from_config, config_args
from models.base_models import NCModel
from utils.data_utils import load_data, get_nei, get_nei_adaptive, GCDataset, split_batch
from utils.train_utils import get_dir_name, format_metrics
from utils.eval_utils import acc_f1

from geoopt import ManifoldParameter as geoopt_ManifoldParameter
from manifolds.base import ManifoldParameter as base_ManifoldParameter
from torch.cuda.amp import autocast, GradScaler  

parser.add_argument('--use_amp', action='store_true', help='Use mixed precision training')
parser.add_argument('--grad_accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
parser.add_argument('--use_neighbor_sampling', action='store_true', help='Use neighbor sampling optimization')

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
torch.cuda.empty_cache()


def train(args):
    
    scaler = GradScaler() if args.use_amp else None
    
    #choose which manifold class to follow 
    if args.use_geoopt == False:
        print("base_ManifoldParameter")
        ManifoldParameter = base_ManifoldParameter
    else:
        print("geoopt_ManifoldParameter")
        ManifoldParameter = geoopt_ManifoldParameter

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if int(args.double_precision):
        torch.set_default_dtype(torch.float64)
    if int(args.cuda) >= 0:
        torch.cuda.manual_seed(args.seed)
    args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'
    # args.device = 'cpu'
    args.patience = args.epochs if not args.patience else int(args.patience)
    logging.getLogger().setLevel(logging.INFO)
    if args.save:
        if not args.save_dir:
            dt = datetime.datetime.now()
            date = f"{dt.year}_{dt.month}_{dt.day}"
            models_dir = os.path.join('logs', args.task, date)
            save_dir = get_dir_name(models_dir)
        else:
            save_dir = args.save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        logging.basicConfig(level=logging.INFO,
                            handlers=[
                                logging.FileHandler(
                                    os.path.join(save_dir, 'log.txt')),
                                logging.StreamHandler()
                            ])

    logging.info(f'Using: {args.device}')
    logging.info("Using seed {}.".format(args.seed))
    logging.info(f"Dataset: {args.dataset}")

    # Load data
    data = load_data(args, os.path.join('data', args.dataset))
    if args.task == 'nc':
        Model = NCModel
        args.n_classes = int(data['labels'].max() + 1)
        args.n_nodes = data['features'].shape[0]  
        args.data = data
        logging.info(f'Num classes: {args.n_classes}')
        logging.info(f'Num nodes: {args.n_nodes}')
    else:
        args.nb_false_edges = len(data['train_edges_false'])
        args.nb_edges = len(data['train_edges'])
        if args.task == 'lp':
            Model = LPModel
            args.n_classes = 2

    if not args.lr_reduce_freq:
        args.lr_reduce_freq = args.epochs

    # Model and optimizer
    model = Model(args)
    logging.info(str(model)) #This prints out the structure of the model
    no_decay = ['bias', 'scale']


    if args.optimizer == 'adam':
        optimizer_grouped_parameters = [{
            'params': [p for n, p in model.named_parameters() if p.requires_grad],
            'weight_decay': args.weight_decay
        }]
    else: #radam and rsgd
        optimizer_grouped_parameters = [{
            'params': [
                p for n, p in model.named_parameters()
                if p.requires_grad and not any(nd in n for nd in no_decay) and not isinstance(p, ManifoldParameter)
            ],
            'weight_decay': args.weight_decay
        }, {
            'params': [
                p for n, p in model.named_parameters()
                if p.requires_grad and (any(nd in n for nd in no_decay) or isinstance(p, ManifoldParameter))
            ],
            'weight_decay': 0.0
        }]


    if args.optimizer == 'radam':
        optimizer = RiemannianAdam(params=optimizer_grouped_parameters,
                                   lr=args.lr,
                                   stabilize=10)
    elif args.optimizer == 'rsgd':
        optimizer = RiemannianSGD(params=optimizer_grouped_parameters,
                                  lr=args.lr,
                                  stabilize=10)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params=optimizer_grouped_parameters, lr=args.lr)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=int(
                                                       args.lr_reduce_freq),
                                                   gamma=float(args.gamma))
    tot_params = sum([np.prod(p.size()) for p in model.parameters()])
    model = model.to(args.device)
    for x, val in data.items():
        if torch.is_tensor(data[x]):
            data[x] = data[x].to(args.device)
    logging.info(f"Total number of parameters: {tot_params}")
    # Train model
    t_total = time.time()
    counter = 0
    best_val_metrics = model.init_metric_dict()
    best_test_metrics = None
    best_emb = None
    if args.n_classes > 2:
        f1_average = 'micro'
    else:
        f1_average = 'binary'

    #Starting form here is node classicication
    if args.task == 'gc':
        dataset = GCDataset((data['adj_train'], data['features'], data['labels']), KP=(args.model == 'HKPNet'),
                             normlize=args.normalize_adj, device = args.device)
        for epoch in range(args.epochs):
            t = time.time()
            model.train()
            tot_metrics = {'loss': 0, 'acc': 0, 'f1': 0}
            outs = None
            labs = None
            bats = 0
            adj = None
            
            # 添加梯度累积支持
            optimizer.zero_grad()
            accumulation_steps = args.grad_accumulation_steps
            accumulated_batch = 0
            
            for i in range(0, len(data['idx_train']), args.batch_size):
                selected_idx = data['idx_train'][i : i + args.batch_size]
                if len(selected_idx) == 0:
                    continue
                
                
                if args.use_amp:
                    with autocast():
                        if args.model == 'HKPNet':
                            nei, nei_mask, features, labels, ed_idx = dataset[selected_idx]
                            embeddings = model.encode(features, (nei, nei_mask))
                        else:
                            adj, features, labels, ed_idx = dataset[selected_idx]
                            embeddings = model.encode(features, adj)
                        
                        train_metrics = model.compute_metrics(embeddings, labels, ed_idx, type=2, adj=adj)
                        # 如果使用梯度累积，对损失进行标准化
                        if accumulation_steps > 1:
                            train_metrics['loss'] = train_metrics['loss'] / accumulation_steps
                else:
                    if args.model == 'HKPNet':
                        nei, nei_mask, features, labels, ed_idx = dataset[selected_idx]
                        embeddings = model.encode(features, (nei, nei_mask))
                    else:
                        adj, features, labels, ed_idx = dataset[selected_idx]
                        embeddings = model.encode(features, adj)
                    
                    train_metrics = model.compute_metrics(embeddings, labels, ed_idx, type=2, adj=adj)
                    # 如果使用梯度累积，对损失进行标准化
                    if accumulation_steps > 1:
                        train_metrics['loss'] = train_metrics['loss'] / accumulation_steps
                
                tot_metrics['loss'] += train_metrics['loss'].detach().cpu().numpy() * (1 if accumulation_steps <= 1 else accumulation_steps)
                bats += 1
                
                # 使用混合精度进行反向传播
                if args.use_amp:
                    scaler.scale(train_metrics['loss']).backward()
                else:
                    train_metrics['loss'].backward()
                
                accumulated_batch += 1
                
                # 当累积的批次达到指定数量或最后一个批次时执行优化器步骤
                if accumulated_batch == accumulation_steps or i + args.batch_size >= len(data['idx_train']):
                    if args.grad_clip is not None:
                        max_norm = float(args.grad_clip)
                        if args.use_amp:
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

                    if args.print_grad == True:
                        for name, param in model.named_parameters():
                            if param.requires_grad and param.grad is not None:
                                print(f"Gradient for {name}: {param.grad.norm().item()}")

                    
                    if args.use_amp:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    
                    optimizer.zero_grad()
                    accumulated_batch = 0
                
                if outs is None:
                    outs = train_metrics['output']
                else:
                    outs = torch.cat([outs, train_metrics['output']], 0)
                if labs is None:
                    labs = labels
                else:
                    labs = torch.cat([labs, labels], 0)
            
            lr_scheduler.step()
            tot_metrics['acc'], tot_metrics['f1'] = acc_f1((outs), (labs), f1_average)
            tot_metrics['loss'] /= bats

            if (epoch + 1) % args.log_freq == 0:
                logging.info(" ".join(['Epoch: {:04d}'.format(epoch + 1),
                                       'lr: {}'.format(lr_scheduler.get_last_lr()),
                                       format_metrics(tot_metrics, 'train'),
                                       'time: {:.4f}s'.format(time.time() - t)
                                       ]))

            if (epoch + 1) % args.eval_freq == 0:
                model.eval()
                val_metrics = {'loss': 0, 'acc': 0, 'f1': 0}
                outs = None
                labs = None
                bats = 0
                adj = None
                
                for i in range(0, len(data['idx_val']), args.batch_size):
                    selected_idx = data['idx_val'][i : i + args.batch_size]
                    if len(selected_idx) == 0:
                        continue
                    if args.model == 'HKPNet':
                        nei, nei_mask, features, labels, ed_idx = dataset[selected_idx]
                        #adj, _, _, _ = dataset[selected_idx]
                        embeddings = model.encode(features, (nei, nei_mask))
                    else:
                        adj, features, labels, ed_idx = dataset[selected_idx]
                        embeddings = model.encode(features, adj)
                    
                    metrics = model.compute_metrics(embeddings, labels, ed_idx, type=2, adj=adj)
                    val_metrics['loss'] += metrics['loss'].detach().cpu().numpy()
                    bats += 1
                    
                    if outs is None:
                        outs = metrics['output']
                    else:
                        outs = torch.cat([outs, metrics['output']], 0)
                    if labs is None:
                        labs = labels
                    else:
                        labs = torch.cat([labs, labels], 0)
                
                val_metrics['acc'], val_metrics['f1'] = acc_f1((outs), (labs), f1_average)
                val_metrics['loss'] /= bats
                
                if (epoch + 1) % args.log_freq == 0:
                    logging.info(" ".join(['Epoch: {:04d}'.format(epoch + 1), format_metrics(val_metrics, 'val')]))

                if model.has_improved(best_val_metrics, val_metrics):
                    best_val_metrics = val_metrics
                    best_test_metrics = {'loss': 0, 'acc': 0, 'f1': 0}
                    outs = None
                    labs = None
                    bats = 0
                    adj = None
                    
                    for i in range(0, len(data['idx_test']), args.batch_size):
                        selected_idx = data['idx_test'][i : i + args.batch_size]
                        if len(selected_idx) == 0:
                            continue
                        if args.model == 'HKPNet':
                            nei, nei_mask, features, labels, ed_idx = dataset[selected_idx]
                            embeddings = model.encode(features, (nei, nei_mask))
                        else:
                            adj, features, labels, ed_idx = dataset[selected_idx]
                            embeddings = model.encode(features, adj)
                        
                        test_metrics = model.compute_metrics(embeddings, labels, ed_idx, type=2, adj=adj)
                        best_test_metrics['loss'] += test_metrics['loss'].detach().cpu().numpy()
                        bats += 1
                        
                        if outs is None:
                            outs = test_metrics['output']
                        else:
                            outs = torch.cat([outs, test_metrics['output']], 0)
                        if labs is None:
                            labs = labels
                        else:
                            labs = torch.cat([labs, labels], 0)
                    
                    best_test_metrics['loss'] /= bats
                    if outs is not None:
                        best_test_metrics['acc'], best_test_metrics['f1'] = acc_f1((outs), (labs), f1_average)
                    counter = 0
                else:
                    counter += 1
                    if counter == args.patience and epoch > args.min_epochs:
                        logging.info("Early stopping")
                        break

        logging.info("Optimization Finished!")
        logging.info("Total time elapsed: {:.4f}s".format(time.time() - t_total))

        if best_test_metrics['loss'] == 0:
            model.eval()
            best_test_metrics = {'loss': 0, 'acc': 0, 'f1': 0}
            outs = None
            labs = None
            bats = 0
            adj = None
            
            for i in range(0, len(data['idx_test']), args.batch_size):
                selected_idx = data['idx_test'][i : i + args.batch_size]
                if len(selected_idx) == 0:
                    continue
                if args.model == 'HKPNet':
                    nei, nei_mask, features, labels, ed_idx = dataset[selected_idx]
                    embeddings = model.encode(features, (nei, nei_mask))
                else:
                    adj, features, labels, ed_idx = dataset[selected_idx]
                    embeddings = model.encode(features, adj)
                
                test_metrics = model.compute_metrics(embeddings, labels, ed_idx, type=2, adj=adj)
                best_test_metrics['loss'] += test_metrics['loss'].detach().cpu().numpy()
                bats += 1
                
                if outs is None:
                    outs = test_metrics['output']
                else:
                    outs = torch.cat([outs, test_metrics['output']], 0)
                if labs is None:
                    labs = labels
                else:
                    labs = torch.cat([labs, labels], 0)
            
            best_test_metrics['loss'] /= bats
            if outs is not None:
                best_test_metrics['acc'], best_test_metrics['f1'] = acc_f1((outs), (labs), f1_average)

        logging.info(" ".join(["Val set results:", format_metrics(best_val_metrics, 'val')]))
        logging.info(" ".join(["Test set results:", format_metrics(best_test_metrics, 'test')]))

 
    #Starting form here is node classicication
    else:
        if args.model == 'HKPNet':
            
            if args.use_neighbor_sampling:
                nei, nei_mask = get_nei_adaptive(data['adj_train'], use_sampling=True)
            else:
                nei, nei_mask = get_nei(data['adj_train'])
            nei = nei.to(args.device)
            nei_mask = nei_mask.to(args.device)
            
        
        accumulation_steps = args.grad_accumulation_steps
            
        for epoch in range(args.epochs):
            t = time.time()
            model.train()
            optimizer.zero_grad()
            
            
            if args.use_amp:
                with autocast():
                    if args.model == 'HKPNet':
                        embeddings = model.encode(data['features'], (nei, nei_mask))
                    else:
                        embeddings = model.encode(data['features'], data['adj_train_norm'])
                    
                    train_metrics = model.compute_metrics(embeddings, data, 'train')
                    
                    
                    if accumulation_steps > 1:
                        train_metrics['loss'] = train_metrics['loss'] / accumulation_steps
            else:
                if args.model == 'HKPNet':
                    embeddings = model.encode(data['features'], (nei, nei_mask))
                else:
                    embeddings = model.encode(data['features'], data['adj_train_norm'])
                
                train_metrics = model.compute_metrics(embeddings, data, 'train')
                
                
                if accumulation_steps > 1:
                    train_metrics['loss'] = train_metrics['loss'] / accumulation_steps
                
            
            if args.use_amp:
                scaler.scale(train_metrics['loss']).backward()
            else:
                train_metrics['loss'].backward()
                
            if args.grad_clip is not None:
                if args.use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            if args.print_grad == True:
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        print(f"Gradient for {name}: {param.grad.norm().item()}")
            
            
            if args.use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            lr_scheduler.step()

            #torch.cuda.empty_cache()# try this
            
            if (epoch + 1) % args.log_freq == 0:
                logging.info(" ".join([
                    'Epoch: {:04d}'.format(epoch + 1),
                    'lr: {}'.format(lr_scheduler.get_last_lr()),
                    format_metrics(train_metrics, 'train'),
                    'time: {:.4f}s'.format(time.time() - t)
                ]))
            with torch.no_grad():
                if (epoch + 1) % args.eval_freq == 0:
                    model.eval()
                    if args.model == 'HKPNet':
                        embeddings = model.encode(data['features'], (nei, nei_mask))
                    else:
                        embeddings = model.encode(data['features'],
                                                data['adj_train_norm'])
                    val_metrics = model.compute_metrics(embeddings, data, 'val')
                    if (epoch + 1) % args.log_freq == 0:
                        logging.info(" ".join([
                            'Epoch: {:04d}'.format(epoch + 1),
                            format_metrics(val_metrics, 'val')
                        ]))
                    if model.has_improved(best_val_metrics, val_metrics):
                        best_test_metrics = model.compute_metrics(
                            embeddings, data, 'test')
                        best_emb = embeddings.cpu()
                        if args.save:
                            np.save(os.path.join(save_dir, 'embeddings.npy'),
                                    best_emb.detach().numpy())
                        best_val_metrics = val_metrics
                        counter = 0
                    else:
                        counter += 1
                        if counter == args.patience and epoch > args.min_epochs:
                            logging.info("Early stopping")
                            break

        logging.info("Optimization Finished!")
        logging.info("Total time elapsed: {:.4f}s".format(time.time() - t_total))
        if not best_test_metrics:
            model.eval()
            best_emb = model.encode(data['features'], data['adj_train_norm'])
            best_test_metrics = model.compute_metrics(best_emb, data, 'test')
        logging.info(" ".join(
            ["Val set results:",
            format_metrics(best_val_metrics, 'val')]))
        logging.info(" ".join(
            ["Test set results:",
            format_metrics(best_test_metrics, 'test')]))
        if args.save:
            np.save(os.path.join(save_dir, 'embeddings.npy'),
                    best_emb.cpu().detach().numpy())
            if hasattr(model.encoder, 'att_adj'):
                filename = os.path.join(save_dir, args.dataset + '_att_adj.p')
                pickle.dump(model.encoder.att_adj.cpu().to_dense(),
                            open(filename, 'wb'))
                print('Dumped attention adj: ' + filename)

            torch.save(model.state_dict(), os.path.join(save_dir, 'model.pth'))
            # Filter out non-serializable objects before saving config
            args_dict = vars(args).copy()
            # Remove fields that contain non-serializable objects
            non_serializable_fields = ['data', 'adj_train', 'adj_train_norm', 'adj_val', 'adj_test']
            for field in non_serializable_fields:
                if field in args_dict:
                    del args_dict[field]
            json.dump(args_dict, open(os.path.join(save_dir, 'config.json'), 'w'))
            logging.info(f"Saved model in {save_dir}")


if __name__ == '__main__':
    args = parser.parse_args()
    #print(args)
    train(args)
