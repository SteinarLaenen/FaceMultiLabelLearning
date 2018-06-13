import numpy as np
import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Trainng')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run (default: 90')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--filename', default='unnamed_model', type=str,
                    help='choose a name to save checkpoints and best models')
parser.add_argument('--split', default='unnamed_model', type=str,
                    help='choose a split of the targetset')
parser.add_argument('--seed', default=1, type=int,
                    help='set random seed')

best_prec1 = 0
loss_progression = []
top5_progression = []
top1_progression = []
loss_progression_val = []
top5_progression_val = []
top1_progression_val = []
time_progression = []
time_progression_val = []
def main():
    global args, best_prec1, loss_progression, top5_progression, top1_progression, loss_progression_val, top5_progression_val, top1_progression_val, time_progression, time_progression_val
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    # create moel
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch](num_classes1 = 6, num_classes2 = 12)

    if not args.distributed:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    else:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            loss_progression = checkpoint['loss_progression']
            loss_progression_val = checkpoint['loss_progression_val']
            top1_progression = checkpoint['top1_progression']
            top1_progression_val = checkpoint['top1_progression_val']
            top5_progression = checkpoint['top5_progression']
            top5_progression_val = checkpoint['top5_progression_val']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    #changed the values of the normalize, computed in another folder.
    normalize = transforms.Normalize(mean=[0.179, 0.104, 0.0848],
                                     std=[0.023392, 0.014109, 0.011765])

    train_dataset = datasets.ImageFolderSplit(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            normalize,
        ]), split = args.split)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolderSplit(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]), split = args.split),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
            'loss_progression' : loss_progression,
            'top1_progression' : top1_progression,
            'top5_progression' : top5_progression,
            'loss_progression_val' : loss_progression_val,
            'top1_progression_val' : top1_progression_val,
            'top5_progression_val' : top5_progression_val,
            'time_progression' : time_progression,
            'time_progression_val' : time_progression_val,
        }, is_best, args.filename)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        #print target_var.shape

        # compute output
        output1, output2 = model(input_var)
        loss1 = criterion(output1, target_var[:,0])
        loss2 = criterion(output2, target_var[:,1])
        loss = loss1 + loss2
        
        # measure accuracy and record loss
        prec1 = accuracy_multitask(output1.data, output2.data, target)
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1, input.size(0))
        #top5.update(prec5[0], input.size(0))
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        record_data(losses.val, top1.val, top5.val, batch_time.val, 0)
        
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
           
            
def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        output1, output2 = model(input_var)
        loss1 = criterion(output1, target_var[:,0])
        loss2 = criterion(output2, target_var[:,1])
        loss = loss1 + loss2
        
        # measure accuracy and record loss
        prec1 = accuracy_multitask(output1.data, output2.data, target)
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1, input.size(0))
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    record_data(losses.avg, top1.avg, top5.avg, batch_time.avg,  1)
        
   
    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    print("Data")
    
    return top1.avg

def save_checkpoint(state, is_best, filename):
    name = 'checkpoint_' + filename + '.pth.tar'
    torch.save(state, name)
    if is_best:
        shutil.copyfile(name, 'model_best_' + filename + '.pth.tar')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def record_data(loss, top1, top5, timevar, mode):
    global loss_progression, top1_progression, top5_progression, loss_progression_val, top1_progression_val, top5_progression_val, time_progression, time_progression_val
    if mode == 1:
        loss_progression_val.append(loss)
        top1_progression_val.append(top1)
        top5_progression_val.append(top5)
        time_progression_val.append(timevar)
    else:
        loss_progression.append(loss)
        top1_progression.append(top1)
        top5_progression.append(top5)
        time_progression.append(timevar)
    
    
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def accuracy_multitask(output1, output2, target):

    batch_size = target.size(0)
    n = 0

    for i in range(batch_size):
        pred1, pred2 = output1[i], output2[i]
        emotion_pred = torch.max(pred1, 0)[1]
        physical_pred = torch.max(pred2, 0)[1]
        prediction = torch.cat((emotion_pred, physical_pred), 0)

        
        if torch.equal(target[i], prediction):
            n = n + 1


    return 100*float(n)/float(batch_size)

if __name__ == '__main__':
    main()
