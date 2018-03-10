import io
import os
from PIL import Image
import numpy as np
import time
import math
import matplotlib.pyplot as plt

import argparse
import torch
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable

from placenet import PlaceNet
from dataset import TripletImageLoader
from tripletnet import TripletNet
from l2normalize import L2Normalize

model = PlaceNet()

def train_model(train_loader, tripletnet, criterion, optimizer, epoch):
    # switch to train mode
    tripletnet.train()
    for batch_idx, (anchor, positive, negative) in enumerate(train_loader):
        if torch.cuda.is_available():
            anchor, positive, negative = anchor.cuda(), positive.cuda(), negative.cuda()
        anchor, positive, negative = Variable(anchor), Variable(positive), Variable(negative)

#        f, axarr = plt.subplots(2,2)
#        axarr[0,0].imshow(anchor[0].data.cpu().numpy().transpose((1, 2, 0)))
#        axarr[0,1].imshow(positive[0].data.cpu().numpy().transpose((1, 2, 0)))
#        axarr[1,0].imshow(negative[0].data.cpu().numpy().transpose((1, 2, 0)))
#        plt.show()

        # compute output
        dist_a, dist_b, embedded_x, embedded_y, embedded_z = tripletnet(anchor, positive, negative)
        print (dist_a - dist_b)
        # 1 means, dist_a should be larger than dist_b
        target = torch.FloatTensor(dist_a.size()).fill_(-1)
        if torch.cuda.is_available():
            target = target.cuda()
        target = Variable(target)
        
        loss_triplet = criterion(dist_a, dist_b, target)
        loss_embedd = embedded_x.norm(2) + embedded_y.norm(2) + embedded_z.norm(2)
        loss = loss_triplet + 0.001 * loss_embedd

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print (loss)

def train(datapath, checkpoint_path, epochs, args):
    global model

    model.train()

    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_loader = torch.utils.data.DataLoader(TripletImageLoader(datapath, size=100000, transform=preprocess), batch_size=args.bsize, shuffle=True, **kwargs)

    tripletnet = TripletNet(model)

#    criterion = torch.nn.TripletMarginLoss(margin=args.margin, p=2)
    criterion = torch.nn.MarginRankingLoss(margin=args.margin)
    optimizer = optim.SGD(tripletnet.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    for epoch in range(1, epochs + 1):
        # train for one epoch
        train_model(train_loader, tripletnet, criterion, optimizer, epoch)
#        # evaluate on validation set
#        acc = test(test_loader, tripletnet, criterion, epoch)
#
#        # remember best acc and save checkpoint
#        is_best = acc > best_acc
#        best_acc = max(acc, best_acc)
        state = {
            'epoch': epoch + 1,
            'tripletnet_state_dict': tripletnet.state_dict(),
            'state_dict': model.state_dict(),
        }
        torch.save(state, os.path.join(checkpoint_path, "checkpoint_{}.pth".format(epoch)))

def test(datapath, preprocess):
    model.eval()
    model.training = False

    with open(os.path.join(datapath, "index.txt"), 'r') as reader:
        import torch.nn.functional as F
        reps = []
        for index in reader:
            index = index.strip()
            with open(os.path.join(datapath, index, "index.txt"), 'r') as image_reader:
                for image_path in image_reader:
                    print (image_path)
                    image_path = image_path.strip()
                    image = Image.open(os.path.join(datapath, index, image_path)).convert('RGB')
                    image_tensor = preprocess(image)

#                    plt.figure()
#                    plt.imshow(image_tensor.cpu().numpy().transpose((1, 2, 0)))
#                    plt.show()

                    image_tensor.unsqueeze_(0)
                    image_variable = Variable(image_tensor).cuda()
                    features = model(image_variable)
                    reps.append(features.data.cpu())

                for i in range(len(reps)):
                    print ("\n\n")
                    for j in range(len(reps)):
                        # d = np.asarray(reps[j] - reps[i])
                        # similarity = np.linalg.norm(d)
                        # print (i, j, similarity)

                        similarity = F.pairwise_distance(reps[j], reps[i], 2)
                        print (i, j, similarity[0][0])

                        # similarity = F.cosine_similarity(reps[j], reps[i])
                        # print (i, j, similarity[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch on TORCS with Multi-modal')

    parser.add_argument('--mode', default='test', type=str, help='support option: train/test')
    parser.add_argument('--datapath', default='datapath', type=str, help='path st_lucia dataset')
    parser.add_argument('--bsize', default=32, type=int, help='minibatch size')
    parser.add_argument('--margin', type=float, default=0.2, metavar='M', help='margin for triplet loss (default: 0.2)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate (default: 0.0001)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--weight_decay', type=float, default=1e-4, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--train_iter', default=20000000, type=int, help='train iters each timestep')
    parser.add_argument('--epsilon', default=50000, type=int, help='linear decay of exploration policy')
    parser.add_argument('--l2norm', dest='l2norm', action='store_true')
    parser.add_argument('--checkpoint_path', default=None, type=str, help='Checkpoint path')
    parser.add_argument('--checkpoint', default=None, type=str, help='Checkpoint')
    args = parser.parse_args()

    normalize = transforms.Normalize(
        #mean=[121.50361069 / 127., 122.37611083 / 127., 121.25987563 / 127.],
        mean=[127. / 255., 127. / 255., 127. / 255.],
        std=[1 / 255., 1 / 255., 1 / 255.]
    )

    preprocess = transforms.Compose([
        transforms.Resize(227),
        transforms.CenterCrop(227),
        transforms.ToTensor(),
        normalize
    ])

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['state_dict'])

    if torch.cuda.is_available():
        model.cuda()

    if args.l2norm == True:
        model = L2Normalize(model)

    args = parser.parse_args()
    if args.mode == 'train':
        train(args.datapath, args.checkpoint_path, args.train_iter, args)
    elif args.mode == 'test':
        test(args.datapath, preprocess)
    else:
        raise RuntimeError('undefined mode {}'.format(args.mode))
