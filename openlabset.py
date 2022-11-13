from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from model import PointNetCls, feature_transform_regularizer
import torch.nn.functional as F
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import torch
import json
import codecs
import numpy as np
import sys
import torchvision.transforms as transforms
import argparse
import json
import open3d

import os
import random
from random import randrange
from IPython.display import clear_output
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

# from datasets import PartDataset
# from pointnet import PointNetCls
import torch.nn.functional as F
import matplotlib.pyplot as plt
import open3d as o3

# from open3d import JVisualizer
if torch.cuda.is_available():
    import torch.backends.cudnn as cudnn

import open3d as o3d
import os
import glob

import torchvision
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

data_path = "" # Your Data Path
print(data_path)
classes = os.listdir(data_path)
print(classes)

class MeshDataset(data.Dataset):
    def __init__(self, root, npoints=1024, classification=True):
        self.npoints = npoints
        self.root = root
        self.catfile = "" # Your Data Path
        self.cat = {}

        self.classification = classification
        # cnt = 0
        for item in os.listdir(self.catfile):
            self.cat[item] = item
            # cnt += 1
        # print(self.cat)
        # print()

        self.meta = {}
        for item in self.cat:
            # print(item)
            # print()
            self.meta[item] = []
            # print(self.meta)
            # print()
            dir_mesh = os.path.join(self.root, self.cat[item])
            # print(dir_mesh)
            # print()
            fns = glob.glob(os.path.join(dir_mesh, "*.obj"))
            random.seed("2022-10-29")
            random.shuffle(fns)
            # print(fns)
            # print()
            # if train:
            #     fns = fns[-int(len(fns) * 0.8) :]  # train_plist
            # else:
            #     fns = fns[: -int(len(fns) * 0.8)]  # valid_plist

            # print(f"train : {len(train_plist)}")
            # print(f"valid : {len(valid_plist)}")

            # print(os.path.basename(fns))
            for fn in fns:
                token = os.path.splitext(fn)[0]
                self.meta[item].append((os.path.join(dir_mesh, token + ".obj"), item))

            # print(self.meta)
            # print()
            # break
        # print("==================================================")
        # print("self.meta\n", self.meta)
        # print()
        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                # print(fn)
                # print(fn[0])
                self.datapath.append((item, fn[0], fn[1]))
            # break
        # print("self.datapath\n", self.datapath)
        # print("=======================================================")
        # print()
        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))
        # print("self.classes\n", self.classes)
        # print()
        # self.num_classes = 0
        # if not self.classification:
        #     for i in range(len(self.datapath) // 50):
        #         l = len(np.unique(np.loadtxt(self.datapath[i][-1]).astype(np.uint8)))
        #         if l > self.num_seg_classes:
        #             self.num_seg_classes = l

    def __getitem__(self, index):
        fn = self.datapath[index]
        # print(fn)
        # print(fn[0], fn[1], fn[2])
        # print(self.classes)
        cls = self.classes[self.datapath[index][0]]
        # print(cls)
        mesh = o3d.io.read_triangle_mesh(fn[1])
        # print(mesh)
        mesh.compute_vertex_normals()
        # print(mesh)
        pcd = mesh.sample_points_uniformly(number_of_points=self.npoints)
        # print(pcd)
        point_set = np.asarray(pcd.points, dtype=np.float32)
        # seg = np.loadtxt(fn[2]).astype(np.int64)
        # print(point_set.shape)

        # choice = np.random.choice(len(seg), self.npoints, replace=True)
        # resample
        # point_set = point_set[choice, :]
        # seg = seg[choice]
        point_set = torch.from_numpy(point_set)
        # seg = torch.from_numpy(seg)
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))
        if self.classification:
            return point_set, cls
        # else:
        #    return point_set, seg

    def __len__(self):
        return len(self.datapath)

class PcdDataset(data.Dataset):
    def __init__(self, root, npoints = 1024, classification = True, state = 0):
        self.root = root
        self.npoints = npoints
        self.catfile = "" # Your Data Path
        self.cat = {}

        self.classification = classification

        for item in os.listdir(self.catfile):
            self.cat[item] = item

        self.meta = {}
        for item in self.cat:
            self.meta[item] = []
            dir_pcd = os.path.join(self.root, self.cat[item])
            fns = glob.glob(os.path.join(dir_pcd,"*.pcd"))
            random.seed("2022-10-29")
            random.shuffle(fns)
            if state == 0: # train 0.6
                fns = fns[:int(len(fns)*0.6)]
            elif state == 1: # valid 0.2
                fns = fns[int(len(fns)*0.6):int(len(fns)*0.8)]
            else: # test 0.2
                fns = fns[int(len(fns)*0.8):]

            for fn in fns:
                token = os.path.splitext(fn)[0]
                self.meta[item].append((os.path.join(dir_pcd,token+".pcd"),item))
        
        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn[0], fn[1]))
        
        self.classes = dict(zip(sorted(self.cat),range(len(self.cat))))
    
    def __getitem__(self, index):
        fn = self.datapath[index]
        cls = self.classes[self.datapath[index][0]]
        pcd = o3d.io.read_point_cloud(fn[1])
        point_set = np.asarray(pcd.points, dtype=np.float32)
        choice = np.random.choice(len(point_set), self.npoints, replace=True)
        point_set = point_set[choice, :]
        point_set = torch.from_numpy(point_set)
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))
        if self.classification:
            return point_set, cls

    def __len__(self):
        return len(self.datapath)

train_dataset = MeshDataset(
    root=data_path, classification=True, npoints=2048
)
valid_dataset = PcdDataset(
    root=data_path, classification=True, npoints=2048, state=1
)
test_dataset = PcdDataset(
    root=data_path, classification=True, npoints=2048, state=2
)
print("train : %d, valid : %d, test : %d" % (len(train_dataset), len(valid_dataset), len(test_dataset)))

batch_size = 32
trainloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
)
validloader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=batch_size,
    shuffle=True,
)
testloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=True,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device:', device)  # 출력결과: cuda
# 출력결과: 2 (2, 3 두개 사용하므로)
print('Count of using GPUs:', torch.cuda.device_count())
# 출력결과: 2 (2, 3 중 앞의 GPU #2 의미)
print('Current cuda device:', torch.cuda.current_device())


_classifier = PointNetCls(k=len(classes))
classifier = nn.DataParallel(_classifier).to(device)
classifier.load_state_dict(torch.load('')) # Your Path

print('Device:', device)  # 출력결과: cuda
# 출력결과: 2 (2, 3 두개 사용하므로)
print('Count of using GPUs:', torch.cuda.device_count())
# 출력결과: 2 (2, 3 중 앞의 GPU #2 의미)
print('Current cuda device:', torch.cuda.current_device())

optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

num_batch = len(train_dataset) / batch_size

blue = lambda x: "\033[94m" + x + "\033[0m"

best_metric = -1
best_metric_epoch = -1
best_metrics_epochs_and_time = [[], [], []]
epoch_loss_values = []
metric_values = []
import time

max_epochs = 50

total_start = time.time()
for epoch in range(max_epochs):
    epoch_start = time.time()
    print("-" * 10)
    print(f"epoch {epoch+1}/{max_epochs}")
    step = 0
    epoch_loss = 0
    for i, data in enumerate(trainloader, 0):
        step_start = time.time()
        step += 1
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.to(device), target.to(device)
        optimizer.zero_grad()
        classifier = classifier.train()
        # try
        result = classifier(points)
        # pred, trans, trans_feat = classifier(points)
        pred, trans = result[0], result[1]
        loss = F.nll_loss(pred, target)
        loss.backward()
        ###############################################
        epoch_loss += loss.item()
        optimizer.step()
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        print(
            "[%d: %d/%d] train loss: %f accuracy: %f"
            % (epoch, i, num_batch, loss.item(), correct.item() / float(batch_size))
        )
        print(f"step time: {(time.time()-step_start):.4f}")

        if i % 10 == 0:
            metric_sum = 0.0
            metric_count = 0
            j, data = next(enumerate(validloader, 0))
            points, target = data
            target = target[:, 0]
            points = points.transpose(2, 1)
            points, target = points.to(device), target.to(device)
            classifier = classifier.eval()
            pred, _, _ = classifier(points)
            loss = F.nll_loss(pred, target)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            print(
                "[%d: %d/%d] %s loss: %f accuracy: %f"
                % (
                    epoch,
                    i,
                    num_batch,
                    blue("test"),
                    loss.item(),
                    correct.item() / float(batch_size),
                )
            )

    if epoch % 10 == 0:
        torch.save(
            classifier.state_dict(),
            os.path.join(
                "", "fine_tune_mesh_cls_model_%d.pth" % (epoch) 
            ),
        ) # Your Path

    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average train_loss: {epoch_loss:.4f}")
    metric = correct.item() / float(batch_size)
    metric_values.append(metric)
    writer.add_scalar('train_loss',epoch_loss,epoch)
    writer.add_scalar('valid_acc',metric,epoch)
    if metric > best_metric:
        best_metric = metric
        best_metric_epoch = epoch + 1
        torch.save(
            classifier.state_dict(),
            os.path.join(
                "",
                "fine_tune_best_mesh_cls_model_%d.pth" % (best_metric_epoch),
            ),
        )# Your Path

torch.save(
            classifier.state_dict(),
            os.path.join(
                "",
                "fine_tune_mesh_cls_model_%d.pth" % (max_epochs),
            ),
        )# Your Path

writer.flush()
writer.close()

total_correct = 0
total_testset = 0
for i, data in enumerate(testloader, 0):
    points, target = data
    target = target[:, 0]
    points = points.transpose(2, 1)
    points, target = points.to(device), target.to(device)
    classifier = classifier.eval()
    pred, _, _ = classifier(points)
    pred_choice = pred.data.max(1)[1]
    correct = pred_choice.eq(target.data).cpu().sum()
    total_correct += correct.item()
    total_testset += points.size()[0]

print("final accuracy {}".format(total_correct / float(total_testset)))
