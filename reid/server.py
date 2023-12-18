import os
from tkinter import Image
import torch
import copy
from .utils.data import transforms as T 
from .user import DomainLocalUpdate
import torch.nn as nn
from PIL import Image
from torchvision.utils import save_image
from .utils.tools import plotTSNE

# trainers in server side, used for federated setting
class FedDomainMemoTrainer(object):
    def __init__(self, args, user_sets, model, snr=False,
                 cls_params=None, one_cls=False, feature_dim=2048):
        super(FedDomainMemoTrainer, self).__init__()
        self.args = args
        self.model = model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        self.train_transformer = T.Compose([
            T.Resize((args.height, args.width), interpolation=3),
            T.RandomHorizontalFlip(p=0.5),
            T.Pad(10), T.RandomCrop((args.height, args.width)),
            T.ToTensor(), normalizer,
            T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
        ])
        self.test_trans = T.Compose([
            T.Resize((args.height, args.width), interpolation=3),
            T.ToTensor(), normalizer
        ])
        self.user_sets = user_sets
        # number of pids in the dataset
        self.pid_list = [user.num_train_pids for user in user_sets]
        self.max_iter = args.max_iter
        # for fedpav
        if not snr:
            if one_cls:
                # use a global classifier
                self.classifier = []
                self.classifier = nn.Linear(
                    feature_dim, sum(self.pid_list)).to(self.device)
            else:
                # indipendently stored in clients
                self.classifier = []
                for idx in range(len(user_sets)):
                    cur_ids = user_sets[idx].num_train_pids
                    cur_linear = nn.Linear(
                        feature_dim, cur_ids, bias=False).to(self.device)
                    if cls_params is not None:
                        cur_linear.load_state_dict(cls_params[idx])
                    self.classifier.append(cur_linear)
        else:
            self.fc, self.fc1, self.fc2, self.fc3 = [], [], [], []
            if one_cls:
                cur_ids = sum(
                    [cur_set.num_train_pids for cur_set in user_sets])
                self.fc = nn.Linear(feature_dim, cur_ids).to(self.device)
                self.fc1 = nn.Linear(256, cur_ids).to(self.device)
                self.fc2 = nn.Linear(512, cur_ids).to(self.device)
                self.fc3 = nn.Linear(1024, cur_ids).to(self.device)
            else:
                for idx in range(len(user_sets)):
                    cur_ids = user_sets[idx].num_train_pids
                    self.fc.append(
                        nn.Linear(feature_dim, cur_ids).to(self.device))
                    self.fc1.append(nn.Linear(256, cur_ids).to(self.device))
                    self.fc2.append(nn.Linear(512, cur_ids).to(self.device))
                    self.fc3.append(nn.Linear(1024, cur_ids).to(self.device))

    def fed_avg(self, w, weights=None, exclude_set=None):
        sample_count = [len(user.train) for user in self.user_sets]
        # filter out unused sets
        if exclude_set is not None:
            sample_count = [val for (idx, val) in enumerate(sample_count) if idx not in exclude_set]
            w = [we for (idx, we) in enumerate(w) if idx not in exclude_set]
        # avgeraging
        sample_num = sum(sample_count)
        w_avg = copy.deepcopy(w[0])
        for k in w_avg.keys():
            w_avg[k] = w_avg[k] * (sample_count[0] /
                                   sample_num) if weights is None else w_avg[k] * weights[0]
        for k in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[k] = w_avg[k] + w[i][k] * sample_count[i] / sample_num \
                    if weights is None else w_avg[k] + w[i][k] * weights[i]  # weighted model
        return w_avg

    def save_images(self, aug_mod, epoch):
        file_list = [
            'msmt17/MSMT17_V1/bounding_box_train/0781_c11_0015.jpg',
            'msmt17/MSMT17_V1/bounding_box_train/0543_c9_0025.jpg',
            'msmt17/MSMT17_V1/bounding_box_train/0780_c6_0026.jpg',
            'msmt17/MSMT17_V1/bounding_box_train/0006_c1_0013.jpg'
        ]
        file_name = [os.path.join(self.args.data_dir, fname) for fname in file_list]
        for fname in file_name:
            cur_img = self.test_trans(Image.open(fname).convert('RGB')).to(self.device)
            novel_img = aug_mod(cur_img)
            # # denorm
            norm_stat = self.test_trans.transforms[2]
            stat_mean, stat_std = torch.as_tensor(norm_stat.mean).float().to(cur_img.device).view(-1,1,1), \
                torch.as_tensor(norm_stat.std).float().to(cur_img.device).view(-1,1,1)
            novel_img.mul_(stat_std).add_(stat_mean)
            novel_img-=novel_img.min()
            novel_img/=novel_img.max()
            # cur_img.mul_(stat_std).add_(stat_mean)
            
            save_image(cur_img, os.path.join(self.args.logs_dir, os.path.basename(fname)))
            # save_image(novel_img, os.path.join(self.args.logs_dir, 
            #                                 f"{os.path.splitext(os.path.basename(fname))[0]}_{epoch}.jpg"))
        
    def save_tsne(self, file_list, model, aug_mod, epoch):
        model.eval()
        is_trans, features, domains, domain_counter = False, [], [], 0
        for cur_file in file_list:
            print(f"Load data: {cur_file}")
            cur_img = torch.load(cur_file).to(self.device)
            features.append(model(cur_img).detach())
            domains.append([domain_counter,]*cur_img.shape[0])
            domain_counter += 1
            if not is_trans:
                novel_img = aug_mod(cur_img)
                novel_features = model(novel_img).detach()
                features.append(novel_features + 0.03*torch.rand_like(novel_features).to(self.device))
                domains.append([domain_counter,]*novel_img.shape[0])
                is_trans = True
                domain_counter += 1
        features = torch.cat(features, 0).cpu().numpy()
        domains = torch.tensor(domains).view(-1).numpy()
        plotTSNE(features, domains, os.path.join(self.args.logs_dir, f"tsne_{epoch}.jpg"), epoch)
    
    # resnet50
    def train_cls(self, net, epoch, client_id, op_type='sgd'):
        net.train(True)
        local = DomainLocalUpdate(args=self.args,
                                  dataset=self.user_sets[client_id],
                                  trans=self.train_transformer)
        w = local.train_cls(net=net, global_epoch=epoch, client_id=client_id,
                            cls_layer=self.classifier[client_id], op_type=op_type)
        return w
    
    # resnet50
    def train_mixstyle(self, net, epoch, client_id, op_type='sgd'):
        net.train(True)
        local = DomainLocalUpdate(args=self.args,
                                  dataset=self.user_sets[client_id],
                                  trans=self.train_transformer)
        w = local.train_mixstyle(net=net, global_epoch=epoch, client_id=client_id, 
                                 cls_layer=self.classifier[client_id], op_type=op_type)
        return w
    
    def train_crossstyle(self, net, epoch, client_id, op_type='sgd'):
        net.train(True)
        local = DomainLocalUpdate(args=self.args,
                                  dataset=self.user_sets[client_id],
                                  trans=self.train_transformer)
        w = local.train_crossstyle(net=net, global_epoch=epoch, client_id=client_id, 
                                   cls_layer=self.classifier[client_id], op_type=op_type)
        return w

    def train_dacs(self, net, avg_net, aug_mod, epoch, client_id, op_type='sgd'):
        net.train(True) # local model, not shareable
        # warning! you must deep copy avg_net to ensure cur_avg is the re-distributed version 
        # (not the updated version received from last domain's local training)
        cur_avg = copy.deepcopy(avg_net) # the local-side global model
        cur_avg.train(True)
        local = DomainLocalUpdate(args=self.args, dataset=self.user_sets[client_id],
                                  trans=self.train_transformer)
        # use other data to aug local data, using the re-distributed version of avg model
        w = local.train_dacs(net=net, avg_net=cur_avg, aug_mod=aug_mod, 
                            global_epoch=epoch, client_id=client_id, 
                            cls_layer=self.classifier[client_id], op_type=op_type)
        return w
    
    def train_moon(self, net, prev_net, avg_net, epoch, client_id, op_type='sgd'):
        net.train(True)
        # cur_avg--global model, 
        avg_net.eval()
        prev_net.eval()
        
        local = DomainLocalUpdate(args=self.args, dataset=self.user_sets[client_id],
                                  trans=self.train_transformer)
        # use other data to aug local data
        w = local.train_moon(net=net, prev_net=prev_net, avg_net=avg_net,
                             global_epoch=epoch, client_id=client_id, 
                             cls_layer=self.classifier[client_id], op_type=op_type)
        return w

    def train_free_dacs(self, net, aug_mod, epoch, client_id, op_type='sgd'):
        net.train(True)

        local = DomainLocalUpdate(args=self.args,
                                  dataset=self.user_sets[client_id],
                                  trans=self.train_transformer)
        # use other data to aug local data
        w = local.train_free_dacs(net=net, aug_mod=aug_mod,
                                  global_epoch=epoch, client_id=client_id,
                                  cls_layer=self.classifier[client_id], op_type=op_type)
        return w

    # resnet50 upper
    def train_cls_nofed_sepcls(self, net, epoch, op_type='sgd'):
        net.train(True)
        local = DomainLocalUpdate(args=self.args,
                                  dataset=self.user_sets,
                                  trans=self.train_transformer)
        w = local.train_cls_nofed_sepcls(net=net, global_epoch=epoch, 
                                         cls_layer=self.classifier, op_type=op_type)
        return w

    # distill, scarfold
    def train_reid(self, net, avg_net, epoch, client_id):
        net.train(True)
        cur_avg = copy.deepcopy(avg_net)
        cur_avg.train(True)
        local = DomainLocalUpdate(args=self.args,
                                  dataset=self.user_sets[client_id],
                                  trans=self.train_transformer)

        w = local.train_fedreid(net=net, avg_net=cur_avg,
                                global_epoch=epoch, client_id=client_id,
                                cls_layer=self.classifier[client_id])
        return w

    def train_scar(self, net, avg_net, epoch,
                   client_id, server_control, client_control):
        net.train(True)
        avg_net.train(False)

        local = DomainLocalUpdate(args=self.args, dataset=self.user_sets[client_id],
                                  trans=self.train_transformer)
        # local train
        w = local.train_scar(net=net, avg_net=avg_net,
                             global_epoch=epoch, client_id=client_id,
                             cls_layer=self.classifier[client_id],
                             server_control=server_control,
                             client_control=client_control[client_id])
        return w

    # training with snr
    def train_snr(self, sub_model, epoch, client_id, op_type='sgd'):
        sub_model.train(True)
        local = DomainLocalUpdate(args=self.args,
                                  dataset=self.user_sets[client_id],
                                  trans=self.train_transformer)
        w = local.train_snr(net=sub_model, global_epoch=epoch, client_id=client_id,
                            fc=self.fc[client_id], fc1=self.fc1[client_id],
                            fc2=self.fc2[client_id], fc3=self.fc3[client_id],
                            op_type=op_type)
        return w

    def train_dacs_snr(self, sub_model, avg_net, aug_mod,
                        epoch, client_id, op_type='sgd'):
        sub_model.train(True)
        cur_avg = copy.deepcopy(avg_net)
        local = DomainLocalUpdate(args=self.args,
                                  dataset=self.user_sets[client_id],
                                  trans=self.train_transformer)
        w = local.train_dacs_snr(net=sub_model, avg_net=cur_avg, aug_mod=aug_mod,
                                  global_epoch=epoch, client_id=client_id,
                                  fc=self.fc[client_id], fc1=self.fc1[client_id],
                                  fc2=self.fc2[client_id], fc3=self.fc3[client_id],
                                  op_type=op_type)
        return w

