from torch.utils.data import DataLoader
from .utils.data import IterLoader, Preprocessor
import torch
from .utils.data.sampler import RandomMultipleGallerySampler
from .utils.tools import get_entropy, get_auth_loss, ScaffoldOptimizer, cn_op_2ins_space_chan, freeze_model, inception_score
from .loss import TripletLoss
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
import os
import copy
from pytorch_msssim import ssim
import numpy as np


# trainers in user side
class DomainLocalUpdate(object):
    def __init__(self, args, dataset=None, trans=None):
        self.args = args
        self.trans = trans
        # only for non-qaconv algos
        if dataset is not None:
            if not isinstance(dataset, list):
                self.local_train = IterLoader(DataLoader(
                    Preprocessor(dataset.train, transform=trans, root=None),
                    batch_size=self.args.batch_size, shuffle=False, drop_last=True,
                    sampler=RandomMultipleGallerySampler(
                        dataset.train, args.num_instances),
                    pin_memory=False, num_workers=self.args.num_workers
                ), length=None)
                self.set_name = dataset.__class__.__name__
            else:
                self.local_train = [IterLoader(DataLoader(
                    Preprocessor(cur_set.train, transform=trans, root=None),
                    batch_size=self.args.batch_size, shuffle=False, drop_last=True,
                    sampler=RandomMultipleGallerySampler(
                        cur_set.train, args.num_instances),
                    pin_memory=False, num_workers=self.args.num_workers
                ), length=None) for cur_set in dataset]
                pid_list = [user.num_train_pids for user in dataset]
                self.padding = np.cumsum([0, ]+pid_list)
        self.max_iter = args.max_iter
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tri_loss = TripletLoss(margin=0.5, is_avg=True)
        self.ce_loss = torch.nn.CrossEntropyLoss(reduce='mean')
        self.dataset = dataset

    def handle_set(self, dataset):
        cur_loader = IterLoader(DataLoader(
            Preprocessor(dataset.train, transform=self.trans, root=None),
            batch_size=self.args.batch_size, shuffle=False, drop_last=True,
            sampler=RandomMultipleGallerySampler(
                dataset.train, self.args.num_instances),
            pin_memory=True, num_workers=self.args.num_workers
        ), length=None)
        return cur_loader

    def get_optimizer(self, nets, epoch, optimizer_type='sgd'):
        if optimizer_type.lower() == 'sgd':
            optimizer = torch.optim.SGD(
                [{'params': sub_net.parameters()} for sub_net in nets],
                lr=self.args.lr, weight_decay=self.args.weight_decay,
                momentum=self.args.momentum
            )
            lr_scheduler = MultiStepLR(
                optimizer, milestones=self.args.milestones, gamma=0.5)
        elif optimizer_type.lower() == 'scaffold':
            optimizer = ScaffoldOptimizer(
                [{'params': sub_net.parameters()} for sub_net in nets],
                lr=self.args.lr, weight_decay=self.args.weight_decay
            )
            lr_scheduler = MultiStepLR(optimizer,
                                       milestones=self.args.milestones, gamma=0.5)
        lr_scheduler.step(epoch)
        return optimizer

    # resnet50, mAP=26.7
    def train_cls(self, net, global_epoch,
                  client_id, cls_layer, op_type='sgd'):
        net.train(True)
        optimizer = self.get_optimizer([net, cls_layer], global_epoch,
                                       optimizer_type=op_type)
        self.local_train.new_epoch()
        # local train, each contains local_ep epochs
        for batch_idx in range(self.max_iter):
            (images, _, labels, _, _) = self.local_train.next()
            images, labels = images.cuda(), labels.cuda()
            feature = net(images)[0]
            score = cls_layer(feature)
            loss_ce = self.ce_loss(score, labels)
            loss_tri = self.tri_loss(feature, labels)

            loss = loss_ce + loss_tri
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f'Update Epoch / Total Epoch: [{global_epoch}/{self.args.epochs}]. Net Client: {client_id}. '
                  f'Iter / Total Iter: [{batch_idx + 1}/{self.max_iter}] (LossCE: {loss_ce.item():.2f}, '
                  f'LossTri: {loss_tri.item():.2f})')
        return net.state_dict()
    
    # resnet50, style
    def train_mixstyle(self, net, global_epoch,
                       client_id, cls_layer, op_type='sgd'):
        net.train(True)
        optimizer = self.get_optimizer([net, cls_layer], global_epoch,
                                       optimizer_type=op_type)
        self.local_train.new_epoch()
        # local train, each contains local_ep epochs
        for batch_idx in range(self.max_iter):
            (images, _, labels, _, _) = self.local_train.next()
            images, labels = images.cuda(), labels.cuda()
            import ipdb;ipdb.set_trace()
            feature = net(images)
            score = cls_layer(feature)
            loss_ce = self.ce_loss(score, labels)
            loss_tri = self.tri_loss(feature, labels)
            loss = loss_ce + loss_tri
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f'Update Epoch / Total Epoch: [{global_epoch}/{self.args.epochs}]. Net Client: {client_id}. '
                  f'Iter / Total Iter: [{batch_idx + 1}/{self.max_iter}] (LossCE: {loss_ce.item():.2f}, '
                  f'LossTri: {loss_tri.item():.2f})')
        return net.state_dict()

    def train_crossstyle(self, net, global_epoch,
                         client_id, cls_layer, op_type='sgd'):
        net.train(True)
        optimizer = self.get_optimizer([net, cls_layer], global_epoch,
                                       optimizer_type=op_type)
        self.local_train.new_epoch()
        # local train, each contains local_ep epochs
        for batch_idx in range(self.max_iter):
            (images, _, labels, _, _) = self.local_train.next()
            images, labels = images.cuda(), labels.cuda()
            images = cn_op_2ins_space_chan(images, beta=0.5, crop='both')
            feature = net(images)[0]
            score = cls_layer(feature)
            loss_ce = self.ce_loss(score, labels)
            loss_tri = self.tri_loss(feature, labels)
            loss = loss_ce + loss_tri
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f'Update Epoch / Total Epoch: [{global_epoch}/{self.args.epochs}]. Net Client: {client_id}. '
                  f'Iter / Total Iter: [{batch_idx + 1}/{self.max_iter}] (LossCE: {loss_ce.item():.2f}, '
                  f'LossTri: {loss_tri.item():.2f})')
        return net.state_dict()

    # vanilla aug, mAP=34.2
    def train_dacs(self, net, avg_net, aug_mod, global_epoch, client_id,
                           cls_layer, op_type='sgd'):
        net.train(True)
        avg_net.train(True)

        self.local_train.new_epoch()
        # avg optimizer
        optimizer = self.get_optimizer(
            nets=[avg_net, cls_layer, aug_mod, ], epoch=global_epoch,
            optimizer_type=op_type
        )
        # local optimizer
        optimizer_local = self.get_optimizer(
            nets=[net, ], epoch=global_epoch,
            optimizer_type=op_type
        )
        # ssim_scores, is_epoch = [], 0
        for batch_idx in range(self.max_iter):
            (images, _, labels, _, _) = self.local_train.next()
            images, labels = images.cuda(), labels.cuda()
            # generate data stats to normalize
            b_size = images.shape[0]
            cur_mean, cur_var = images.mean((2, 3)).view(
                b_size, -1, 1, 1), images.var((2, 3)).view(b_size, -1, 1, 1)
            norm_image = (images-cur_mean).div(cur_var.sqrt()+1e-8)

            # stage1: expert train
            feature, feature_avg = net(images)[0], avg_net(images)[0]
            score, score_avg = cls_layer(feature), cls_layer(feature_avg)
            loss_erm = self.ce_loss(score, labels) + self.tri_loss(feature, labels)
            optimizer_local.zero_grad()
            loss_erm.backward()
            optimizer_local.step()

            # stage2: joint train
            # basic cls loss with ori images and avg_net
            loss_ce = self.ce_loss(score_avg, labels)
            loss_tri = self.tri_loss(feature_avg, labels)
            loss_aux, loss_aug, loss_wd = 0, 0, 0

            # aug avg model
            if global_epoch > 0:
                # generate freezed global model to detach grad, training=False version
                freeze_avg = freeze_model(copy.deepcopy(avg_net)) 
                # transformed image
                aug_image = aug_mod(norm_image)
                # ssim_scores.append(ssim(aug_image.detach(), images.detach(), data_range=1, size_average=True).item())

                # obtain H(fG(x')), use a frozen avg_net to avoid updating avg_net model
                aug_feature_avg_freeze = freeze_avg(aug_image) 
                aug_score_avg_freeze = cls_layer(aug_feature_avg_freeze)
                # obtain H(fL(x')), optimizer does not contain net.params(), so we do not need to use 'freeze_model'
                aug_feature_local = net(aug_image)[0]
                aug_score_local = cls_layer(aug_feature_local)
                # generate H(fG(x)), use a frozen avg_net
                score_avg_freeze = cls_layer(freeze_avg(images))
                # au loss,  H(fG(x)) < H(fG(x')) < H(fL(x'))
                loss_aux = get_auth_loss(
                    get_entropy(F.softmax(aug_score_avg_freeze)),
                    get_entropy(F.softmax(score_avg_freeze)),
                    get_entropy(F.softmax(aug_score_local))
                )
                
                # aug images to update avg_net
                aug_feature_avg = avg_net(aug_image)[0]
                aug_score_avg = cls_layer(aug_feature_avg)
                loss_aug = self.ce_loss(aug_score_avg, labels) + self.tri_loss(aug_feature_avg, labels)
                
                # div loss
                shift_mean, shift_var = aug_mod.get_mean_var()
                loss_wd = -F.mse_loss(shift_mean, cur_mean) - \
                    F.mse_loss(cur_var, shift_var)

            # optimize avg model, share across domains
            loss = loss_ce + loss_tri + loss_aug + loss_wd + self.args.lam * loss_aux
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f'Update Epoch / Total Epoch: [{global_epoch}/{self.args.epochs}]. Net Client: {client_id}. '
                  f'Iter / Total Iter: [{batch_idx + 1}/{self.max_iter}] (LossCE: {loss_ce.item():.2f}, '
                  f'LossTri: {loss_tri.item():.2f}, LossAux:{float(loss_aux):.2f})')
        # # print ssim
        # ssim_epoch = max(ssim_scores) if len(ssim_scores) else 0
        # is_epoch = inception_score(aug_image.detach()) if global_epoch>0 else 0

        # if global_epoch % 5 == 0:
        #     print(f'Dataset {self.set_name}. SSIM: {ssim_epoch:4.3f}, IS: {is_epoch:4.3f}.')

        return avg_net.state_dict()

    # vanilla aug, mAP=26
    def train_moon(self, net, prev_net, avg_net, global_epoch, client_id,
                   cls_layer, op_type='sgd'):
        net.train(True)
        avg_net.eval()
        prev_net.eval()
        self.local_train.new_epoch()

        cos_func = torch.nn.CosineSimilarity(dim=-1)
        # local optimizer
        optimizer_local = self.get_optimizer(
            nets=[net, cls_layer], epoch=global_epoch,
            optimizer_type=op_type
        )
        for batch_idx in range(self.max_iter):
            (images, _, labels, _, _) = self.local_train.next()
            images, labels = images.cuda(), labels.cuda()
            # basic cls loss for local model
            feature = net(images)[0]
            score = cls_layer(feature)
            loss_ce = self.ce_loss(score, labels)
            loss_tri = self.tri_loss(feature, labels)
            loss_con = 0

            # aug avg model
            if global_epoch > 0:
                avg_feat = avg_net(images)
                prev_feat = prev_net(images)

                score_pos = cos_func(feature, avg_feat).reshape(-1, 1)
                score_neg = cos_func(feature, prev_feat).reshape(-1, 1)

                denominator_score = torch.cat(
                    [score_pos, score_neg], dim=1) / self.args.temp
                con_labels = torch.zeros(
                    score_pos.shape[0]).to(self.device).long()
                loss_con = F.cross_entropy(denominator_score, con_labels)

            # optimize avg model, sahre across domains
            loss = loss_ce + loss_tri + self.args.lam * loss_con
            optimizer_local.zero_grad()
            loss.backward()
            optimizer_local.step()

            print(f'Update Epoch / Total Epoch: [{global_epoch}/{self.args.epochs}]. Net Client: {client_id}. '
                  f'Iter / Total Iter: [{batch_idx + 1}/{self.max_iter}] (LossCE: {loss_ce.item():.2f}, '
                  f'LossTri: {loss_tri.item():.2f}, LossCon:{float(loss_con):.2f})')

        return net.state_dict()

    def train_dacs_snr(self, net, avg_net, aug_mod,
                        global_epoch, client_id,
                        fc, fc1, fc2, fc3, op_type='sgd'):
        net.train(True)
        avg_net.train(True)

        self.local_train.new_epoch()
        # avg optimizer
        optimizer = self.get_optimizer(
            nets=[avg_net, fc, fc1, fc2, fc3, aug_mod],
            epoch=global_epoch, optimizer_type=op_type
        )
        # local optimizer
        optimizer_local = self.get_optimizer(
            nets=[net, ], epoch=global_epoch,
            optimizer_type=op_type
        )
        # local train, each contains local_ep epochs
        for batch_idx in range(self.max_iter):
            (images, _, labels, _, _) = self.local_train.next()
            images, labels = images.cuda(), labels.cuda()
            # generate data stats to normalize
            cur_mean, cur_var = images.mean(0), images.var(0)
            norm_image = (images-cur_mean).div(cur_var.sqrt()+1e-8)

            # fine tune local snr
            local_features, x_IN_1_pool, x_1_useful_pool, x_1_useless_pool, \
                x_IN_2_pool, x_2_useful_pool, x_2_useless_pool, \
                x_IN_3_pool, x_3_useful_pool, x_3_useless_pool = net(images)
            x_IN_1_prob = F.softmax(fc1(x_IN_1_pool))
            x_1_useful_prob = F.softmax(fc1(x_1_useful_pool))
            x_1_useless_prob = F.softmax(fc1(x_1_useless_pool))
            x_IN_2_prob = F.softmax(fc2(x_IN_2_pool))
            x_2_useful_prob = F.softmax(fc2(x_2_useful_pool))
            x_2_useless_prob = F.softmax(fc2(x_2_useless_pool))
            x_IN_3_prob = F.softmax(fc3(x_IN_3_pool))
            x_3_useful_prob = F.softmax(fc3(x_3_useful_pool))
            x_3_useless_prob = F.softmax(fc3(x_3_useless_pool))
            local_score = fc(local_features)
            loss_causality = 0.01 * get_auth_loss(get_entropy(x_IN_1_prob), get_entropy(x_1_useful_prob), get_entropy(x_1_useless_prob)) + \
                0.01 * get_auth_loss(get_entropy(x_IN_2_prob), get_entropy(x_2_useful_prob), get_entropy(x_2_useless_prob)) + \
                0.01 * get_auth_loss(get_entropy(x_IN_3_prob), get_entropy(
                    x_3_useful_prob), get_entropy(x_3_useless_prob))
            loss = loss_causality + \
                self.tri_loss(local_features, labels) + \
                self.ce_loss(local_score, labels)
            optimizer_local.zero_grad()
            loss.backward()
            optimizer_local.step()

            # basic cls loss for avg model
            feature_avg, x_IN_1_pool, x_1_useful_pool, x_1_useless_pool, \
                x_IN_2_pool, x_2_useful_pool, x_2_useless_pool, \
                x_IN_3_pool, x_3_useful_pool, x_3_useless_pool = avg_net(
                    images)
            x_IN_1_prob = F.softmax(fc1(x_IN_1_pool))
            x_1_useful_prob = F.softmax(fc1(x_1_useful_pool))
            x_1_useless_prob = F.softmax(fc1(x_1_useless_pool))
            x_IN_2_prob = F.softmax(fc2(x_IN_2_pool))
            x_2_useful_prob = F.softmax(fc2(x_2_useful_pool))
            x_2_useless_prob = F.softmax(fc2(x_2_useless_pool))
            x_IN_3_prob = F.softmax(fc3(x_IN_3_pool))
            x_3_useful_prob = F.softmax(fc3(x_3_useful_pool))
            x_3_useless_prob = F.softmax(fc3(x_3_useless_pool))
            score_avg = fc(feature_avg)
            # Causality loss for avg model:
            loss_causality_avg = 0.01 * get_auth_loss(get_entropy(x_IN_1_prob), get_entropy(x_1_useful_prob), get_entropy(x_1_useless_prob)) + \
                0.01 * get_auth_loss(get_entropy(x_IN_2_prob), get_entropy(x_2_useful_prob), get_entropy(x_2_useless_prob)) + \
                0.01 * get_auth_loss(get_entropy(x_IN_3_prob), get_entropy(
                    x_3_useful_prob), get_entropy(x_3_useless_prob))
            score_avg = fc(feature_avg)
            loss_ce_avg = self.ce_loss(score_avg, labels)
            loss_tri_avg = self.tri_loss(feature_avg, labels)
            loss_aug, loss_aux_avg = 0, 0

            # aug avg model
            if global_epoch > 0:
                aug_image = aug_mod(norm_image)

                aug_feature_avg, x_IN_1_pool_aug, x_1_useful_pool_aug, x_1_useless_pool_aug, \
                    x_IN_2_pool_aug, x_2_useful_pool_aug, x_2_useless_pool_aug, \
                    x_IN_3_pool_aug, x_3_useful_pool_aug, x_3_useless_pool_aug = avg_net(
                        aug_image)
                x_IN_1_prob_aug = F.softmax(fc1(x_IN_1_pool_aug))
                x_1_useful_prob_aug = F.softmax(fc1(x_1_useful_pool_aug))
                x_1_useless_prob_aug = F.softmax(fc1(x_1_useless_pool_aug))
                x_IN_2_prob_aug = F.softmax(fc2(x_IN_2_pool_aug))
                x_2_useful_prob_aug = F.softmax(fc2(x_2_useful_pool_aug))
                x_2_useless_prob_aug = F.softmax(fc2(x_2_useless_pool_aug))
                x_IN_3_prob_aug = F.softmax(fc3(x_IN_3_pool_aug))
                x_3_useful_prob_aug = F.softmax(fc3(x_3_useful_pool_aug))
                x_3_useless_prob_aug = F.softmax(fc3(x_3_useless_pool_aug))
                loss_aug_causality = 0.01 * get_auth_loss(get_entropy(x_IN_1_prob_aug), get_entropy(x_1_useful_prob_aug), get_entropy(x_1_useless_prob_aug)) + \
                    0.01 * get_auth_loss(get_entropy(x_IN_2_prob_aug), get_entropy(x_2_useful_prob_aug), get_entropy(x_2_useless_prob_aug)) + \
                    0.01 * get_auth_loss(get_entropy(x_IN_3_prob_aug), get_entropy(
                        x_3_useful_prob_aug), get_entropy(x_3_useless_prob_aug))

                aug_feature_local = net(aug_image)[0]
                aug_score_avg, aug_score_local = fc(
                    aug_feature_avg), fc(aug_feature_local)
                # loss to disentangle, fL(I) < fA(I) < fA(I') < fL(I')
                loss_aux_avg = get_auth_loss(
                    get_entropy(F.softmax(aug_score_avg)),
                    get_entropy(F.softmax(score_avg)),
                    get_entropy(F.softmax(aug_score_local))
                )
                loss_aug = self.ce_loss(
                    aug_score_avg, labels) + self.tri_loss(aug_feature_avg, labels) + loss_aug_causality

            # optimize avg model, sahre across domains
            loss = loss_ce_avg + loss_tri_avg + loss_causality_avg + \
                loss_aug + self.args.lam * loss_aux_avg
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f'Update Epoch / Total Epoch: [{global_epoch}/{self.args.epochs}]. Net Client: {client_id}. '
                  f'Iter / Total Iter: [{batch_idx + 1}/{self.max_iter}] (LossCE: {loss_ce_avg.item():.2f}, '
                  f'LossTri: {loss_tri_avg.item():.2f}, LossAux:{float(loss_aux_avg):.2f})')

        return avg_net.state_dict()

    # vanilla aug, mAP=31.9
    def train_free_dacs(self, net, aug_mod, global_epoch, client_id,
                        cls_layer, op_type='sgd'):
        net.train(True)
        self.local_train.new_epoch()
        # avg optimizer
        optimizer = self.get_optimizer(
            nets=[net, cls_layer, aug_mod], epoch=global_epoch,
            optimizer_type=op_type
        )
        # local train, each contains local_ep epochs
        for batch_idx in range(self.max_iter):
            (images, _, labels, _, _) = self.local_train.next()
            images, labels = images.cuda(), labels.cuda()
            # generate data stats to normalize
            cur_mean, cur_var = images.mean(0), images.var(0)
            norm_image = (images-cur_mean).div(cur_var.sqrt()+1e-8)

            # basic cls loss for local model
            feature = net(images)[0]
            score = cls_layer(feature)
            loss_tri = self.tri_loss(feature, labels)
            loss_ce = self.ce_loss(score, labels)
            loss = loss_ce + loss_tri
            loss_aux, loss_aug = 0, 0

            # aug avg model
            if global_epoch > 0:
                aug_image = aug_mod(norm_image)
                aug_feature_avg = net(aug_image)[0]
                aug_score_avg = cls_layer(aug_feature_avg)
                aug_mean, aug_var = aug_mod.get_mean_var()
                loss_aux = -(F.mse_loss(cur_mean, aug_mean) +
                             F.mse_loss(cur_var, aug_var))
                loss_aug = self.ce_loss(
                    aug_score_avg, labels) + self.tri_loss(aug_feature_avg, labels)

            loss = loss + loss_aug + self.args.lam * loss_aux
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f'Update Epoch / Total Epoch: [{global_epoch}/{self.args.epochs}]. Net Client: {client_id}. '
                  f'Iter / Total Iter: [{batch_idx + 1}/{self.max_iter}] (LossCE: {loss_ce.item():.2f}, '
                  f'LossTri: {loss_tri.item():.2f}, LossAux:{float(loss_aux):.2f})')

        return net.state_dict()

    # nofed resnet50, mAP=33
    def train_cls_nofed_sepcls(self, net, global_epoch,
                               cls_layer, op_type='sgd'):
        net.train(True)
        optimizer = self.get_optimizer(
            [net, cls_layer, ], global_epoch, optimizer_type=op_type
        )
        [daset.new_epoch() for daset in self.local_train]
        num_domains = len(self.local_train)

        # local train, each contains local_ep epochs
        for batch_idx in range(self.max_iter):
            loss = 0
            for client_id in range(num_domains):
                padding = self.padding[client_id]
                (images, _, labels, _, _) = self.local_train[client_id].next()
                images, labels = images.cuda(), labels.cuda() + padding
                feature = net(images)[0]
                score = cls_layer(feature)
                loss_tri = self.tri_loss(feature, labels)
                loss_ce = self.ce_loss(score, labels)

                loss += loss_ce + loss_tri

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f'Update Epoch / Total Epoch: [{global_epoch}/{self.args.epochs}].'
                  f'Iter / Total Iter: [{batch_idx + 1}/{self.max_iter}] (LossCE: {loss_ce.item():.2f}, '
                  f'LossTri: {loss_tri.item():.2f})')
        return net.state_dict()

    # others, distill + scarfold
    def train_fedreid(self, net, avg_net, global_epoch, client_id, cls_layer):
        net.train(True)
        avg_net.train(True)
        optimizer = self.get_optimizer([net, ], global_epoch)
        optim_avg = self.get_optimizer([avg_net, cls_layer], global_epoch)

        self.local_train.new_epoch()
        # local train, each contains local_ep epochs
        for batch_idx in range(self.max_iter):
            (images, _, labels, _, _) = self.local_train.next()
            images, labels = images.cuda(), labels.cuda()

            # extract features from avg model and vanilla model
            feature = net(images)[0]
            score = cls_layer(feature)
            loss_ce = self.ce_loss(score, labels)
            loss_tri = self.tri_loss(feature, labels)
            loss = loss_ce + loss_tri
            loss_kl = 0
            # update local model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if global_epoch > 0:
                feat_avg = avg_net(images)[0]
                score_avg = cls_layer(feat_avg)
                loss_ce = self.ce_loss(score_avg, labels)
                loss_tri = self.tri_loss(feat_avg, labels)
                score = score.detach()  # optim avg only
                loss_kl = (F.softmax(score, 1)*F.log_softmax(score, 1)).sum(1).mean() -\
                    (F.softmax(score, 1) * F.log_softmax(score_avg, 1)).sum(1).mean()
                # update local model
                loss_consist = loss_ce + loss_tri + self.args.temp**2 * loss_kl
                optim_avg.zero_grad()
                loss_consist.backward()
                optim_avg.step()

            print(f'Update Epoch / Total Epoch: [{global_epoch}/{self.args.epochs}]. Net Client: {client_id}. '
                  f'Iter / Total Iter: [{batch_idx + 1}/{self.max_iter}] (LossCE: {loss_ce.item():.2f}, '
                  f'LossTri: {loss_tri.item():.2f}, LossKL: {float(loss_kl):.2f})')

        return avg_net.state_dict() if global_epoch > 0 else net.state_dict()

    def train_scar(self, net, avg_net, global_epoch, client_id, cls_layer,
                   client_control, server_control):
        net.train(True)
        avg_net.train(False)
        optimizer = self.get_optimizer(
            [net, cls_layer], global_epoch,
            optimizer_type='scaffold'
        )

        self.local_train.new_epoch()
        # local train, each contains local_ep epochs
        for batch_idx in range(self.max_iter):
            (images, _, labels, _, _) = self.local_train.next()
            images, labels = images.cuda(), labels.cuda()

            # extract features from avg model and vanilla model
            feature = net(images)[0]
            score = cls_layer(feature)

            # per-sample loss
            loss_ce = self.ce_loss(score, labels)
            loss_tri = self.tri_loss(feature, labels)
            loss = loss_tri + loss_ce

            # update local model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step(
                client_controls=client_control,
                server_controls=server_control,
                global_epoch=global_epoch
            )

            print(f'Update Epoch / Total Epoch: [{global_epoch}/{self.args.epochs}]. Net Client: {client_id}. '
                  f'Iter / Total Iter: [{batch_idx + 1}/{self.max_iter}] (LossCE: {loss_ce.mean().item():.2f}, '
                  f'LossTri: {loss_tri.item():.2f})')
        return net.state_dict()

    # snr version
    def train_snr(self, net, global_epoch, client_id,
                  fc, fc1, fc2, fc3, op_type='sgd'):
        # preparation
        net.train(True)
        optimizer = self.get_optimizer(
            [net, fc, fc1, fc2, fc3], global_epoch,
            optimizer_type=op_type
        )
        self.local_train.new_epoch()
        # local train, each contains local_ep epochs
        for batch_idx in range(self.max_iter):
            (images, _, labels, _, _) = self.local_train.next()
            images, labels = images.cuda(), labels.cuda()

            # forward with the original parameters
            features, x_IN_1_pool, x_1_useful_pool, x_1_useless_pool, \
                x_IN_2_pool, x_2_useful_pool, x_2_useless_pool, \
                x_IN_3_pool, x_3_useful_pool, x_3_useless_pool = net(images)

            x_IN_1_prob = F.softmax(fc1(x_IN_1_pool))
            x_1_useful_prob = F.softmax(fc1(x_1_useful_pool))
            x_1_useless_prob = F.softmax(fc1(x_1_useless_pool))

            x_IN_2_prob = F.softmax(fc2(x_IN_2_pool))
            x_2_useful_prob = F.softmax(fc2(x_2_useful_pool))
            x_2_useless_prob = F.softmax(fc2(x_2_useless_pool))

            x_IN_3_prob = F.softmax(fc3(x_IN_3_pool))
            x_3_useful_prob = F.softmax(fc3(x_3_useful_pool))
            x_3_useless_prob = F.softmax(fc3(x_3_useless_pool))

            # Causality loss:
            loss_causality = get_auth_loss(get_entropy(x_IN_1_prob), get_entropy(x_1_useful_prob), get_entropy(x_1_useless_prob)) + \
                get_auth_loss(get_entropy(x_IN_2_prob), get_entropy(x_2_useful_prob), get_entropy(x_2_useless_prob)) + \
                get_auth_loss(get_entropy(x_IN_3_prob), get_entropy(
                    x_3_useful_prob), get_entropy(x_3_useless_prob))
            # common loss
            score = fc(features)
            loss = self.ce_loss(score, labels) + loss_causality

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f'Update Epoch / Total Epoch: [{global_epoch}/{self.args.epochs}]. Client: {client_id}. '
                  f'Iter / Total Iter: [{batch_idx + 1}/{self.max_iter}] (Loss: {loss.item():.4f})')
        return net.state_dict()

