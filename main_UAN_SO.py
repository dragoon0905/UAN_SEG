import os
import random
import logging
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm, trange
import numpy as np
from torch.autograd import Variable
import hydra
from omegaconf import OmegaConf, DictConfig
from torch.utils.tensorboard import SummaryWriter

from datasets.cityscapes_Dataset import City_Dataset, inv_preprocess, decode_labels
from datasets.gta5_Dataset import GTA5_Dataset
from datasets.synthia_Dataset import SYNTHIA_Dataset
from perturbations.augmentations import augment, get_augmentation
from perturbations.fourier import fourier_mix
from perturbations.cutmix import cutmix_combine
from models import get_model
from models.ema import EMA
from utils.eval import Eval, synthia_set_16, synthia_set_13

from utils.util import MemoryQueue
from utils.lib import seed_everything, sinkhorn, ubot_CCD, adaptive_filling
import ot
from net import *
from lib import *
palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)



def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask


class TotalNet(nn.Module):
    def __init__(self):
        super(TotalNet, self).__init__()
        #self.feature_extractor = model_dict['resnet50']('/data/dragoon0905/UniDA_methods/UAN/resnet50.pth')
        classifier_output_dim = 13
        self.classifier = CLS2(classifier_output_dim, bottle_neck_dim=256)
        self.discriminator = AdversarialNetwork(256)
        self.discriminator_separate = AdversarialNetwork(256)

    def forward(self, x):
        #f = self.feature_extractor(x)
        f, _, __, y = self.classifier(f)
        d = self.discriminator(_)
        d_0 = self.discriminator_separate(_)
        return y, d, d_0

class AdversarialNetwork(nn.Module):
    """
    AdversarialNetwork with a gredient reverse layer.
    its ``forward`` function calls gredient reverse layer first, then applies ``self.main`` module.
    """
    def __init__(self, in_feature):
        super(AdversarialNetwork, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(in_feature, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024,1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
        self.grl = GradientReverseModule(lambda step: aToBSheduler(step, 0.0, 1.0, gamma=10, max_iter=10000))

    def forward(self, x):
        x_ = self.grl(x)
        y = self.main(x_)
        return y



def weight_norm(w):
    norm = w.norm(p=2, dim=1, keepdim=True)
    w = w.div(norm.expand_as(w))
    return w  
class Trainer():
    def __init__(self, cfg, logger, writer):

        # Args
        self.cfg = cfg
        self.device = torch.device('cuda')
        self.logger = logger
        self.writer = writer

        # Counters
        self.epoch = 0
        self.iter = 0
        self.current_MIoU = 0
        self.best_MIou = 0
        self.best_source_MIou = 0

        # Metrics
        self.evaluator = Eval(self.cfg.data.num_classes)

        # Loss
        self.ignore_index = -1
        self.loss = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

        # Model
        self.model, params = get_model(self.cfg)
        # self.model = nn.DataParallel(self.model, device_ids=[0])  # TODO: test multi-gpu
        self.model.to(self.device)

        # EMA
        self.ema = EMA(self.model, self.cfg.ema_decay)

        self.k=cfg.k
        # UniOT
        self.num_classes=cfg.data.num_classes

        
        # Optimizer
        #if self.cfg.opt.kind == "SGD":
        #    self.optimizer = torch.optim.SGD(
        #        params, momentum=self.cfg.opt.momentum, weight_decay=self.cfg.opt.weight_decay)

        self.bs=cfg.bs
        self.fbs=cfg.fbs

        self.totalNet = TotalNet()
        self.totalNet.to(self.device)
        
        self.classifier=self.totalNet.classifier
        self.discriminator=self.totalNet.discriminator
        self.discriminator_separate=self.totalNet.discriminator_separate

        optimizer_params = {'lr':cfg.opt.lr, 'weight_decay':self.cfg.opt.weight_decay, 'momentum':self.cfg.opt.momentum}

        


        #sinkhorn
        self.temperature=cfg.temperature
        self.sinkhorn_iterations=cfg.sinkhorn_iterations
        self.epsilon=cfg.epsilon
        self.class_distribution = torch.zeros([self.num_classes])
        optimizer_params = {'lr':cfg.opt.lr, 'weight_decay':self.cfg.opt.weight_decay, 'momentum':self.cfg.opt.momentum}
        if self.cfg.opt.kind == "SGD":
            optimizer_cls = torch.optim.SGD
            self.optimizer = optimizer_cls([{'params':self.model.get_1x_lr_params_NOscale(), 'lr':optimizer_params['lr']}, 
                                           {'params':self.model.get_10x_lr_params(), 'lr':optimizer_params['lr']*10},
                                           {'params':self.classifier.parameters(), 'lr':optimizer_params['lr']*5,'nesterov':True},
                                           {'params':self.discriminator.parameters(), 'lr':optimizer_params['lr']*5,'nesterov':True},
                                           {'params':self.discriminator_separate.parameters(), 'lr':optimizer_params['lr']*5,'nesterov':True}], **optimizer_params)
        elif self.cfg.opt.kind == "Adam":
            self.optimizer = torch.optim.Adam(params, betas=(
                0.9, 0.99), weight_decay=self.cfg.opt.weight_decay)
        else:
            raise NotImplementedError()
        self.lr_factor = 10


        # Memory
        # Source
        if self.cfg.data.source.dataset == 'synthia':
            source_train_dataset = SYNTHIA_Dataset(split='train', **self.cfg.data.source.kwargs)
            source_val_dataset = SYNTHIA_Dataset(split='val', **self.cfg.data.source.kwargs)
        elif self.cfg.data.source.dataset == 'gta5':
            source_train_dataset = GTA5_Dataset(split='train', **self.cfg.data.source.kwargs)
            source_val_dataset = GTA5_Dataset(split='val', **self.cfg.data.source.kwargs)
        
        elif self.cfg.data.target.dataset == 'cityscapes':
            source_train_dataset = City_Dataset(split='train', **self.cfg.data.target.kwargs)
            source_val_dataset = City_Dataset(split='val', **self.cfg.data.target.kwargs)
        else:
            raise NotImplementedError()
        self.source_dataloader = DataLoader(
            source_train_dataset, shuffle=True, drop_last=True, **self.cfg.data.loader.kwargs)
        self.source_val_dataloader = DataLoader(
            source_val_dataset, shuffle=False, drop_last=False, **self.cfg.data.loader.kwargs)

        # Target
        if self.cfg.data.target.dataset == 'cityscapes':
            target_train_dataset = City_Dataset(split='train', **self.cfg.data.target.kwargs)
            target_val_dataset = City_Dataset(split='val', **self.cfg.data.target.kwargs)
        
        elif self.cfg.data.source.dataset == 'gta5':
            target_train_dataset = GTA5_Dataset(split='train', **self.cfg.data.source.kwargs)
            target_val_dataset = GTA5_Dataset(split='val', **self.cfg.data.source.kwargs)
        else:
            raise NotImplementedError()
        self.target_dataloader = DataLoader(
            target_train_dataset, shuffle=True, drop_last=True, **self.cfg.data.loader.kwargs)
        self.target_val_dataloader = DataLoader(
            target_val_dataset, shuffle=False, drop_last=False, **self.cfg.data.loader.kwargs)

        # Perturbations
        if self.cfg.lam_aug > 0:
            self.aug = get_augmentation()

    def train(self):

        # Loop over epochs
        self.continue_training = True
        while self.continue_training:

            # Train for a single epoch
            self.train_one_epoch()

            # Use EMA params to evaluate performance
            self.ema.apply_shadow()
            self.ema.model.eval()
            self.ema.model.cuda()

            # Validate on source (if possible) and target
            if self.cfg.data.source_val_iterations > 0:
                self.validate(mode='source')
            PA, MPA, MIoU, FWIoU = self.validate(mode='source')
            self.evaluator.Print_Every_class_Eval(
                    out_16_13=(int(self.num_classes) in [16]))
            # Restore current (non-EMA) params for training
            self.ema.restore()

            # Log val results
            self.writer.add_scalar('PA', PA, self.epoch)
            self.writer.add_scalar('MPA', MPA, self.epoch)
            self.writer.add_scalar('MIoU', MIoU, self.epoch)
            self.writer.add_scalar('FWIoU', FWIoU, self.epoch)

            # Save checkpoint if new best model
            self.current_MIoU = MIoU
            is_best = MIoU > self.best_MIou
            if is_best:
                self.best_MIou = MIoU
                self.best_iter = self.iter
                self.logger.info("=> Saving a new best checkpoint...")
                self.logger.info("=> The best val MIoU is now {:.3f} from iter {}".format(
                    self.best_MIou, self.best_iter))
                self.save_checkpoint('best.pth')
            else:
                self.logger.info("=> The MIoU of val did not improve.")
                self.logger.info("=> The best val MIoU is still {:.3f} from iter {}".format(
                    self.best_MIou, self.best_iter))
            self.epoch += 1

        # Save final checkpoint
        self.logger.info("=> The best MIou was {:.3f} at iter {}".format(
            self.best_MIou, self.best_iter))
        self.logger.info(
            "=> Saving the final checkpoint to {}".format('final.pth'))
        self.save_checkpoint('final.pth')

    def train_one_epoch(self):

        # Load and reset
        self.model.train()
        self.evaluator.reset()
        
        # Helper
        def unpack(x):
            return (x[0], x[1]) if isinstance(x, tuple) else (x, None)

        # Training loop
        total = min(len(self.source_dataloader), len(self.target_dataloader))
        for batch_idx, (batch_s, batch_t) in enumerate(tqdm(
            zip(self.source_dataloader, self.target_dataloader),
            total=total, desc=f"Epoch {self.epoch + 1}",mininterval=60
        )):

            # Learning rate
            self.poly_lr_scheduler(optimizer=self.optimizer)
            self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]["lr"], self.iter)
            
            # Losses
            losses = {}

            ##########################
            # Source supervised loss #
            ##########################
            x, y, _ = batch_s

            if True:  # For VS Code collapsing
                with torch.autograd.set_detect_anomaly(True):
                    # Data
                    x = x.to(self.device)
                    y = y.squeeze(dim=1).to(device=self.device,
                                            dtype=torch.long, non_blocking=True)
                    
                    x_source=x.clone()
                    # Forward
                    s_pred = self.model(x)
                    source_out=s_pred['out']
                    source_feat=s_pred['feat']
                    input_size = x.size()[2:]
                    f_size=source_feat.size()[2:]        
                    feat_SL = source_feat.transpose(1, 2).transpose(2, 3).contiguous().view(self.bs,-1,256)
                    feat_SL=feat_SL.squeeze(0)
                    feat_SL_DS = torch.nn.functional.normalize(feat_SL)
                    
                    feature_source, fc2_s, predict_prob_source = self.classifier.forward(feat_SL_DS)
                    domain_prob_discriminator_source = self.discriminator.forward(feature_source)
                    domain_prob_discriminator_source_separate = self.discriminator_separate.forward(feature_source.detach())
                    source_share_weight = get_source_share_weight(domain_prob_discriminator_source_separate, fc2_s, domain_temperature=1.0, class_temperature=10.0)
                    source_share_weight = normalize_weight(source_share_weight)
                    
                    predict_prob_source2=predict_prob_source.unsqueeze(0)
                    out=predict_prob_source2.transpose(1,2).contiguous().view(self.bs,self.num_classes,f_size[0],f_size[1])
                    
                    out=F.interpolate(out,size=input_size,mode='bilinear', align_corners=True)
                    loss_source=self.loss(out,y)
                    #ce = torch.mean(ce, dim=0, keepdim=True)
                    loss_source.backward() 
                    losses['loss_source'] = loss_source.cpu().item()
                    del x
            
            ######################
            # Target Pseudolabel #
            ######################
            """
            x, y, _ = batch_t
            x = x.to(self.device)
            
            y = y.squeeze(dim=1).to(device=self.device,
                                    dtype=torch.long, non_blocking=True)
        
            target_pred = self.model(x.to(self.device))
                
            target_pred = self.model(x)
            target_out=target_pred['out']
            target_feat=target_pred['feat']

            feat_SL = target_pred['feat']
            out_SL = target_pred['out']
            target_f_size=feat_SL.size()[2:]
            #  ============ self-labeling loss ... ============  
            feat_SL = feat_SL.transpose(1, 2).transpose(2, 3).contiguous().view(self.bs, -1, 256)
            # randomly sampling pixel features
            rand_index = torch.randperm(feat_SL.shape[1])
            feat_SL = feat_SL[:,rand_index]
            feat_SL_DS = feat_SL[:, :self.fbs]
            
            feat_SL_DS = torch.nn.functional.normalize(feat_SL_DS, dim = 2, p=2)
            feat_SL_DS=feat_SL_DS.squeeze(0)

            feature_target, fc2_t, predict_prob_target = self.classifier.forward(feat_SL_DS)
            domain_prob_discriminator_target = self.discriminator.forward(feature_target)
            domain_prob_discriminator_target_separate = self.discriminator_separate.forward(feature_target.detach())
            target_share_weight = get_target_share_weight(domain_prob_discriminator_target_separate, fc2_t, domain_temperature=1.0, class_temperature=1.0)
            target_share_weight2=target_share_weight
            target_share_weight = normalize_weight(target_share_weight)
            
            # ==============================compute loss
            adv_loss = torch.zeros(1, 1).to(self.device)
            adv_loss_separate = torch.zeros(1, 1).to(self.device)

            tmp = source_share_weight * nn.BCELoss(reduction='none')(domain_prob_discriminator_source, torch.ones_like(domain_prob_discriminator_source))
            adv_loss += torch.mean(tmp, dim=0, keepdim=True)
            tmp = target_share_weight * nn.BCELoss(reduction='none')(domain_prob_discriminator_target, torch.zeros_like(domain_prob_discriminator_target))
            adv_loss += torch.mean(tmp, dim=0, keepdim=True)

            adv_loss_separate += nn.BCELoss()(domain_prob_discriminator_source_separate, torch.ones_like(domain_prob_discriminator_source_separate))
            adv_loss_separate += nn.BCELoss()(domain_prob_discriminator_target_separate, torch.zeros_like(domain_prob_discriminator_target_separate))

            # ============================== cross entropy loss
            #ce = nn.CrossEntropyLoss(reduction='none')(predict_prob_source, label)
            #ce = torch.mean(ce, dim=0, keepdim=True)
             
    
            #self.prototypes=weight_norm(self.prototypes) # very important for proto-classifier
            #self.target_prototypes=weight_norm(self.target_prototypes)
            losses['adv_loss'] = adv_loss.cpu().item()
            losses['adv_loss_separate'] = adv_loss_separate.cpu().item()
            loss = loss_source + adv_loss + adv_loss_separate
            loss.backward() 
            del adv_loss,adv_loss_separate,loss
            """
            if self.epoch>=0 and batch_idx > 0 and batch_idx % 1000 == 0:

                # Use EMA params to evaluate performance
                self.ema.apply_shadow()
                self.ema.model.eval()
                self.ema.model.cuda()

                # Validate on source (if possible) and target
                # if self.cfg.data.source_val_iterations > 0:
                #     self.validate(mode='source')
                PA, MPA, MIoU, FWIoU = self.validate(mode='source')
                self.evaluator.Print_Every_class_Eval(
                    out_16_13=(int(self.num_classes) in [16]))
                # Restore current (non-EMA) params for training
                self.ema.restore()

                # Log val results
                self.writer.add_scalar('PA', PA, self.epoch)
                self.writer.add_scalar('MPA', MPA, self.epoch)
                self.writer.add_scalar('MIoU', MIoU, self.epoch)
                self.writer.add_scalar('FWIoU', FWIoU, self.epoch)

                # Save checkpoint if new best model
                self.current_MIoU = MIoU
                is_best = MIoU > self.best_MIou
                if is_best:
                    self.best_MIou = MIoU
                    self.best_iter = self.iter
                    self.logger.info("=> Saving a new best checkpoint...")
                    self.logger.info("=> The best val MIoU is now {:.3f} from iter {}".format(
                        self.best_MIou, self.best_iter))
                    self.save_checkpoint('best.pth')
                else:
                    self.logger.info("=> The MIoU of val did not improve.")
                    self.logger.info("=> The best val MIoU is still {:.3f} from iter {}".format(
                        self.best_MIou, self.best_iter))
            
            self.optimizer.step()
       
            self.optimizer.zero_grad()

            # Update model EMA parameters each step
            self.ema.update_params()
            # weight_norm
            
            #self.memqueue.update_queue(feat_SL_DS2, self.id_target.cuda())
            #self.prototypes=weight_norm(self.prototypes) # very important for proto-classifier
            #self.target_prototypes=weight_norm(self.target_prototypes)
            # Calculate total loss
            
            total_loss = sum(losses.values())

            # Log main losses
            for name, loss in losses.items():
                self.writer.add_scalar(f'train/{name}', loss, self.iter)

            # Log
            if batch_idx % 100 == 0:
                log_string = f"[Epoch {self.epoch}]\t"
                log_string += '\t'.join([f'{n}: {l:.3f}' for n, l in losses.items()])
                self.logger.info(log_string)
            
            if self.iter>0 and self.iter%500==0:
                feat_SL = source_feat.transpose(1, 2).transpose(2, 3).contiguous().view(self.bs,-1,256)
                feat_SL=feat_SL.squeeze(0)
                feat_SL_DS = torch.nn.functional.normalize(feat_SL)
                
                __, before_softmax, predict_prob = self.classifier.forward(feat_SL_DS)
                domain_prob = self.discriminator_separate.forward(__)
                
                target_share_weight = get_target_share_weight(domain_prob, before_softmax, domain_temperature=1.0,
                                                                    class_temperature=1.0)
                target_share_weight2=target_share_weight.unsqueeze(0)
                target_share_weight2=target_share_weight2.transpose(1,2).contiguous().view(self.bs,1,f_size[0],f_size[1])
                target_share_weight2=F.interpolate(target_share_weight2,size=input_size,mode='bilinear', align_corners=True)
                    
               
                T = -0.5
                mask_1 = (target_share_weight2 > T)
                ignore_tensor = torch.ones(1).to(
                    self.device, dtype=torch.long)
                ignore_tensor2 = torch.zeros(1).to(
                    self.device, dtype=torch.long)
                label_1 = torch.where(mask_1, ignore_tensor, ignore_tensor2).squeeze(0)

                self.save_pixelmix(label_1,y)

            # Increment global iteration counter
            self.iter += 1

            # End training after finishing iterations
            if self.iter > self.cfg.opt.iterations:
                self.continue_training = False
                return

        # After each epoch, update model EMA buffers (i.e. batch norm stats)
        self.ema.update_buffer()
 
    @torch.no_grad()
    def sinkhorn(self, out):
        Q = torch.exp(out / self.epsilon).t() # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1]  # number of samples to assign
        K = Q.shape[0] # how many prototypes
        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
    #     dist.all_reduce(sum_Q)
        Q /= sum_Q
        for it in range(self.sinkhorn_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= sum_of_rows
            self.class_distribution=self.class_distribution.to(self.device)
            Q *= self.class_distribution.unsqueeze(1)

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B # the colomns must sum to 1 so that Q is an assignment
        # Q = torch.argmax(Q, 0)
        return Q.t()    
    
    @ torch.no_grad()
    def validate(self, mode='target'):
        """Validate on target"""
        self.logger.info('Validating')
        self.evaluator.reset()
        self.model.eval()

        # Select dataloader
        if mode == 'target':
            val_loader = self.target_val_dataloader
        elif mode == 'source':
            val_loader = self.source_val_dataloader
        else:
            raise NotImplementedError()

        # Loop
        for val_idx, (x, y, id) in enumerate(tqdm(val_loader, desc=f"Val Epoch {self.epoch + 1}")):
            if mode == 'source' and val_idx >= self.cfg.data.source_val_iterations:
                break

            # Forward
            x = x.to(self.device)
            y = y.to(device=self.device, dtype=torch.long)
            pred = self.model(x)
            #pred=pred['output']
            
            source_feat=pred['feat']
            source_out=pred['out']
            input_size = x.size()[2:]
            target_f_size=source_feat.size()[2:]
            feat_SL = source_feat.transpose(1, 2).transpose(2, 3).contiguous().view(self.bs,-1,256)
            feat_SL=feat_SL.squeeze(0)
            feat_SL_DS = torch.nn.functional.normalize(feat_SL)
            __, before_softmax, predict_prob = self.classifier.forward(feat_SL_DS)
            domain_prob = self.discriminator_separate.forward(__)
            target_share_weight = get_target_share_weight(domain_prob, before_softmax, domain_temperature=1.0,
                                                                  class_temperature=1.0)
            
            target_share_weight2=target_share_weight.unsqueeze(0)
            predict_prob2=predict_prob.unsqueeze(0)
            out=predict_prob2.transpose(1,2).contiguous().view(self.bs,self.num_classes,target_f_size[0],target_f_size[1])
            target_share_weight2=target_share_weight2.transpose(1,2).contiguous().view(self.bs,1,target_f_size[0],target_f_size[1])
            out=F.interpolate(out,size=input_size,mode='bilinear', align_corners=True)
            target_share_weight2=F.interpolate(target_share_weight2,size=input_size,mode='bilinear', align_corners=True)
            #source Supervision

            # Convert to numpy
            label = y.squeeze(dim=1).cpu().numpy()
            argpred = np.argmax(out.data.cpu().numpy(), axis=1)
            #maxpred_1, argpred_1 = torch.max(out.detach(), dim=1)
            #T = -0.5
            #mask_1 = (target_share_weight2 > T)
            #ignore_tensor = torch.ones(1).to(
            #    self.device, dtype=torch.long) * 13

            #label_1 = torch.where(mask_1, argpred_1, ignore_tensor).squeeze(0)

            #argpred=label_1.cpu().numpy()
            # Add to evaluator
            self.evaluator.add_batch(label, argpred)

        # Tensorboard images
        vis_imgs = 2
        images_inv = inv_preprocess(x.clone().cpu(), vis_imgs, numpy_transform=True)
        labels_colors = decode_labels(label, vis_imgs)
        preds_colors = decode_labels(argpred, vis_imgs)
        for index, (img, lab, predc) in enumerate(zip(images_inv, labels_colors, preds_colors)):
            self.writer.add_image(str(index) + '/images', img, self.epoch)
            self.writer.add_image(str(index) + '/labels', lab, self.epoch)
            self.writer.add_image(str(index) + '/preds', predc, self.epoch)

        # Calculate and log
        if self.cfg.data.source.kwargs.class_16:
            PA = self.evaluator.Pixel_Accuracy()
            MPA_16, MPA_13 = self.evaluator.Mean_Pixel_Accuracy()
            MIoU_16, MIoU_13 = self.evaluator.Mean_Intersection_over_Union()
            FWIoU_16, FWIoU_13 = self.evaluator.Frequency_Weighted_Intersection_over_Union()
            PC_16, PC_13 = self.evaluator.Mean_Precision()
            self.logger.info('Epoch:{:.3f}, PA:{:.3f}, MPA_16:{:.3f}, MIoU_16:{:.3f}, FWIoU_16:{:.3f}, PC_16:{:.3f}'.format(
                self.epoch, PA, MPA_16, MIoU_16, FWIoU_16, PC_16))
            self.logger.info('Epoch:{:.3f}, PA:{:.3f}, MPA_13:{:.3f}, MIoU_13:{:.3f}, FWIoU_13:{:.3f}, PC_13:{:.3f}'.format(
                self.epoch, PA, MPA_13, MIoU_13, FWIoU_13, PC_13))
            self.writer.add_scalar('PA', PA, self.epoch)
            self.writer.add_scalar('MPA_16', MPA_16, self.epoch)
            self.writer.add_scalar('MIoU_16', MIoU_16, self.epoch)
            self.writer.add_scalar('FWIoU_16', FWIoU_16, self.epoch)
            self.writer.add_scalar('MPA_13', MPA_13, self.epoch)
            self.writer.add_scalar('MIoU_13', MIoU_13, self.epoch)
            self.writer.add_scalar('FWIoU_13', FWIoU_13, self.epoch)
            PA, MPA, MIoU, FWIoU = PA, MPA_13, MIoU_13, FWIoU_13
        else:
            PA = self.evaluator.Pixel_Accuracy()
            MPA = self.evaluator.Mean_Pixel_Accuracy()
            MIoU = self.evaluator.Mean_Intersection_over_Union()
            FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
            PC = self.evaluator.Mean_Precision()
            self.logger.info('Epoch:{:.3f}, PA1:{:.3f}, MPA1:{:.3f}, MIoU1:{:.3f}, FWIoU1:{:.3f}, PC:{:.3f}'.format(
                self.epoch, PA, MPA, MIoU, FWIoU, PC))
            self.writer.add_scalar('PA', PA, self.epoch)
            self.writer.add_scalar('MPA', MPA, self.epoch)
            self.writer.add_scalar('MIoU', MIoU, self.epoch)
            self.writer.add_scalar('FWIoU', FWIoU, self.epoch)

        return PA, MPA, MIoU, FWIoU

    def save_checkpoint(self, filename='checkpoint.pth'):
        torch.save({
            'epoch': self.epoch + 1,
            'iter': self.iter,
            'state_dict': self.ema.model.state_dict(),
            'shadow': self.ema.shadow,
            'optimizer': self.optimizer.state_dict(),
            'best_MIou': self.best_MIou
        }, filename)

    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename, map_location='cpu')

        # Get model state dict
        if not self.cfg.train and 'shadow' in checkpoint:
            state_dict = checkpoint['shadow']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Remove DP/DDP if it exists
        state_dict = {k.replace('module.', ''): v for k,
                      v in state_dict.items()}

        # Load state dict
        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(state_dict)
        self.logger.info(f"Model loaded successfully from {filename}")

        # Load optimizer and epoch
        if self.cfg.train and self.cfg.model.resume_from_checkpoint:
            if 'optimizer' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.logger.info(f"Optimizer loaded successfully from {filename}")
            if 'epoch' in checkpoint and 'iter' in checkpoint:
                self.epoch = checkpoint['epoch']
                self.iter = checkpoint['iter'] if 'iter' in checkpoint else checkpoint['iteration']
                self.logger.info(f"Resuming training from epoch {self.epoch} iter {self.iter}")
        else:
            self.logger.info(f"Did not resume optimizer")

    def poly_lr_scheduler(self, optimizer, init_lr=None, iter=None, max_iter=None, power=None):
        init_lr = self.cfg.opt.lr if init_lr is None else init_lr
        iter = self.iter if iter is None else iter
        max_iter = self.cfg.opt.iterations if max_iter is None else max_iter
        power = self.cfg.opt.poly_power if power is None else power
        new_lr = init_lr * (1 - float(iter) / max_iter) ** power
        optimizer.param_groups[0]["lr"] = new_lr
        if len(optimizer.param_groups) == 2:
            optimizer.param_groups[1]["lr"] = 10 * new_lr

    @ torch.no_grad()
    def save_pixelmix(self,y,x_target):
        
        #gt
       
        pseudo_ = y
        pseudo_ = pseudo_.detach().cpu().numpy()
        pseudo_ = np.asarray(pseudo_, dtype=np.uint8).squeeze(0)

        #pseudo_.save('/data/seunan/CUDA_image/fornewpixelmix/s_gt_%s.png' % (self.iter))
        #output_col = colorize_mask(pseudo_)
        output_col = colorize_mask(pseudo_)
        output_col.save('/data/dragoon0905/CUDA_image/foggy/s_gt_UANSO%s_color.png' % (self.iter))

        
        
        pseudo_ = x_target
       
        pseudo_ = pseudo_.detach().cpu().numpy()
        pseudo_ = np.asarray(pseudo_, dtype=np.uint8).squeeze(0)
        #pseudo_.save('/data/seunan/CUDA_image/fornewpixelmix/s_gt_%s.png' % (self.iter))
        output_col = colorize_mask(pseudo_)
        output_col.save('/data/dragoon0905/CUDA_image/foggy/t_gt_UANSO%s_color.png' % (self.iter))
        """
        #source image
        pseudo_ = x_source
        
        pseudo_ = pseudo_.detach().cpu().numpy()
        pseudo_ = np.asarray(pseudo_, dtype=np.uint8).squeeze(0).transpose((1,2,0))
        #pseudo_.save('/data/seunan/CUDA_image/fornewpixelmix/s_image_%s.png' % (self.iter))
        output_col = Image.fromarray(pseudo_).convert("RGB")
        output_col.save('/data/seunan/CUDA_image/foggy/s_image_%s_color.png' % (self.iter))

        #target image
        pseudo_ = x_target
       
        pseudo_ = pseudo_.detach().cpu().numpy()
        pseudo_ = np.asarray(pseudo_, dtype=np.uint8).squeeze(0).transpose((1,2,0))
        #pseudo_.save('/data/seunan/CUDA_image/fornewpixelmix/t_image_%s.png' % (self.iter))
        output_col = Image.fromarray(pseudo_).convert("RGB")
        output_col.save('/data/seunan/CUDA_image/foggy/t_image_%s_color.png' % (self.iter))

        #target pred
        pseudo_ = pred_t
       
        pseudo_ = pseudo_.detach().cpu().numpy()
        pseudo_ = np.asarray(pseudo_, dtype=np.uint8).squeeze(0)
        #pseudo_.save('/data/seunan/CUDA_image/fornewpixelmix/t_pred_%s.png' % (self.iter))
        output_col = colorize_mask(pseudo_)
        output_col.save('/data/seunan/CUDA_image/foggy/t_pred_%s_color.png' % (self.iter))
        #mix image

        pseudo_ = x_new
        
        pseudo_ = pseudo_.detach().cpu().numpy()
        pseudo_ = np.asarray(pseudo_, dtype=np.uint8).squeeze(0).transpose((1,2,0))
        output_col = Image.fromarray(pseudo_).convert("RGB")
        output_col.save('/data/seunan/CUDA_image/foggy/t_%s.png' % (self.iter))
        """

@hydra.main(config_path='configs', config_name='gta5')
def main(cfg: DictConfig):

    # Seeds
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.random.manual_seed(cfg.seed)

    # Logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Monitoring
    if cfg.wandb:
        import wandb
        wandb.init(project='pixmatch', name=cfg.name, config=cfg, sync_tensorboard=True)
    writer = SummaryWriter(cfg.name)

    # Trainer
    trainer = Trainer(cfg=cfg, logger=logger, writer=writer)

    # Load pretrained checkpoint
    if cfg.model.checkpoint:
        assert Path(cfg.model.checkpoint).is_file(), f'not a file: {cfg.model.checkpoint}'
        trainer.load_checkpoint(cfg.model.checkpoint)

    # Print configuration
    logger.info('\n' + OmegaConf.to_yaml(cfg))

    # Train
    if cfg.train:
        trainer.train()

    # Evaluate
    else:
        trainer.validate()
        trainer.evaluator.Print_Every_class_Eval(
            out_16_13=(int(cfg.data.num_classes) in [16, 13]))


if __name__ == '__main__':
    main()
