# -*- coding: utf-8 -*-

import copy
import torch
import math
from torch import autograd, optim, nn
from torch.nn import functional as F
from my_transformers.transformers import AdamW
from dataloader import get_dataloader
from torch.autograd import Variable
from collections import OrderedDict
import torch.distributions as dist

def zero_grad(params):
    for p in params:
        if p.grad is not None:
            p.grad.zero_()
     
def train_maml(support,support_label,query,query_label,net,steps,task_lr):
    '''first step'''
    loss=net.loss(net.maml(support),support_label)
    zero_grad(net.parameters())
    grads=autograd.grad(loss,net.fc.parameters())
    fast_weights,orderd_params=net.cloned_fc_dict(),OrderedDict()
    for (key,val),grad in zip(net.fc.named_parameters(),grads): 
        fast_weights[key]=orderd_params[key]=val-task_lr*grad
    '''steps remaining'''
    for k in range(steps-1):
        loss=net.loss(net.maml(support,fast_weights),support_label)
        zero_grad(orderd_params.values())
        grads=torch.autograd.grad(loss,orderd_params.values())
        for (key,val),grad in zip(orderd_params.items(),grads): 
            fast_weights[key]=orderd_params[key]=val-task_lr*grad
    '''return'''
    logits_q=net.maml(query,fast_weights)
    return net.loss(logits_q,query_label),net.accuracy(logits_q,query_label)

class calc_lr(object):
    """docstring for get_lr"""
    def __init__(self):
        self.lr=1e-3
        self.T=1000
    def Triangle(self,it):
        if it<=500: return self.lr*(it/500.0)
        elif it<=self.T+500: return self.lr*(1.0-(it-500.0)/self.T)
        else: return 0
    def Line(self,it):
        if it<=self.T: return self.lr*(1.0-it/self.T)
        else: return 0
    def Parabola(self,it):
        if it<=self.T: return self.lr*(1-math.exp(it-self.T))
        else: return 0
    def reParabola(self,it):
        if it<=self.T: return self.lr*math.exp(-it)
        else: return 0
    def Inverse(self,it):
        if it<=self.T: return self.lr/it
        else: return 0

get_lr=calc_lr()

def get_noise(params):
    random_noise=torch.FloatTensor(params.shape).uniform_(0,1).cuda()
    random_noise.requires_grad=True
    return F.normalize(random_noise,p=2,dim=-1)

def train_ones(fc,support,support_label,query,net,steps,task_lr,N,K,att=None):
    if K!=0:
        for k in range(steps):
            logits=net(support,fc)
            loss=Variable(torch.FloatTensor([0.0]*(N*K)).cuda())
            for i in range(N*K):
                loss[i]+=net.loss(logits[i].unsqueeze(0),support_label[0][i])
            if att is None: loss=loss.sum(-1)/N/K
            else: loss=((att*loss).sum(-1))/N

            zero_grad(fc)
            grads=autograd.grad(loss,fc)
            fc=fc-task_lr*grads[0]
    logits_q=net(query,fc)
    return logits_q

def train_one_batch(idx,class_name,support0,support_label,query0,query_label,net,steps,task_lr,it,zero_shot=False):
    
    N=net.n_way
    if zero_shot: K=0
    else: K=net.k_shot
    support,query=net.coder(support0),net.coder(query0) #[N*K,bert_size]
    if not net.MAML: 
        meta_information=net.get_info(class_name) #[N,embedding_size]

        vat_lr=get_lr.Line(it+1.0)
        if net.VAT and vat_lr>1e-10:
            bernoulli=dist.Bernoulli
            random_noise=vat_lr*get_noise(meta_information)
            if net.MF: 
                fc1,att1=net.prework(meta_information,support)
                fc2,att2=net.prework(meta_information+random_noise,support)
                logits1=train_ones(fc1,support,support_label,query,net,steps,task_lr,N,K,att1)
                logits2=train_ones(fc2,support,support_label,query,net,steps,task_lr,N,K,att2)
            else: 
                fc1=net.prework(meta_information,support)
                fc2=net.prework(meta_information+random_noise,support)
                logits1=train_ones(fc1,support,support_label,query,net,steps,task_lr,N,K)
                logits2=train_ones(fc2,support,support_label,query,net,steps,task_lr,N,K)

            zero_grad(random_noise)
            KL=dist.kl_divergence(bernoulli(probs=logits1),bernoulli(probs=logits2))
            grads=autograd.grad(KL.sum(),random_noise,retain_graph=True)
            vat_noise=vat_lr*F.normalize(grads[0].detach(),p=2,dim=-1)
            if net.MF:
                if net.MI:
                    net.fc[idx],att=net.prework(meta_information+vat_noise,support)
                else:
                    _,att=net.prework(meta_information+vat_noise,support)
                logits_q=train_ones(net.fc[idx],support,support_label,query,net,steps,task_lr,N,K,att)
            else:
                if net.MI:
                    net.fc[idx]=net.prework(meta_information+vat_noise,support)
                else:
                    _=net.prework(meta_information+vat_noise,support)
                logits_q=train_ones(net.fc[idx],support,support_label,query,net,steps,task_lr,N,K)
            vat_loss=torch.mean(dist.kl_divergence(bernoulli(probs=logits1),bernoulli(probs=logits_q)))
            return net.loss(logits_q,query_label)+vat_loss,net.accuracy(logits_q,query_label)
        else:
            if net.MF:
                if net.MI:
                    net.fc[idx],att=net.prework(meta_information,support)
                else:
                    _,att=net.prework(meta_information,support)
                logits_q=train_ones(net.fc[idx],support,support_label,query,net,steps,task_lr,N,K,att)
            else:
                if net.MI:
                    net.fc[idx]=net.prework(meta_information,support)
                else:
                    _=net.prework(meta_information,support)
                logits_q=train_ones(net.fc[idx],support,support_label,query,net,steps,task_lr,N,K)
            return net.loss(logits_q,query_label),net.accuracy(logits_q,query_label)
    else:
        return train_maml(support,support_label,query,query_label,net,steps,task_lr)


def test_model(cuda,data_loader,model,val_iter,test_update_step,task_lr,zero_shot=False):
    accs=0.0
    model.eval()
    if model.MI: model.reset_fc()
    for it in range(val_iter):
        net=copy.deepcopy(model)
        class_name,support,support_label,query,query_label=next(data_loader)
        if cuda: support_label,query_label=support_label.cuda(),query_label.cuda()
        loss,right=train_one_batch(0,class_name,support,support_label,query,query_label,net,test_update_step,task_lr,100000000,zero_shot)
        accs+=right
        if (it+1)%500==0: print('step: {0:4} | accuracy: {1:3.2f}%'.format(it+1,100*accs/(it+1)))
    return accs/val_iter

def train_model(model,B,N,K,Q,data_dir,
            meta_lr=1,
            task_lr=7e-2,
            train_iter=10000,
            val_iter=1000,
            val_step=100,
            test_iter=10000,
            test_step=1000,
            test_update_step=100):
    n_way_k_shot=str(N)+'-way-'+str(K)+'-shot'
    if model.MAML: n_way_k_shot='maml-'+n_way_k_shot
    else: n_way_k_shot='miml-'+n_way_k_shot
    print('Start training '+n_way_k_shot)
    cuda=torch.cuda.is_available()
    if cuda: model=model.cuda()

    data_loader={}
    data_loader['train']=get_dataloader(data_dir['train'],N,K,Q,data_dir['noise_rate'])
    data_loader['val']=get_dataloader(data_dir['val'],N,K,Q,data_dir['noise_rate'])
    data_loader['test']=get_dataloader(data_dir['test'],N,K,Q,data_dir['noise_rate'])

    optim_params=[{'params':model.coder.parameters(),'lr':5e-5},]
    if model.MI: 
        optim_params.append({'params':model.mlp.trans.parameters(),'lr':meta_lr})
    if model.MF:
        optim_params.append({'params':model.mlp.att_map.parameters(),'lr':meta_lr})
    meta_optimizer=AdamW(optim_params,lr=meta_lr)

    best_acc,best_step,best_test_acc,best_test_step,best_changed=0.0,0,0.0,0,False
    iter_loss,iter_right,iter_sample=0.0,0.0,0.0

    for it in range(train_iter):
        meta_loss,meta_right=0.0,0.0
        for batch in range(B):
            model.train()
            class_name,support,support_label,query,query_label=next(data_loader['train'])
            #[N], [N*K,length], [1,N*K], [N*Q,length], [1,N*Q]
            if cuda: support_label,query_label=support_label.cuda(),query_label.cuda()

            loss,right=train_one_batch(batch,class_name,support,support_label,query,query_label,model,test_update_step,task_lr,it)
            meta_loss+=loss
            meta_right+=right
        meta_loss/=B
        meta_right/=B
        meta_optimizer.zero_grad()
        meta_loss.backward()
        meta_optimizer.step()
    
        iter_loss+=meta_loss
        iter_right+=meta_right
        iter_sample+=1 

        if it%val_step==0: iter_loss,iter_right,iter_sample=0.0,0.0,0.0
        if ((it+1)%100==0) or ((it+1)%val_step==0): print('step: {0:4} | loss: {1:2.6f}, accuracy: {2:3.2f}%'.format(it + 1, iter_loss / iter_sample, 100 * iter_right / iter_sample))
        
        if (it+1)%val_step==0:
            acc=test_model(cuda,data_loader['val'],model,val_iter,test_update_step,task_lr)
            print('[EVAL] | accuracy: {0:2.2f}%'.format(acc*100))
            if acc>best_acc:
                print('Best checkpoint!')
                if model.MI: model.reset_fc()
                best_model=copy.deepcopy(model)
                best_acc,best_step,best_changed=acc,(it+1),True

        if (it+1)%test_step==0 and best_changed:
            best_changed=False
            test_acc=test_model(cuda,data_loader['test'],model,test_iter,test_update_step,task_lr)
            print('[TEST] | accuracy: {0:2.2f}%'.format(test_acc*100))
            if test_acc<1.5/N:
                print("That's so bad!")
                break
            if test_acc>best_test_acc:
                #torch.save(best_model.state_dict(),n_way_k_shot+'.ckpt')
                best_test_acc,best_test_step=test_acc,best_step
            best_acc=0.0

    print("\n####################\n")
    print('Finish training model! Best acc: '+str(best_test_acc)+' at step '+str(best_test_step))
    