# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from my_transformers.transformers import BertConfig,BertModel,BertTokenizer

class MIML(nn.Module):
    def __init__(self,B,N,K,max_length,data_dir,MAML,MI,MF,VAT):
        nn.Module.__init__(self)

        self.Batch=B
        self.n_way=N
        self.k_shot=K
        self.max_length=max_length
        self.data_dir=data_dir
        self.MAML=MAML
        self.MI=MI
        self.MF=MF
        self.VAT=VAT
        self.hidden_size=768*2

        self.cost=nn.NLLLoss()
        self.coder=BERT(N,max_length,data_dir) # <[N*K,length], >[N*K,hidden_size]
        self.softmax=nn.Softmax(-1)

        if not MAML:
            self.mlp=MLP(MF,N,data_dir,self.hidden_size) 
            fc_params=nn.Linear(self.hidden_size,self.n_way,bias=None)
            self.fc=[fc_params.weight.detach().cuda()]*B
            for i in range(B): self.fc[i].requires_grad=True
        else: 
            self.fc=nn.Linear(self.hidden_size,self.n_way,bias=None)

    def loss(self,logits,label):
        #logits = (N*K, N) label = (1, N*K)
        #CrossEntropyLoss((N*K, N), (N*K))
        return self.cost(logits.log(),label.view(-1))
    def accuracy(self,logits,label):
        label=label.view(-1)
        logits=logits.data.cpu().numpy()
        label=label.data.cpu().numpy()
        logits=np.argmax(logits,axis=1)
        return np.sum(logits==label)/float(label.size)
    def reset_fc(self):
        self.fc=[torch.Tensor(self.n_way,self.hidden_size)]*self.Batch
    def cloned_fc_dict(self):
        return {key:val.clone() for key,val in self.fc.state_dict().items()}
    def maml(self,inputs,params=None): #[1,bert_size]
        if params==None: out=self.fc(inputs) #[1,N]
        else: out=F.linear(inputs,params['weight']) #[1,N]
        return F.softmax(out,dim=-1)#[1,N]
    def forward(self,inputs,params): #[1,bert_size]
        out=F.linear(inputs,params,bias=None) #[1,N]
        return F.softmax(out,dim=-1)#[1,N]
    def get_info(self,class_name):
        return self.mlp.get_embedding(class_name)
    def prework(self,meta_information,query): #attention, [N,bert_size]
        N=self.n_way
        K=self.k_shot

        if self.MF:
            query=F.normalize(query,dim=-1)
            params,rel=self.mlp(meta_information)

            idx=torch.zeros(N*K).long().cuda()
            for i in range(N): idx[i*K:(i+1)*K]=i #[0,0,...0,1,1...1,...N-1...N-1]
            att=(query*rel[idx]).sum(-1) #([N*K,bert_size]·[N*K,bert_size]).sum(-1)=[N*K]
            for i in range(N): att[i*K:(i+1)*K]=self.softmax(att[i*K:(i+1)*K]) #[N*K]
            return params,att
        else: 
            return self.mlp(meta_information)

class MLP(nn.Module):
    def __init__(self,MF,N,data_dir,bert_size):
        super(MLP,self).__init__()
        self.n_way=N
        self.MF=MF
        self.embedding_dim=50

        word_info=data_dir['meta-info']
        if word_info is None or not os.path.isfile(word_info):
            raise Exception("[ERROR] word information file doesn't exist")
        self.num2embed=json.load(open(word_info,'r'))
        for key in self.num2embed.keys():
            self.num2embed[key]=torch.Tensor(self.num2embed[key])
        
        self.dropout=nn.Dropout(0.5)
        self.trans=nn.Linear(self.embedding_dim,bert_size)
        self.att_map=nn.Linear(self.embedding_dim,bert_size)
    def get_embedding(self,class_name):
        res=torch.zeros(self.n_way,self.embedding_dim)
        for i in range(self.n_way):
            num=class_name[i][0]
            embeds=torch.from_numpy(np.stack(self.num2embed[num],0)).reshape(-1,self.embedding_dim).float()
            for j in range(embeds.size(0)): res[i]+=embeds[j]
        return res.cuda() # >[N], <[N,embedding_dim]
    def forward(self,inputs): #[N,embedding_dim]
        params=self.trans(inputs) #[N,bert_size]
        params=F.normalize(params,dim=-1) #[N,bert_size]
        if self.MF:
            att=self.att_map(inputs) #attention: [N,bert_size]
            att=F.normalize(att,dim=-1)
            return params,att
        else: 
            return params
               
class BERT(nn.Module):
    def __init__(self,N,max_length,data_dir,blank_padding=True):
        super(BERT,self).__init__()
        self.cuda=torch.cuda.is_available()
        self.n_way=N
        self.max_length=max_length
        self.blank_padding=blank_padding
        self.pretrained_path='bert-base-uncased'
        if os.path.exists(self.pretrained_path):
            config=BertConfig.from_pretrained(os.path.join(self.pretrained_path,'bert-base-uncased-config.json'))
            self.bert=BertModel.from_pretrained(os.path.join(self.pretrained_path,'bert-base-uncased-pytorch_model.bin'),config=config)
            self.tokenizer=BertTokenizer.from_pretrained(os.path.join(self.pretrained_path,'bert-base-uncased-vocab.txt'))
        else:
            self.bert=BertModel.from_pretrained(self.pretrained_path)
            self.tokenizer=BertTokenizer.from_pretrained(self.pretrained_path)
        self.dropout=nn.Dropout(0.5)

    def forward(self,inputs):
        tokens,att_masks,head_poses,outputs=[],[],[],[]
        for _ in inputs: 
            token,att_mask,head_pos=self.tokenize(_)
            tokens.append(token)
            att_masks.append(att_mask)
            head_poses.append(head_pos)
        token=torch.cat([t for t in tokens],0)#[N*K,max_length]
        att_mask=torch.cat([a for a in att_masks],0)#[N*K,max_length]
        #sequence_output,pooled_output=self.bert(token,attention_mask=att_mask)
        sequence_output=self.bert(token,attention_mask=att_mask) #[N*K,max_length,bert_size]
        for i in range(token.size(0)):
            outputs.append(self.entity_start_state(head_poses[i],sequence_output[i]))
        outputs=torch.cat([o for o in outputs],0)
        outputs=self.dropout(outputs)#[N*K,bert_size*2]
        return outputs

    def entity_start_state(self,head_pos,sequence_output):
        if head_pos[0]==-1 or head_pos[0]>=self.max_length:
            head_pos[0]=0
            #raise Exception("[ERROR] no head entity")
        if head_pos[1]==-1 or head_pos[1]>=self.max_length:
            head_pos[1]=0
            #raise Exception("[ERROR] no tail entity")
        res=torch.cat([sequence_output[head_pos[0]],sequence_output[head_pos[1]]],0)
        return res.unsqueeze(0)

    def tokenize(self,inputs):
        tokens = inputs['tokens']
        pos_head = inputs['h'][2][0]
        pos_tail = inputs['t'][2][0]

        re_tokens,cur_pos=['[CLS]',],0
        head_pos=[-1,-1]
        for token in tokens:
            token=token[0].lower()
            if cur_pos==pos_head[0]: 
                head_pos[0]=len(re_tokens)
                re_tokens.append('[unused0]')
            if cur_pos==pos_tail[0]: 
                head_pos[1]=len(re_tokens)
                re_tokens.append('[unused1]')
            re_tokens+=self.tokenizer.tokenize(token)
            if cur_pos==pos_head[-1]-1: re_tokens.append('[unused2]')
            if cur_pos==pos_tail[-1]-1: re_tokens.append('[unused3]')
            cur_pos+=1
        re_tokens.append('[SEP]')

        indexed_tokens=self.tokenizer.convert_tokens_to_ids(re_tokens)
        avai_len=len(indexed_tokens)

        if self.blank_padding:
            while len(indexed_tokens)<self.max_length: indexed_tokens.append(0)
            indexed_tokens=indexed_tokens[:self.max_length]
        
        indexed_tokens=torch.tensor(indexed_tokens).long().unsqueeze(0)
        att_mask=torch.zeros(indexed_tokens.size()).long()
        att_mask[0,:avai_len]=1
        if self.cuda: indexed_tokens,att_mask=indexed_tokens.cuda(),att_mask.cuda()
        return indexed_tokens,att_mask,head_pos #both [1,max_length]
