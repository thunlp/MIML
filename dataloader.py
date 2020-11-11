# -*- coding: utf-8 -*-
import os
import json
import random
import numpy as np
import torch
from torch.autograd import Variable
import torch.utils.data as data

class FewRel(data.Dataset):
    def __init__(self,file_name,N,K,Q,noise_rate):
        super(FewRel,self).__init__()
        if not os.path.isfile(file_name):
            raise Exception("[ERROR] Data file doesn't exist")
        self.json_data=json.load(open(file_name,'r'))
        self.classes=list(self.json_data.keys())
        self.N,self.K,self.Q=N,K,Q
        self.noise_rate=noise_rate
    def __len__(self):
        return 1000000000
    def __getitem__(self,index):
        N,K,Q=self.N,self.K,self.Q
        class_name=random.sample(self.classes,N)
        support,support_label,query,query_label=[],[],[],[]
        for i,name in enumerate(class_name):
            rel=self.json_data[name]
            samples=random.sample(rel,K+Q)
            for j in range(K):
                support.append([samples[j],i])
            for j in range(K,K+Q):
                query.append([samples[j],i])
        #support=random.sample(support,N*K)
        query=random.sample(query,N*Q)
        for i in range(N*K):
            support_label.append(support[i][1])
            support[i]=support[i][0]
        if self.noise_rate>0:
            other_classes=[]
            for _ in self.classes:
                if _ not in class_name:
                    other_classes.append(_)
            for i in range(N*K):
                if(random.randint(1,10)<=self.noise_rate):
                    noise_name=random.sample(other_classes,1)
                    rel=self.json_data[noise_name[0]]
                    support[i]=random.sample(rel,1)[0]
        for i in range(N*Q):
            query_label.append(query[i][1])
            query[i]=query[i][0]
        support_label=Variable(torch.from_numpy(np.stack(support_label,0).astype(np.int64)).long())
        query_label=Variable(torch.from_numpy(np.stack(query_label,0).astype(np.int64)).long())
        #if torch.cuda.is_available():support_label,query_label=support_label.cuda(),query_label.cuda()
        return class_name,support,support_label,query,query_label
def get_dataloader(file_name,N,K,Q,noise_rate):
    data_loader=data.DataLoader(
            dataset=FewRel(file_name,N,K,Q,noise_rate),
            batch_size=1,
            shuffle=False,
            num_workers=0
        )
    return iter(data_loader)