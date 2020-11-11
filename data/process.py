# -*- coding: utf-8 -*-
import json

word2vec={}
WikiData={}
num2embed={}

def load_word2vec(file_name):
    w2v=json.load(open(file_name,'r'))
    for _ in w2v:
        word2vec[_['word'].lower()]=_['vec']

def read_info(file_name):
    WikiDatafile=json.load(open(file_name,'r'))
    for relation in WikiDatafile:
        relation['name']=relation['name'].lower()
        relation['desc']=relation['desc'].lower()
        for i in range(len(relation['alias'])):
            relation['alias'][i]=relation['alias'][i].lower()
        WikiData[relation['id']]=relation

def read_classnames(file_name):
    class_names=json.load(open(file_name,'r'))
    num=0
    for val in class_names.values(): num+=len(val)
    print(file_name,len(class_names.keys()),num)
    return class_names.keys()

def plus(a,b):
    a+=b
    return a

def turn2embed(task_name,class_names):
    print(task_name,'has',str(len(class_names)),'classes')
    embedding_dim=len(word2vec['the'])
    for name in class_names:
        relation=WikiData[name]

        #name_embed,name_num=[0.0]*embedding_dim,0.0
        name_embed,name_num=[],0.0
        #e,word_num=[0.0]*embedding_dim,0.0
        e,word_num=[],0.0
        for _ in relation['name'].split():
            if _ not in word2vec:
                print([name,relation['name'],relation['alias'],_])
                raise Exception("[ERROR] class name doesn't exist")
            e=plus(e,word2vec[_])
            word_num+=1
        for i in range(len(e)): e[i]/=word_num
        name_embed=e
        name_num+=1
        for alias in relation['alias']:
            #e,word_num,can_use=[0.0]*embedding_dim,0.0,True
            e,word_num,can_use=[],0.0,True
            for _ in alias.replace('[',' ').replace(']',' ').replace('(',' ').replace(')',' ').replace("'",' ').replace(".",' ').replace("-",' ').replace(":",' ').replace(";",' ').replace('"',' ').replace('/',' ').split():
                if _ not in word2vec: 
                    can_use=False
                else:
                    word_num+=1
                    e=plus(e,word2vec[_])
            if can_use:
                for i in range(len(e)): e[i]/=word_num
                name_embed=plus(name_embed,e)
                name_num+=1
        for i in range(len(name_embed)): name_embed[i]/=name_num 
        #num2embed[name]=name_embed

        desc_embed,desc_num=[0.0]*embedding_dim,0.0
        for desc in relation['desc'].split(','):
            e,word_num,can_use,bad_words=[0.0]*embedding_dim,0.0,True,[]
            for _ in desc.replace('[',' ').replace(']',' ').replace('(',' ').replace(')',' ').replace("'",' ').replace(".",' ').replace("-",' ').replace(":",' ').replace(";",' ').replace('"',' ').replace('/',' ').split():
                if _ not in word2vec:
                    if not((_[0] in 'pq') and (_[1] in '0123456789')):
                        can_use=False
                        bad_words.append(_)
                else:
                    word_num+=1
                    e=plus(e,word2vec[_])
            if word_num!=0:
                if can_use:
                    for i in range(embedding_dim): e[i]/=word_num
                    desc_embed=plus(desc_embed,e)
                    desc_num+=1
                else:
                    print([name,relation['desc'],desc,bad_words],'\n')
        for i in range(embedding_dim): desc_embed[i]/=desc_num 

        num2embed[name]=name_embed#+desc_embed

if __name__=='__main__':
    load_word2vec('./glove.6B.50d.json')
    read_info('./P_info.json')
    turn2embed('train',read_classnames('./train.json'))
    turn2embed('val',read_classnames('./val.json'))
    with open('meta-info.json','w') as fout:
        fout.write(json.dumps(num2embed))
        fout.write('\n')





