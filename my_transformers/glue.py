import os
import numpy as np
import torch
from torch.utils.data import SequentialSampler,DataLoader,TensorDataset
CUDA=torch.cuda.is_available()

model_type='bert'
task_name='cola'
model_name_or_path='bert-base-uncased'
output_dir='/output/Model_'+model_type+'_on_task_'+task_name+'/'
data_dir='/data/cola/'
batch_size=8
max_seq_length=128

from transformers import glue_processors as processors
from transformers import glue_output_modes as output_modes
from transformers import glue_convert_examples_to_features as convert_examples_to_features
#for key in processors.keys(): print(key)
processor=processors[task_name]()
label_list=processor.get_labels()
num_labels=len(label_list)

from transformers import BertConfig,BertForSequenceClassification,BertTokenizer
MODEL_CLASSES={'bert':(BertConfig,BertForSequenceClassification,BertTokenizer),}
config_class,model_class,tokenizer_class=MODEL_CLASSES[model_type]
config=config_class.from_pretrained(model_name_or_path,num_labels=num_labels,finetuning_task=task_name)
tokenizer=tokenizer_class.from_pretrained(model_name_or_path,do_lower_case=True)
model=model_class.from_pretrained(model_name_or_path,from_tf=False,config=config)
if CUDA: mode=model.cuda()
exit()
'''evaluate(model,tokenizer)'''
if not os.path.exists(output_dir): os.makedirs(output_dir)
def load_and_cache_examples(data_dir,cach_name,task_name,tokenizer,max_seq_length,evaluate):
	processor=processors[task_name]()
	label_list=processor.get_labels()
	output_mode=output_modes[task_name]
	cached_features_file=os.path.join(data_dir,cach_name)
	if evaluate: examples=processor.get_dev_examples(data_dir)
	else: examples=processor.get_train_examples(data_dir)
	features=convert_examples_to_features(
			examples,
			tokenizer,
			label_list=label_list,
			max_length=max_seq_length,
			output_mode=output_mode,
			pad_on_left=False,
			pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
			pad_token_segment_id=0,
		)
	all_input_ids=torch.tensor([f.input_ids for f in features],dtype=torch.long)
	all_attention_mask=torch.tensor([f.attention_mask for f in features],dtype=torch.long)
	all_token_type_ids=torch.tensor([f.token_type_ids for f in features],dtype=torch.long)
	all_labels=torch.tensor([f.laabel for f in features],dtype=(torch.long if output_mode=='classification' else torch.float))
	dataset=TensorDataset(all_input_ids,all_attention_mask,all_token_type_ids,all_labels)
	return dataset

cach_name='cached_{}_{}_{}_{}'.format('dev',list(filter(None,model_name_or_path.split('/'))).pop(),str(max_seq_length),task_name)
dataset=load_and_cache_examples(data_dir,cach_name,task_name,tokenizer,max_seq_length,evaluate=True)
sampler=SequentialSampler(dataset)
dataloader=DataLoader(dataset,sampler=sampler,batch_size=batch_size)
	
loss,cnt=0.0,0
logits=[]
for it in dataloader:
	inputs={
		'input_ids':it[0],
		'attention_mask':it[1],
		'token_type_ids':it[2],
		'labels':it[3],
	}
	outputs=model(**inputs)
	loss_,logits_=outputs[:2]
	loss+=loss_.mean().item()
	logits.append(logits_)
	cnt+=1

loss=loss/float(cnt)

















