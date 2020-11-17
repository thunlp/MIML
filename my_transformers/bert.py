import os
import numpy as np
import torch
from torch.utils.data import SequentialSampler,DataLoader,TensorDataset
from transformers import glue_processors as processors
from transformers import glue_output_modes as output_modes
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import BertConfig,BertForSequenceClassification,BertTokenizer

class BertModel(object):
	def __init__(self,model_type='bert',task_name='fewrel',model_name_or_path='bert-base-uncased'):
		self.model_type=model_type
		self.task_name=task_name 
		self.model_name_or_path=model_name_or_path
		#self.output_dir='/output/Model_'+model_type+'_on_task_'+task_name+'/'
		#self.data_dir='/data/cola/' #'/data/fewrel/'
		self.batch_size=5
		self.max_seq_length=128
		self.CUDA=torch.cuda.is_available()

		self.processor=processors[task_name]()
		self.label_list=self.processor.get_labels()
		self.output_mode=output_modes[task_name]

		config_class,model_class,tokenizer_class=BertConfig,BertForSequenceClassification,BertTokenizer
		self.config=config_class.from_pretrained(model_name_or_path,num_labels=len(self.label_list),finetuning_task=task_name)
		self.tokenizer=tokenizer_class.from_pretrained(model_name_or_path,do_lower_case=True)
		self.model=model_class.from_pretrained(model_name_or_path,from_tf=False,config=self.config)
		if self.CUDA: self.mode=self.model.cuda()

	def get_dataset(self,inputs):
		examples=self.processor.get_dev_examples(inputs)
		features=convert_examples_to_features(
			examples,
			self.tokenizer,
			label_list=self.label_list,
			max_length=self.max_seq_length,
			output_mode=self.output_mode,
			pad_on_left=False,
			pad_token=self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0],
			pad_token_segment_id=0,
		)
		all_input_ids=torch.tensor([f.input_ids for f in features],dtype=torch.long)
		all_attention_mask=torch.tensor([f.attention_mask for f in features],dtype=torch.long)
		all_token_type_ids=torch.tensor([f.token_type_ids for f in features],dtype=torch.long)
		all_labels=torch.tensor([f.label for f in features],dtype=(torch.long if self.output_mode=='classification' else torch.float))
		dataset=TensorDataset(all_input_ids,all_attention_mask,all_token_type_ids,all_labels)
		return dataset

	def fewrel(self,inputs):
		dataset=self.get_dataset(inputs)
		sampler=SequentialSampler(dataset)
		dataloader=DataLoader(dataset,sampler=sampler,batch_size=self.batch_size)
		outputs=[]
		for it in dataloader:
			if self.CUDA: it=tuple(t.cuda() for t in it)
			inputs={
				'input_ids':it[0],
				'attention_mask':it[1],
				'token_type_ids':it[2],
				'labels':it[3],
				'MLP':True,
			}
			sequence_output=self.model(**inputs)
			outputs.append(sequence_output)
		return outputs

if __name__=='__main__':
	bert=BertModel()
	inputs=[]
	for i in range(10):
		inputs.append("subject item belongs to a specific group of domestic animals, generally given by association")
	outputs=bert.fewrel(inputs)#[batch_size,hidden_size]
	for _ in outputs:
		print(_.size())












