import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer

__all__ = ['BertTextEncoder']

TRANSFORMERS_MAP = {
    'bert': (BertModel, BertTokenizer),
    'roberta': (RobertaModel, RobertaTokenizer),
}

class BertTextEncoder(nn.Module):
    def __init__(self, use_finetune=False, transformers='bert', pretrained='bert-base-uncased'):
        super().__init__()

        tokenizer_class = TRANSFORMERS_MAP[transformers][1]
        model_class = TRANSFORMERS_MAP[transformers][0]
        self.tokenizer = tokenizer_class.from_pretrained(pretrained)
        self.model = model_class.from_pretrained(pretrained)
        self.use_finetune = use_finetune
    
    def get_tokenizer(self):
        return self.tokenizer
    
    # def from_text(self, text):
    #     """
    #     text: raw data
    #     """
    #     input_ids = self.get_id(text)
    #     with torch.no_grad():
    #         last_hidden_states = self.model(input_ids)[0]  # Models outputs are now tuples
    #     return last_hidden_states.squeeze()
    
    def forward(self, text):
        """
        text: (batch_size, 3, seq_len)
        3: input_ids, input_mask, segment_ids
        input_ids: input_ids,
        input_mask: attention_mask,
        segment_ids: token_type_ids
        """
        input_ids, input_mask, segment_ids = text[:,0,:].long(), text[:,1,:].float(), text[:,2,:].long()
        if self.use_finetune:
            last_hidden_states = self.model(input_ids=input_ids,
                                            attention_mask=input_mask,
                                            token_type_ids=segment_ids)[0]  # Models outputs are now tuples
        else:
            with torch.no_grad():
                last_hidden_states = self.model(input_ids=input_ids,
                                                attention_mask=input_mask,
                                                token_type_ids=segment_ids)[0]  # Models outputs are now tuples
        return last_hidden_states


class Bert_Raw_TextEncoder(nn.Module):
    def __init__(self, use_finetune=True, pretrained="/home/yyj/pretrained_berts/bert_cn"):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(pretrained)
        self.model = BertModel.from_pretrained(pretrained)
        self.embedding = self.model.embeddings.word_embeddings
        self.use_finetune = use_finetune
    def forward(self, raw_text):
        model_input = self.raw_text2text(raw_text=raw_text)
        input_ids = model_input['input_ids']
        mask = model_input['attention_mask']
        output = self.model(input_ids = model_input['input_ids'],attention_mask = model_input['attention_mask'])
        output_seq , outputs = output['last_hidden_state'],output['pooler_output']
        embedding = self.embedding(model_input['input_ids'])
        return outputs, embedding, input_ids, mask,output_seq
    def raw_text2text(self,raw_text):
        res = self.tokenizer(raw_text,padding='max_length',truncation=True,max_length=50,return_tensors='pt',return_length=True).to('cuda')
        return res


class RoBerta_raw_embedding(nn.Module):
    def __init__(self,use_finetune=True, pretrained="hfl/chinese-roberta-wwm-ext-large"):
        super(RoBerta_raw_embedding, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(pretrained)
        self.model = RobertaModel.from_pretrained(pretrained)
        self.embedding = self.model.embeddings.word_embeddings
        self.use_finetune = use_finetune
    def forward(self,raw_text):
        encoded_input = self.raw2model_input(raw_text)
        del encoded_input['length']
        inputs_ids, mask = encoded_input['input_ids'], encoded_input['attention_mask']
        output = self.model(**encoded_input)
        out,out_seq = output['pooler_output'],output['last_hidden_state']
        embedding = self.embedding(inputs_ids)
        return out, embedding, inputs_ids, mask,out_seq

    def raw2model_input(self,raw_text):
        res = self.tokenizer(raw_text,padding='max_length',truncation=True,max_length=50,return_tensors='pt',return_length=True).to('cuda')
        return res