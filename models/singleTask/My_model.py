"""
暑期MSA任务
"""
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from ..subNets import BertTextEncoder

__all__ = ['My_model']



class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class My_model(nn.Module):
    def __init__(self,args):
        super(My_model, self).__init__()
        self.batch_size = args.batch_size
        self.a_length,self.v_length = args.seq_len
        self.t_dim,self.a_dim,self.v_dim = args.pre_dim
        self.t_hieen_dim,self.a_hidden_dim,self.v_hidden_dim =args.hidden_dims
        self.post_fusion_dim = args.post_fusion_dim
        self.drop, self.layers = args.dropouts, args.num_layers
        self.text_model = Text_model(args)
        self.video_model = Seq2Seq(en_input_dim=self.t_hieen_dim,en_hidden_dim=self.v_hidden_dim,layers=self.layers,dropout=self.drop,
                             d_i_dim=self.v_dim,d_h_dim=self.v_hidden_dim,d_o_dim=self.v_dim)
        self.audio_model = Seq2Seq(en_input_dim=self.t_hieen_dim,en_hidden_dim=self.a_hidden_dim,layers=self.layers,dropout=self.drop,
                             d_i_dim=self.a_dim,d_h_dim=self.a_hidden_dim,d_o_dim=self.a_dim)
        self.a_hidden_linear = nn.Sequential(nn.Linear(self.a_hidden_dim,self.a_hidden_dim,bias=True),
                                             Swish(),
                                             nn.Linear(self.a_hidden_dim,self.post_fusion_dim,bias=True))
        self.v_hidden_linear = nn.Sequential(nn.Linear(self.v_hidden_dim,self.v_hidden_dim,bias=True),
                                             Swish(),
                                             nn.Linear(self.v_hidden_dim,self.post_fusion_dim,bias=True))
        self.t_hidden_linear = nn.Sequential(nn.Linear(self.t_hieen_dim, self.t_hieen_dim,bias=True),
                                             Swish(),
                                             nn.Linear(self.t_hieen_dim, self.post_fusion_dim,bias=True))
        self.v_ln = nn.LayerNorm( self.v_length * self.v_dim)
        self.a_ln = nn.LayerNorm( self.a_length * self.a_dim)
        self.fusion_linear = nn.Sequential(
            nn.Linear(self.a_length*self.a_dim + self.v_length*self.v_dim + self.t_hieen_dim,self.t_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.t_dim,self.post_fusion_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.post_fusion_dim,self.post_fusion_dim,bias=True)
        )
        self.fusion_linear2 = nn.Sequential(
            nn.Linear(self.t_hieen_dim+self.a_hidden_dim+self.v_hidden_dim,self.post_fusion_dim,bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(self.post_fusion_dim,self.post_fusion_dim,bias=True)
        )
        self.out_linear = nn.Sequential(
            nn.Linear(self.post_fusion_dim,self.post_fusion_dim,bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(self.post_fusion_dim,1,bias=True)
        )
    def forward(self,text,audio,video):
        audio, audio_lengths = audio
        video, video_lengths = video

        mask_len = torch.sum(text[:, 1, :], dim=1, keepdim=True)
        text_lengths = mask_len.squeeze().int().detach().cpu()
        text_seq,text_hidden = self.text_model(text,text_lengths)
        # Seq2Seq  out_dim:   [batch_size,d_input_seq,d_dim] , encoder出来的[batch_size,d_hidden_dim], decoder出来的[batch_size,d_hidden_dim]
        t2a, t2a_hidden, a_hidden = self.audio_model(e_inputs=text_seq,e_lengths=text_lengths,d_inputs=audio,d_lengths=audio_lengths)
        t2v, t2v_hidden, v_hidden= self.video_model(e_inputs=text_seq,e_lengths=text_lengths,d_inputs=video,d_lengths=video_lengths)
        '''
            text_seq: [seq_len,batch_size,dim]   t_hidden: [batch_size,dim]
            t2a t2v  audio video a_hidden v_hidden
        '''
        #语言和语言算对比学习 对于encoder的处理
        text_hidden_,t2a_hidden,t2v_hidden= self.t_hidden_linear(text_hidden),self.a_hidden_linear(t2a_hidden),self.v_hidden_linear(t2v_hidden)
        loss = cor_loss(text_hidden_,t2a_hidden,t2v_hidden)
        #对于decoder的处理
        padding_a = torch.zeros([self.batch_size,self.a_length-t2a.shape[1],self.a_dim]).to('cuda')
        t2a = torch.cat([t2a,padding_a],dim=1).flatten(start_dim=1,end_dim=2)
        padding_v = torch.zeros([self.batch_size,self.v_length-t2v.shape[1],self.v_dim]).to('cuda')
        t2v = torch.cat([t2v,padding_v],dim=1).flatten(start_dim=1,end_dim=2)
        # 碾平
        audio,video = audio.flatten(start_dim=1,end_dim=2),video.flatten(start_dim=1,end_dim=2)
        audio , t2a = self.a_ln(audio), self.a_ln(t2a)
        video , t2v = self.v_ln(video), self.v_ln(t2v)
        # 连接inputs
        input1,input2 = torch.cat([text_hidden,audio,video],dim=-1),torch.cat([text_hidden,t2a,t2v],dim=-1)
        hidden1 = self.fusion_linear(input1)
        hidden2 = self.fusion_linear(input2)
        input3 = torch.cat([text_hidden,a_hidden,v_hidden],dim=-1)
        hidden3 = self.fusion_linear2(input3)
        out1,out2,out3 = self.out_linear(hidden1),self.out_linear(hidden2),self.out_linear(hidden3)
        out = torch.cat([out1,out2,out3],dim=0)
        return out,loss/self.batch_size



def count_extra_loss(text_hidden,t2a_hidden):
    te = 5e-5
    text_hidden, t2a_hidden = text_hidden.cpu().detach().numpy(), t2a_hidden.cpu().detach().numpy()
    text_norm, t2a_norm = np.linalg.norm(text_hidden, ord=2, axis=1, keepdims=True),np.linalg.norm(t2a_hidden, ord=2, axis=1, keepdims=True)
    text_hidden, t2a_hidden = text_hidden/text_norm, t2a_hidden/t2a_norm
    logits = torch.tensor(np.dot(text_hidden, t2a_hidden.T) * np.exp(te))
    labels = torch.tensor(np.arange(text_hidden.shape[0]))
    loss = 0.0
    loss += F.cross_entropy(logits.t(), labels)
    loss += F.cross_entropy(logits, labels)
    return loss/2
def cor_loss(t1,t2,t3):
    loss = 0.0
    loss += count_extra_loss(t1, t2)
    loss += count_extra_loss(t1, t3)
    loss += count_extra_loss(t2, t3)
    return loss

class Text_model(nn.Module):
    def __init__(self,args):
        super(Text_model, self).__init__()
        self.bert = BertTextEncoder(use_finetune=args.use_finetune, transformers=args.transformers,pretrained=args.pretrained)
        self.text_encoder = Encoder(args.pre_dim[0],args.hidden_dims[0],args.num_layers,args.dropouts)
    def forward(self,text,text_lengths):
        text = self.bert(text)
        text_seq,_,text_hidden = self.text_encoder(text,text_lengths)
        return text_seq.permute(1,0,2),text_hidden

class Seq2Seq(nn.Module):
    def __init__(self, en_input_dim, en_hidden_dim, layers, dropout, d_i_dim, d_h_dim, d_o_dim):
        super(Seq2Seq, self).__init__()
        self.en_input_dim,self.en_hidden_dim,self.layers,self.dropout = en_input_dim, en_hidden_dim, layers, dropout
        self.decoder_input_dim, self.decoder_hidden_dim, self.decoder_out_dim = d_i_dim, d_h_dim, d_o_dim
        self.encoder = Encoder(input_dim=self.en_input_dim,hidden_dim=self.en_hidden_dim,layers=self.layers,dropout=self.dropout)
        self.decoder = Decoder(i_dim=self.decoder_input_dim,h_dim=self.decoder_hidden_dim,o_dim=self.decoder_out_dim,layers=self.layers,dropout=self.dropout)

    def forward(self,e_inputs,e_lengths,d_inputs,d_lengths):
        '''输出维度：
         en_outputs：[seq_len,batch_size,hidden_dim]
         en_hidden:[num_layers,batch_size,hidden_dim]
         utter : [batch,hidden_dim]
        '''
        en_outputs ,en_hidden,utter = self.encoder(e_inputs,e_lengths)
        d_outputs,d_hidden = self.decoder(d_inputs,d_lengths,en_hidden,en_outputs)
        #
        return d_outputs.permute(1,0,2),utter,d_hidden


class Encoder(nn.Module):
    def __init__(self,input_dim,hidden_dim,layers,dropout):
        super(Encoder, self).__init__()
        self.input_dim, self.hidden_dim, self.layers, self.dropout = input_dim, hidden_dim, layers,dropout
        self.gru = nn.GRU(input_size=self.input_dim,hidden_size=self.hidden_dim,num_layers=self.layers,dropout=self.dropout,bidirectional=True)
    def forward(self,inputs,lengths):
        inputs = inputs.permute(1,0,2)
        packed = nn.utils.rnn.pack_padded_sequence(inputs,lengths.cpu(),enforce_sorted=False)
        outputs, hidden = self.gru(packed)
        outputs,_ = nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = torch.split(outputs,self.hidden_dim,dim=2)
        outputs = sum(outputs)/len(outputs)
        hidden = torch.split(hidden.view(self.layers, -1, hidden.size(1), hidden.size(2)), 1, dim=1)
        hidden = torch.squeeze(sum(hidden) / len(hidden), 1)
        utter = torch.add(hidden[0],hidden[-1])
        return outputs, hidden, utter


class Decoder(nn.Module):
    def __init__(self,i_dim,h_dim,o_dim,layers,dropout):
        super(Decoder, self).__init__()
        self.i_dim = i_dim
        self.h_dim = h_dim
        self.o_dim = o_dim
        self.layers = layers
        self.dropout = dropout
        self.gru = nn.GRU(i_dim,h_dim,num_layers=layers,dropout=dropout,bidirectional=False)
        self.linear_1 = nn.Linear(2*h_dim, h_dim,bias=True)
        self.act_1 = Swish()
        self.linear_2 = nn.Linear(h_dim, o_dim,bias=True)
    def forward(self,d_inputs,d_lengths,hidden,e_outputs):
        d_inputs = d_inputs.permute(1, 0, 2)
        packed = nn.utils.rnn.pack_padded_sequence(d_inputs, d_lengths.cpu(), enforce_sorted=False)
        outputs, d_hidden = self.gru(packed, hidden)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)  # out: [seq_len,batch_size,hidden_dim]
        # outputs = Q, e_outputs = K = V
        Q = outputs.permute(1, 0, 2)
        K = e_outputs.permute(1, 2, 0)
        V = e_outputs.permute(1, 0, 2)
        attn = attention(Q, K, V)  # [seq_len,batch_size,hidden_dim]
        outputs = torch.cat([outputs, attn], 2)
        outputs = self.act_1(self.linear_1(outputs))
        outputs = self.act_1(self.linear_2(outputs))
        #处理Hidden
        d_hidden = torch.mean(d_hidden,dim=0)

        return outputs,d_hidden  # [seq_len,batch_size,hidden_dim]
        # [seq_len,batch_size,hidden_dim]



def attention(query, key, value):
    if query.dim() == 3:
        score = F.softmax(torch.bmm(query, key), dim=2)
        attn = torch.bmm(score, value)
        return attn.transpose(0, 1)
    else:
        score = F.softmax(torch.matmul(query, key), dim=1)
        attn = torch.matmul(score, value)
        return attn