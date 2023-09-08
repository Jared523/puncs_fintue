import torch
import torch.nn as nn
from transformers import BertPreTrainedModel
from crf import CRF


def valid_sequence_output(sequence_output, valid_mask, attention_mask):
    batch_size, max_len, feat_dim = sequence_output.shape
    # valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32,
    #                            device='cuda' if torch.cuda.is_available() else 'cpu')
    # valid_attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long,
    #                                    device='cuda' if torch.cuda.is_available() else 'cpu')
    valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32, device = sequence_output.device)
    valid_attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long, device = sequence_output.device)
    for i in range(batch_size):
        jj = -1
        for j in range(max_len):
            if valid_mask[i][j].item() == 1:
                jj += 1
                valid_output[i][jj] = sequence_output[i][j]
                valid_attention_mask[i][jj] = attention_mask[i][j]
    return valid_output, valid_attention_mask


class PuncBert(BertPreTrainedModel):
    def __init__(self, args):
        super(PuncBert, self).__init__(args.bert.config)
        self.num_labels = args.num_labels
        self.bert = args.bert
        self.dropout = nn.Dropout(args.dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size, args.num_labels)
        self.loss_fun = nn.CrossEntropyLoss()
        # self.post_init()
        
    def forward(
            self,
            input_ids,
            attention_mask,
            token_type_ids,
            valid_mask = None,
            mask_labels = None,
            mode = 'test'
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = outputs[0] 
        sequence_output, attention_mask = valid_sequence_output(sequence_output, valid_mask, attention_mask)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output) # bs x seq_len x num_labels
        
        active_loss = attention_mask.view(-1) == 1
        active_logits = logits.contiguous().view(-1, self.num_labels)[active_loss]

        if mode == 'train':
            active_labels = mask_labels[(mask_labels != -100)]
            loss = self.loss_fun(active_logits, active_labels)
            return loss
        else:
            pred = active_logits.argmax(-1)
            return pred


class PuncBert_freeze(BertPreTrainedModel):
    def __init__(self, args):
        super(PuncBert_freeze, self).__init__(args.bert.config)
        self.num_labels = args.num_labels
        self.bert = args.bert
        self.dropout = nn.Dropout(args.dropout_prob)
        self.classifier = nn.Linear(4*self.bert.config.hidden_size, args.num_labels)
        self.loss_fun = nn.CrossEntropyLoss()
        # self.post_init()
        
    def forward(
            self,
            input_ids,
            attention_mask,
            token_type_ids,
            pun_state,
            valid_mask = None,
            mask_labels = None,
            mode = 'test'
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = outputs[0] 
        sequence_output, attention_mask = valid_sequence_output(sequence_output, valid_mask, attention_mask)
        sequence_output = self.dropout(sequence_output)
        pun=pun_state.repeat(sequence_output.shape[0],sequence_output.shape[1],1)
        sequence_output=torch.concat((sequence_output,pun),dim=2)
        
        logits = self.classifier(sequence_output) # bs x seq_len x num_labels
        
        active_loss = attention_mask.view(-1) == 1
        active_logits = logits.contiguous().view(-1, self.num_labels)[active_loss]

        if mode == 'train':
            active_labels = mask_labels[(mask_labels != -100)]
            loss = self.loss_fun(active_logits, active_labels)
            return loss
        else:
            pred = active_logits.argmax(-1)
            return pred


class PuncLSTM(BertPreTrainedModel):
    def __init__(self, args):
        super(PuncLSTM, self).__init__(args.bert.config)
        self.num_labels = args.num_labels
        self.bert = args.bert
        self.dropout = nn.Dropout(args.dropout_prob)
        self.num_layers = 2
        self.rnn_hidden = 128
        self.lstm = nn.LSTM(self.bert.config.hidden_size, self.rnn_hidden, self.num_layers,
                            bidirectional=True, batch_first=True, dropout=args.dropout_prob)
        self.classifier = nn.Linear(self.rnn_hidden * 2, args.num_labels)
        self.loss_fun = nn.CrossEntropyLoss()
        # self.post_init()
        
    def forward(
            self,
            input_ids,
            attention_mask,
            token_type_ids,
            valid_mask = None,
            mask_labels = None,
            mode = 'test'
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = outputs[0] 
        sequence_output, attention_mask = valid_sequence_output(sequence_output, valid_mask, attention_mask)
        sequence_output = self.lstm(sequence_output)[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output) # bs x seq_len x num_labels
        
        active_loss = attention_mask.view(-1) == 1
        active_logits = logits.contiguous().view(-1, self.num_labels)[active_loss]

        if mode == 'train':
            active_labels = mask_labels[(mask_labels != -100)]
            loss = self.loss_fun(active_logits, active_labels)
            return loss
        else:
            pred = active_logits.argmax(-1)
            return pred
             
class PuncCRF(BertPreTrainedModel):
    def __init__(self, args):
        super(PuncCRF, self).__init__(args.bert.config)
        self.num_labels = args.num_labels
        self.bert = args.bert
        self.dropout = nn.Dropout(args.dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size, args.num_labels)
        self.crf = CRF(num_tags=self.num_labels, batch_first=True)
        # self.post_init()
        
    def forward(
            self,
            input_ids,
            attention_mask,
            token_type_ids,
            valid_mask=None,
            mask_labels=None,
            mode = "test",
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = outputs[0] 
        sequence_output, attention_mask = valid_sequence_output(sequence_output, valid_mask, attention_mask)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output) # bs x seq_len x num_labels
          
        if mode == 'train':
            labels = torch.where(mask_labels >= 0, mask_labels, torch.zeros_like(mask_labels))
            loss = -1 * self.crf(emissions=logits, tags=labels, mask=attention_mask)
            return loss
        else:
            pred = self.crf.decode(logits, attention_mask) # bs x seq_len
            pred = pred[attention_mask==1]
            return pred



class ChineseRobertaNoCNNLstmPunc(nn.Module):
    # NOTE bert hidden_size=384
    def __init__(self, segment_size, output_size, dropout, vocab_size):
        super(ChineseRobertaNoCNNLstmPunc, self).__init__()
        self.bert = BertModel.from_pretrained('./models/chinese-roberta-wwm-ext')
        self.bert.embedding_size = 768
        self.hidden_size = 200
        print(type(self.bert))
        # self.bert_vocab_size = vocab_size
        # self.bn = nn.BatchNorm1d(segment_size*self.bert_vocab_size)
        # self.fc = nn.Linear(segment_size*self.bert_vocab_size, output_size)
        self.encoder = EncoderStackedNoCnnLSTMLSTM(
                bert=self.bert,
                hidden_size=self.hidden_size,
                n_layers=3,
                # 额外的参数
                # cnn -> rnn
                # NOTE 使用了args作为参数传递工具
                cnn_kernel_size=(5, 20),
                # rnn -> cnn
                # cnn_kernel_size=(3, hidden_size*2),
                cnn_filter_num=5
            )

        # 以rnn输出为输入
        self.fc = nn.Linear(
                # fc_input：late_fusion `f_t = a_t W_{fa} ◦ σ(a_t W_{fa} W_{ff} + h_t W_{fh} + b_f) + h_t `
                # size: [self.hidden_size*2] 双向的尺寸
                self.hidden_size*2,
                output_size
            )

        # self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # print('enc_hidden size', enc_hidden.shape)
        # [B, S, H]
        outputs = self.encoder(x)     # 输入pack，lstm默认输出pack

        # 记录尺寸，进行变换，以对每一时间步的输出进行late fuse
        out_size_B = outputs.size(0)
        out_size_S = outputs.size(1)
        out_size_H = outputs.size(2)

        # print(out_size_S, out_size_B)
        # print(outputs[1].shape)
        outputs = outputs.contiguous()
        # print(outputs.shape)
        outputs = outputs.view(out_size_S*out_size_B, out_size_H)

        outputs = outputs.contiguous()


        # for i in range(l):
        #     print('{} shape:'.format(i), x[i].shape)
        # print('x', type(x))
        # print('shape', x.shape)
        # x = self.fc(self.dropout(self.bn(x)))
        x = self.fc(outputs)
        return x


class EncoderStackedNoCnnLSTMLSTM(nn.Module):
    """添加额外CNN向量
        1、cnn提取特征，后RNN；
        2、bert直连LSTM
        1+2
    """
    def __init__(self, bert, hidden_size, n_layers, cnn_kernel_size, cnn_filter_num):
        super(EncoderStackedNoCnnLSTMLSTM, self).__init__()
        self.bert = bert
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_filter_num = cnn_filter_num

        # GRU输入为bert + cnn_out
        # self.gru = nn.GRU(bert_size+(bert_size-cnn_kernel_size[1]+1)*cnn_filter_num, hidden_size, n_layers, bidirectional=True, batch_first=False)

        # cnn_kernel_size包含高度（多少词一个卷）和宽度（每个词的多少位一个卷）
        self.conv = nn.ModuleList()
        cnn_layer_num = 4
        for i in range(cnn_layer_num):
            module_tmp = nn.ModuleDict({
                'conv_w_{}'.format(i):
                nn.Conv2d(1 if i == 0 else cnn_filter_num,
                          cnn_filter_num,
                          cnn_kernel_size,
                          padding=((cnn_kernel_size[0] - 1) // 2,
                                   (cnn_kernel_size[1] - 1) // 2)),
                'conv_v_{}'.format(i):
                nn.Conv2d(1 if i == 0 else cnn_filter_num,
                          cnn_filter_num,
                          cnn_kernel_size,
                          padding=((cnn_kernel_size[0] - 1) // 2,
                                   (cnn_kernel_size[1] - 1) // 2)),
            })
            self.conv.append(module_tmp)

        # LSTM输入为bert + cnn_out
        self.lstm = nn.ModuleDict({
            'cnn_lstm': nn.LSTM(
                # cnn的每个词对应的输入尺寸为 cnn_layer_num=4层CNN，每层cnn_filter_num=10 filter
                input_size=(bert.embedding_size-((cnn_kernel_size[1]-1) % 2 * cnn_layer_num))*cnn_filter_num,
                hidden_size=hidden_size, num_layers=n_layers, bidirectional=True, batch_first=True
                ),
            'emb_lstm': nn.LSTM(
                bert.embedding_size,
                hidden_size,
                n_layers,
                bidirectional=True,
                batch_first=False
                ),
        })

    # @torchsnooper.snoop()
    def forward(self, word_inputs):
        # Note: we run this all at once (over the whole input sequence)
        # [B, S, Emb]
        embedded = self.bert(word_inputs)[0]
        emb_lstm_input = embedded
        # output1 = [B, S, 2H]      hidden1=[lyr*direct, B, H]
        output2, hidden2 = self.lstm['emb_lstm'](emb_lstm_input, None)

        # 在第三维cat起来
        output = output2

        # output size: [S, B, 4H];      hidden: ([lyr*direct, B, H], [lyr*direct, B, H])
        return output
