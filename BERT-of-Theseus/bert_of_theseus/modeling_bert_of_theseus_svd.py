"""PyTorch BERT-of-Theseus model. """

from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.distributions.bernoulli import Bernoulli

from transformers.modeling_utils import PreTrainedModel, prune_linear_layer
from transformers.configuration_bert import BertConfig
from transformers.modeling_bert import BERT_PRETRAINED_MODEL_ARCHIVE_MAP, load_tf_weights_in_bert, gelu, gelu_new, \
    swish, mish, ACT2FN, BertLayerNorm, BertEmbeddings, BertSelfAttention, BertSelfOutput, BertAttention, \
    BertIntermediate, BertOutput, BertLayer, BertPooler, BertPredictionHeadTransform, BertLMPredictionHead, \
    BertOnlyMLMHead, BertOnlyNSPHead, BertPreTrainingHeads

import sys


sys.path.append('/home/Project/DistillingMPO/OPF')


from compress_tools.MPOtorch import LinearDecomMPO
from compress_tools.Matrix2MPO import MPO
import re


logger = logging.getLogger(__name__)
# input3072_size = [2,2,2,2,2,2,2,2,2,2,3]
# input768_size = [2,2,2,2,2,1,1,2,2,2,3]

# input3072_size = [64,1,1,1,1,1,48]
# input768_size = [32,1,1,1,1,1,24]

input3072_size = [64,48]
input768_size = [32,24]

input4096_size = [64,1,64]
input1024_size = [32,1,32]

input512_size = [32,1,1,1,1,1,1,1,1,1,1,1,1,1,16]
input2048_size = [16,2,1,1,1,1,1,1,1,1,1,1,1,2,32]
SHAPE_CONFIG = {
    "attention":(input768_size,input768_size),
    "FFN_1":(input3072_size,input768_size),
    "FFN_2":(input768_size,input3072_size)
}


def _determine_type(name):
    if 'intermediate' in name:
        return 'FFN_1'
    elif 'output' in name and 'attention' not in name:
        return 'FFN_2'
    elif 'query' in name or 'key' in name or 'value' in name or ('attention' in name and 'output' in name):
        return 'attention'



class BertEncoder(nn.Module):
    def __init__(self, config, scc_n_layer=6):
        super(BertEncoder, self).__init__()
        self.prd_n_layer = config.num_hidden_layers
        self.scc_n_layer = scc_n_layer
        assert self.prd_n_layer % self.scc_n_layer == 0
        self.compress_ratio = self.prd_n_layer // self.scc_n_layer
        self.bernoulli = None
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(self.prd_n_layer)])
        self.scc_layer = nn.ModuleList([BertLayer(config) for _ in range(self.scc_n_layer)])


    def loss_layer_fn(self, choose_name):
        loss_layer = 0
        for module_name in choose_name:
            # get_module_from_name 
            ind = re.findall(r"\d+",module_name)[0]
            ind_teacher = str(int(re.findall(r"\d+",module_name)[0]) *2 + 1)

            module_name_teacher = module_name.replace(f"{ind}",f"[{ind_teacher}]")
            module_name = module_name.replace(f"{ind}",f"[{ind}]")
            
            module_name_teacher = module_name_teacher.replace(".weight","")
            module_name = module_name.replace(".weight","")

            layer_module_teacher = eval("self.layer"+module_name_teacher)
            layer_module = eval("self.scc_layer"+module_name)

            type_name_teacher = _determine_type(module_name_teacher)
            type_name = _determine_type(module_name)


            FINE_INPUT_SHAPE_teacher, FINE_OUTPUT_SHAPE_teacher = SHAPE_CONFIG[type_name_teacher]
            FINE_INPUT_SHAPE, FINE_OUTPUT_SHAPE = SHAPE_CONFIG[type_name]

            bias_teacher = layer_module_teacher.bias
            bias = layer_module.bias

            # (1) TODO: mpo decomposition for MPO module
            # layer_module.from_pretrained(None, None, mpo_tensor_set, layer_module.bias)
            # (2) mpo decomposition for ori module
            obj_name_teacher, weight_name_teacher = module_name_teacher.rsplit('.',1)
            obj_name, weight_name = module_name.rsplit('.',1)

            module_name_teacher = module_name_teacher + '.tensor_set.1'

            module_name_teacher = module_name_teacher.split('.', 1)[1]
            module_name = module_name.split('.', 1)[1]
            layer_teacher = self.layer[int(ind_teacher)].state_dict()[module_name_teacher]
            layer_student = self.scc_layer[int(ind)].state_dict()[module_name_teacher]

            loss_layer = loss_layer + MSELoss()(layer_teacher, layer_student)
        return loss_layer/len(choose_name)



    def fine_grained_decomposition(self, module_name):
        # get_module_from_name 
        ind = re.findall(r"\d+",module_name)[0]
        module_name = module_name.replace(f"{ind}",f"[{ind}]")
        module_name = module_name.replace(".weight","")
        layer_module = eval("self.scc_layer"+module_name)

        type_name = _determine_type(module_name)
        FINE_INPUT_SHAPE, FINE_OUTPUT_SHAPE = SHAPE_CONFIG[type_name]
        mpo = MPO(FINE_INPUT_SHAPE, FINE_OUTPUT_SHAPE, None)
        device = layer_module.weight.device
        # mpo_tensor_set, _, _ = mpo.matrix2mpo(layer_module.get_weight().cpu().detach().numpy()) # .query_mpo
        #得到分解tensor
        mpo_tensor_set, _, _ = mpo.matrix2mpo(layer_module.weight.cpu().detach().numpy()) # .query
        bias = layer_module.bias
        # (1) TODO: mpo decomposition for MPO module
        # layer_module.from_pretrained(None, None, mpo_tensor_set, layer_module.bias)
        # (2) mpo decomposition for ori module
        obj_name, weight_name = module_name.rsplit('.',1)
        obj = eval("self.scc_layer"+obj_name)
        setattr(obj, weight_name, LinearDecomMPO(FINE_INPUT_SHAPE, FINE_OUTPUT_SHAPE, None))
        layer_module_new = eval("self.scc_layer"+module_name)
        # 放进device中
        layer_module_new.from_pretrained(None, None, mpo_tensor_set, bias, device=device)
        
    def fine_grained_decomposition2(self, module_name):
            # get_module_from_name 
            ind = re.findall(r"\d+",module_name)[0]
            module_name = module_name.replace(f"{ind}",f"[{ind}]")
            module_name = module_name.replace(".weight","")
            layer_module = eval("self.layer"+module_name)

            type_name = _determine_type(module_name)
            FINE_INPUT_SHAPE, FINE_OUTPUT_SHAPE = SHAPE_CONFIG[type_name]
            mpo = MPO(FINE_INPUT_SHAPE, FINE_OUTPUT_SHAPE, None)
            device = layer_module.weight.device
            # mpo_tensor_set, _, _ = mpo.matrix2mpo(layer_module.get_weight().cpu().detach().numpy()) # .query_mpo
            #得到分解tensor
            mpo_tensor_set, _, _ = mpo.matrix2mpo(layer_module.weight.cpu().detach().numpy()) # .query
            bias = layer_module.bias
            # (1) TODO: mpo decomposition for MPO module
            # layer_module.from_pretrained(None, None, mpo_tensor_set, layer_module.bias)
            # (2) mpo decomposition for ori module
            obj_name, weight_name = module_name.rsplit('.',1)
            obj = eval("self.layer"+obj_name)
            setattr(obj, weight_name, LinearDecomMPO(FINE_INPUT_SHAPE, FINE_OUTPUT_SHAPE, None))
            layer_module_new = eval("self.layer"+module_name)
            # 放进device中
            layer_module_new.from_pretrained(None, None, mpo_tensor_set, bias, device=device)
    def set_replacing_rate(self, replacing_rate):
        if not 0 < replacing_rate <= 1:
            raise Exception('Replace rate must be in the range (0, 1]!')
        self.bernoulli = Bernoulli(torch.tensor([replacing_rate]))

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None,
                encoder_attention_mask=None):
        all_hidden_states = ()
        all_attentions = ()
        layer_outputs_list = () 
        scc_layer_outputs_list = ()  
        

        inference_layers1 = []
        for i in range(self.scc_n_layer):
            inference_layers1.append(self.scc_layer[i])
        for i, layer_module in enumerate(inference_layers1):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i], encoder_hidden_states,
                                         encoder_attention_mask)
            scc_layer_outputs_list = scc_layer_outputs_list + (layer_outputs[0],)


        inference_layers2 = []
        for i in range(self.scc_n_layer):
            for offset in range(self.compress_ratio):
                inference_layers2.append(self.layer[i * self.compress_ratio + offset])
        for i, layer_module in enumerate(inference_layers2):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i], encoder_hidden_states,
                                         encoder_attention_mask)
            layer_outputs_list = layer_outputs_list + (layer_outputs[0],)


        if self.training:
            inference_layers = []
            for i in range(self.scc_n_layer):
                if self.bernoulli.sample() == 1:  # REPLACE
                    inference_layers.append(self.scc_layer[i])
                else:  # KEEP the original
                    for offset in range(self.compress_ratio):
                        inference_layers.append(self.layer[i * self.compress_ratio + offset])

        else:  # inference with compressed model
            inference_layers = self.scc_layer

        for i, layer_module in enumerate(inference_layers):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i], encoder_hidden_states,
                                         encoder_attention_mask)
            hidden_states = layer_outputs[0]




            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs,layer_outputs_list,scc_layer_outputs_list  # last-layer hidden state, (all hidden states), (all attentions)


class BertPreTrainedModel(PreTrainedModel):
    config_class = BertConfig
    pretrained_model_archive_map = BERT_PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "bert"

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class BertModel(BertPreTrainedModel):
    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,
                head_mask=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                causal_mask = causal_mask.to(
                    torch.long)  # not converting to long will cause errors with pytorch version < 1.3
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError("Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(input_shape,
                                                                                                        attention_mask.shape))

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)

            if encoder_attention_mask.dim() == 3:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
            elif encoder_attention_mask.dim() == 2:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
            else:
                raise ValueError(
                    "Wrong shape for encoder_hidden_shape (shape {}) or encoder_attention_mask (shape {})".format(
                        encoder_hidden_shape,
                        encoder_attention_mask.shape))

            encoder_extended_attention_mask = encoder_extended_attention_mask.to(
                dtype=next(self.parameters()).dtype)  # fp16 compatibility
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(
                    -1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype)  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids,
                                           token_type_ids=token_type_ids, inputs_embeds=inputs_embeds)
        encoder_outputs,layer_outputs_list,scc_layer_outputs_list = self.encoder(embedding_output,
                                       attention_mask=extended_attention_mask,
                                       head_mask=head_mask,
                                       encoder_hidden_states=encoder_hidden_states,
                                       encoder_attention_mask=encoder_extended_attention_mask)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[
                                                      1:]  # add hidden_states and attentions if they are here
        return outputs,layer_outputs_list,scc_layer_outputs_list  # sequence_output, pooled_output, (hidden_states), (attentions)


class BertForPreTraining(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForPreTraining, self).__init__(config)

        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None,
                masked_lm_labels=None, next_sentence_label=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        outputs = (prediction_scores, seq_relationship_score,) + outputs[
                                                                 2:]  # add hidden states and attention if they are here

        if masked_lm_labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss
            outputs = (total_loss,) + outputs

        return outputs  # (loss), prediction_scores, seq_relationship_score, (hidden_states), (attentions)


class BertForMaskedLM(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForMaskedLM, self).__init__(config)

        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None,
                masked_lm_labels=None, encoder_hidden_states=None, encoder_attention_mask=None, lm_labels=None, ):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds,
                            encoder_hidden_states=encoder_hidden_states,
                            encoder_attention_mask=encoder_attention_mask)

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

        # Although this may seem awkward, BertForMaskedLM supports two scenarios:
        # 1. If a tensor that contains the indices of masked labels is provided,
        #    the cross-entropy is the MLM cross-entropy that measures the likelihood
        #    of predictions for masked words.
        # 2. If `lm_labels` is provided we are in a causal scenario where we
        #    try to predict the next token for each input in the decoder.
        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)  # -1 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            outputs = (masked_lm_loss,) + outputs

        if lm_labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            prediction_scores = prediction_scores[:, :-1, :].contiguous()
            lm_labels = lm_labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            ltr_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), lm_labels.view(-1))
            outputs = (ltr_lm_loss,) + outputs

        return outputs  # (masked_lm_loss), (ltr_lm_loss), prediction_scores, (hidden_states), (attentions)


class BertForNextSentencePrediction(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForNextSentencePrediction, self).__init__(config)

        self.bert = BertModel(config)
        self.cls = BertOnlyNSPHead(config)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None,
                next_sentence_label=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        pooled_output = outputs[1]

        seq_relationship_score = self.cls(pooled_output)

        outputs = (seq_relationship_score,) + outputs[2:]  # add hidden states and attention if they are here
        if next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            outputs = (next_sentence_loss,) + outputs

        return outputs  # (next_sentence_loss), seq_relationship_score, (hidden_states), (attentions)


class BertForSequenceClassificationSVD(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForSequenceClassificationSVD, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None):

        outputs,layer_outputs_list,scc_layer_outputs_list = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        loss_layer = MSELoss()(layer_outputs_list[1], scc_layer_outputs_list[0])+MSELoss()(layer_outputs_list[3], scc_layer_outputs_list[1])+MSELoss()(layer_outputs_list[5], scc_layer_outputs_list[2])
        
        
        return outputs,loss_layer  # (loss), logits, (hidden_states), (attentions)


class BertForMultipleChoice(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForMultipleChoice, self).__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None):
        num_choices = input_ids.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        outputs = (reshaped_logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)


class BertForTokenClassification(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForTokenClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)


class BertForQuestionAnswering(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForQuestionAnswering, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None,
                start_positions=None, end_positions=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits,) + outputs[2:]
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)
