import os
import os.path as osp

import contextlib

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import transformers
from deepspeed.pipe import PipelineModule, TiedLayerSpec, LayerSpec
from .minigpt4qwen import Minigpt4Qwen
import importlib


class Qwen2RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim)) 
        if importlib.util.find_spec("einops") is None:
            raise RuntimeError("einops is required for Rotary Embedding")

        self._rotary_pos_emb_cache = None
        self._seq_len_cached = 0
        self._ntk_alpha_cached = 1.0

    def update_rotary_pos_emb_cache(self, max_seq_len, offset=0, ntk_alpha=1.0):
        seqlen = max_seq_len + offset
        if seqlen > self._seq_len_cached or ntk_alpha != self._ntk_alpha_cached:
            # 动态的调整base值，需要后续理解三角函数内插和外插很数学的东西。
            base = self.base * ntk_alpha ** (self.dim / (self.dim - 2))
            self.inv_freq = 1.0 / (
                base
                ** (
                    torch.arange(0, self.dim, 2, device=self.inv_freq.device).float()
                    / self.dim
                )
            )    
            # inv_freq 就是那个旋转的角度 theta, 
            # 采用原始transformer论文里的 base 值，原论文是加性运算。而 RoPE 是乘性运算。
            # 两者都能实现相对位置编码的功能。
            self._seq_len_cached = seqlen
            self._ntk_alpha_cached = ntk_alpha
            seq = torch.arange(seqlen, device=self.inv_freq.device)
            freqs = torch.outer(seq.type_as(self.inv_freq), self.inv_freq)   # 计算 m * theta 
            # Different from paper, but it uses a different permutation in order to obtain the same calculation  
            # 理解半天：原来是 dim 向量分量 中的 element 发生交换 
            emb = torch.cat((freqs, freqs), dim=-1) 
            from einops import rearrange
            self._rotary_pos_emb_cache = rearrange(emb, "n d -> 1 n 1 d")

    def forward(self, max_seq_len, offset=0, ntk_alpha=1.0):
        self.update_rotary_pos_emb_cache(max_seq_len, offset, ntk_alpha)
        return self._rotary_pos_emb_cache[:, offset : offset + max_seq_len] 


class RotaryEmbedding2(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        
        self._cos_cached = None
        self._sin_cached = None
        self._seq_len_cached = 0
        self._ntk_alpha_cached = 1.0

    def update_rotary_pos_emb_cache(self, max_seq_len, offset=0, ntk_alpha=1.0):
        seqlen = max_seq_len + offset
        if seqlen > self._seq_len_cached or ntk_alpha != self._ntk_alpha_cached:
            base = self.base * ntk_alpha ** (self.dim / (self.dim - 2))
            self.inv_freq = 1.0 / (
                base ** (torch.arange(0, self.dim, 2, device=self.inv_freq.device).float() / self.dim)
            )
            self._seq_len_cached = seqlen
            self._ntk_alpha_cached = ntk_alpha
            
            seq = torch.arange(seqlen, device=self.inv_freq.device)
            freqs = torch.outer(seq.type_as(self.inv_freq), self.inv_freq)
            
            # 计算余弦和正弦值并缓存
            emb = torch.cat((freqs, freqs), dim=-1)
            from einops import rearrange
            emb = rearrange(emb, "n d -> 1 n 1 d")
            self._cos_cached = emb.cos()
            self._sin_cached = emb.sin()

    def forward(self, max_seq_len, offset=0, ntk_alpha=1.0):
        self.update_rotary_pos_emb_cache(max_seq_len, offset, ntk_alpha)
        return (
            self._cos_cached[:, offset : offset + max_seq_len],
            self._sin_cached[:, offset : offset + max_seq_len],
        )


def enable_input_require_grads(module):
    def make_inputs_require_grads(module,input,output):
        output.requires_grad_(True)

    module.register_forward_hook(make_inputs_require_grads)

class VisionPipe(nn.Module):
    def __init__(self, model: Minigpt4Qwen):
        super().__init__()
        self.visual_encoder = model.visual_encoder
        self.ln_vision = model.ln_vision
        self.Qformer, self.query_tokens = model.Qformer, model.query_tokens

        self.maybe_autocast = model.maybe_autocast()
        self.enable_autocast = model.enable_autocast

        self.llm_proj = model.llm_proj

    def forward(self,ipt):
        image = ipt
        with (self.maybe_autocast if self.enable_autocast else contextlib.nullcontext()):
            image_embeds = self.visual_encoder(image)
            image_embeds = self.ln_vision(image_embeds)

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)

        bs = image.size(0)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        inputs_llm = self.llm_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])

        return inputs_llm

class EmbeddingPipeLayer(nn.Module):
    def __init__(self, model: Minigpt4Qwen):
        super().__init__()
        #  Qwen中的model.llm_model.transformer.wte 对应 Qwen2的model.llm_model.base_model.embed_tokens
        self.word_embeddings = model.llm_model.base_model.embed_tokens
        # self.word_embeddings = model.llm_model.transformer.wte
        # enable_input_require_grads(self.word_embeddings)

    def forward(self, ipt):
        llm_tokens = ipt.long()
        return self.word_embeddings(llm_tokens)

class TokenizerPipeLayer(nn.Module):
    def __init__(self, model:Minigpt4Qwen):
        super().__init__()
        self.replace_image_token_id = model.replace_image_token_id
        # self.replace_image_token_id = 151646
        self.visionpipe = VisionPipe(model)
        self.wtepipe = EmbeddingPipeLayer(model)
        # 在Qwen2中应该没有使用 Dropout
        # self.drop = model.llm_model.transformer.drop
        # self.drop = model.llm_model.base_model.drop

        # self.config = model.llm_model.transformer.config
        self.config = model.llm_model.base_model.config
        # self.use_dynamic_ntk = model.llm_model.transformer.use_dynamic_ntk
        self.use_dynamic_ntk = False # 可能是这个 Qwen里面对应的是False
        self.llm_training = model.llm_model.base_model.training

        # rope + ntk
        # self.rotary_emb = model.llm_model.base_model.rotary_emb
        self.rotary_emb = RotaryEmbedding2(
            dim=128
        )

        # rotary_emb 对应到Qwen2是 model.llm_model.base_model.layers[0].self_attn.rotary_emb
        # a = model.llm_model.base_model.layers[0].self_attn
        # all_layers = model.llm_model.base_model.layers
        # for i, layer in enumerate(all_layers):
        #     print(f"Encoder Layer {i} structure:")
        #     print(layer)
        # rotary_emb_layers = [layer.attention.self.rotary_emb for layer in all_layers if hasattr(layer.attention.self, 'rotary_emb')]
        # self.rotary_emb = nn.Sequential(*list(model.llm_model.base_model))
        # print("test")

        # rope+ntk related func
        # 尝试
        # self.get_ntk_alpha = model.llm_model.base_model.get_ntk_alpha
        # self.get_ntk_alpha = model.llm_model.model.get_ntk_alpha
        # self.get_head_mask = model.llm_model.transformer.get_head_mask

    def forward(self,ipt):
        image, llm_tokens, targets, attention_mask = ipt
        inputs_llm = self.visionpipe(image)

        device = inputs_llm.device

        # llm_tokens = torch.where(llm_tokens==151646, 27, llm_tokens)
        replace_image_idxs = torch.where(llm_tokens == self.replace_image_token_id)
        inputs_embeds = self.wtepipe(llm_tokens) # B, L, C
        _,_,channels = inputs_embeds.shape

        inputs_embeds = inputs_embeds.clone()
        # 不加这个
        inputs_embeds[replace_image_idxs[0],replace_image_idxs[1]] = inputs_llm.view(-1,channels).to(inputs_embeds.dtype)

        # rope + ntk
        # get rotary_pos_emb_list
        input_shape = inputs_embeds.size()[:-1]
        position_ids = torch.arange(
                0,
                input_shape[-1],
                dtype=torch.long,
                device=device,
            )
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        kv_seq_len = inputs_embeds.size()[1]
        if self.llm_training or not self.use_dynamic_ntk:
            ntk_alpha_list = [1.0]
        # else:
        #     ntk_alpha_list = []
        #     ntk_alpha = self.get_ntk_alpha(kv_seq_len)
        #     ntk_alpha_list.append(ntk_alpha)
        # 注释掉了
        # self.rotary_emb._ntk_alpha_cached_list = ntk_alpha_list
        # ntk_alpha = ntk_alpha_list[0]
        # 这里去掉了一个参数
        rotary_pos_emb_list = self.rotary_emb(kv_seq_len)
        # rotary_pos_emb_list = self.rotary_emb(kv_seq_len, ntk_alpha=ntk_alpha)
        rotary_pos_emb_list = torch.stack(rotary_pos_emb_list,dim=0)
        rotary_pos_emb_list = rotary_pos_emb_list.to("cuda")
        # print(rotary_pos_emb_list);exit(0)

        # 在Qwen2中应该没有使用 Dropout
        # inputs_embeds = self.drop(inputs_embeds)
        inputs_embeds = inputs_embeds
        output_shape = input_shape + (inputs_embeds.size(-1),)
        output_shape = torch.tensor(output_shape,device="cuda")

        batch_size = inputs_embeds.shape[0]
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=self.wtepipe.word_embeddings.weight.dtype)
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.wtepipe.word_embeddings.weight.dtype).min

        rotary_pos_emb_list.requires_grad_(True)
        attention_mask.requires_grad_(True)

        return inputs_embeds, attention_mask, targets, rotary_pos_emb_list, position_ids, output_shape

class QwenBlockPipeLayer(torch.nn.Module):
    def __init__(self, model: Minigpt4Qwen, layer_idx, llm_grad_ckpt):
        super().__init__()
        # Qwen中的model.llm_model.transformer.h 对应 Qwen2的model.llm_model.base_model.layers
        # self.layer = model.llm_model.transformer.h[layer_idx]
        self.layer = model.llm_model.base_model.layers[layer_idx]
        
        self.layer_idx = layer_idx
        # self.llm_grad_ckpt = llm_grad_ckpt
        self.llm_grad_ckpt = False

    def forward(self, ipt):
        inputs_embeds, attention_mask, targets, rotary_pos_emb_list, position_ids, output_shape = ipt
        # print("grad: ", inputs_embeds.requires_grad)

        # support grad-ckpt
        if self.llm_grad_ckpt:
            expanded_attention_mask = attention_mask.expand(-1, -1, 1536, -1)
            expanded_attention_mask = expanded_attention_mask.to("cuda")
            inputs_embeds = checkpoint(
                self.layer,
                inputs_embeds,
                [[rotary_pos_emb_list[0],rotary_pos_emb_list[1]]],
                None,
                expanded_attention_mask,
                None,
            )[0]
        else:
            # 扩展维度：如果 seq_len 为 1536，你可能需要将这个掩码扩展到 [1, 1, 1536, 1536]，表示对所有的序列元素进行掩码。Qwen2要求的attention_mask是[1, 1, 1536, 1536]
            expanded_attention_mask = attention_mask.expand(-1, -1, 1536, -1)
            expanded_attention_mask = expanded_attention_mask.to("cuda")
            expanded_attention_mask.requires_grad_(True)
            inputs_embeds = self.layer(
                inputs_embeds,
                rotary_pos_emb_list=[[rotary_pos_emb_list[0],rotary_pos_emb_list[1]]],
                attention_mask=expanded_attention_mask,
                head_mask=None,
            )[0]
        return inputs_embeds, attention_mask, targets, rotary_pos_emb_list, position_ids, output_shape


class FLNPipeLayer(torch.nn.Module):
    def __init__(self, model: Minigpt4Qwen):
        super().__init__()
        # Qwen中的model.llm_model.transformer.ln_f 对应 Qwen2的model.llm_model.base_model.norm
        # self.final_layernorm = model.llm_model.transformer.ln_f
        self.final_layernorm = model.llm_model.base_model.norm

    def forward(self, ipt):
        inputs_embeds, attention_mask, targets, rotary_pos_emb_list, position_ids, output_shape = ipt
        inputs_embeds = self.final_layernorm(inputs_embeds)
        inputs_embeds = inputs_embeds.view(list(output_shape)).contiguous()
        # print(inputs_embeds)
        return inputs_embeds, targets


class LMPipeLayer(torch.nn.Module):
    def __init__(self, model: Minigpt4Qwen):
        super().__init__()
        self.lm_head = model.llm_model.lm_head

    def forward(self, ipt):
        hidden_states, labels = ipt
        logits = self.lm_head(hidden_states)
        # print(logits)
        return logits, labels

class LossPipeLayer(torch.nn.Module):
    def __init__(self, model: Minigpt4Qwen):
        super().__init__()
        self.freeze_llm = model.freeze_llm

    def forward(self, ipt):
        logits, labels = ipt
        # print(logits.size());print(labels.size());exit(0)

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        bs = shift_labels.size(0)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
        # print(loss)
        return (loss, torch.tensor(bs)) if self.freeze_llm else loss

class IndentityPipeLayerLast(nn.Module):
    def __init__(self, model: Minigpt4Qwen):
        super().__init__()
        self.occupy = nn.Linear(1000,1000,bias=False)
        nn.init.constant_(self.occupy.weight,0.)
    
    def forward(self,ipt):
        loss, bs = ipt
        # zero_in = torch.zeros((bs,self.occupy.in_features),device='cuda')
        # return loss + 0. * self.occupy(zero_in).sum()
        return loss

class IndentityPipeLayer(nn.Module):
    def __init__(self, model: Minigpt4Qwen):
        super().__init__()
    
    def forward(self,ipt):
        inputs_embeds, attention_mask, targets, rotary_pos_emb_list, position_ids, output_shape = ipt
        return inputs_embeds, attention_mask, targets, rotary_pos_emb_list, position_ids, output_shape

def get_model(model, freeze_llm, llm_grad_ckpt):
    layers = [LayerSpec(TokenizerPipeLayer,model=model),
            *[LayerSpec(IndentityPipeLayer,model=model) for _ in range(4)], # 调节控制多卡的显存分配
            *[LayerSpec(QwenBlockPipeLayer, model=model, layer_idx=idx, llm_grad_ckpt=llm_grad_ckpt) for idx in
                # Qwen中的model.llm_model.transformer 对应 Qwen2的model.llm_model.base_model
                range(model.llm_model.base_model.config.num_hidden_layers)],
                # range(model.llm_model.transformer.config.num_hidden_layers)],
            LayerSpec(FLNPipeLayer, model=model),
            LayerSpec(LMPipeLayer, model=model),
            LayerSpec(LossPipeLayer, model=model),
        ]
    if freeze_llm:
        layers.append(LayerSpec(IndentityPipeLayerLast,model=model))
    return layers