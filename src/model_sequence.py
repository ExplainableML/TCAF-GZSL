#!/usr/bin/python3
# -*- coding: utf-8 -*-

# system, numpy
import os
import numpy as np
from einops import rearrange, repeat
# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# user defined
from src.optimizer import SAM


def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, nn.BatchNorm1d):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)


def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, nn.BatchNorm1d) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, num_cls_token=0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
        self.attend = nn.Softmax(dim=-1)

        self.num_cls_token = num_cls_token

    def forward(self, x, context=None, mask=None):
        if context is None:
            context = x
        q, k, v = self.to_q(context), self.to_k(x), self.to_v(x)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (q, k, v)
        )

        dots = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (self.num_cls_token, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], "mask has incorrect dimensions"
            mask = mask[:, None, :] * mask[:, :, None]
            mask = mask.unsqueeze(1).repeat(1, self.heads, 1, 1)
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = self.attend(dots)

        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return out, attn


class NewAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, num_cls_token=0, use_cross_attention=True, use_self_attention=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
        self.attend = nn.Softmax(dim=-1)
        self.use_cross_attention=use_cross_attention
        self.use_self_attention=use_self_attention

        self.num_cls_token = num_cls_token

    def forward(self, x, seq_len_audio, context=None, mask=None):
        if context is None:
            context = x
        q, k, v = self.to_q(context), self.to_k(x), self.to_v(x)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (q, k, v)
        )

        dots = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            if self.use_self_attention==True and self.use_cross_attention==False:
                mask_initial = F.pad(mask.flatten(1), (self.num_cls_token, 0), value=True)
                mask_audio_truncated=torch.clone(mask_initial[:,1:])
                mask_video_truncated=torch.clone(mask_initial[:,1:])
                mask_audio_truncated[:,seq_len_audio:]=False
                mask_video_truncated[:,:seq_len_audio]=False
                multiplied_mask_audio=mask_audio_truncated[:,None,:]*mask_audio_truncated[:,:,None]
                multiplied_mask_video=mask_video_truncated[:,None,:]*mask_video_truncated[:,:,None]
                summed_masks=multiplied_mask_audio+multiplied_mask_video
                mask = torch.zeros((summed_masks.shape[0], summed_masks.shape[1]+1, summed_masks.shape[2]+1), dtype=torch.bool).cuda()
                mask[:,0,0]=True
                mask[:,0,:]=mask_initial
                mask[:,:,0]=mask_initial
                mask[:,1:,1:]=summed_masks
                mask = mask.unsqueeze(1).repeat(1, self.heads, 1, 1)
                dots.masked_fill_(~mask, mask_value)

            elif self.use_self_attention==False and self.use_cross_attention==True:
                mask = F.pad(mask.flatten(1), (self.num_cls_token, 0), value=True)
                mask = mask[:, None, :] * mask[:, :, None]

                mask[:, 1:seq_len_audio+1,1:seq_len_audio+1]=False
                mask[:,seq_len_audio+1:,seq_len_audio+1:]=False

                mask = mask.unsqueeze(1).repeat(1, self.heads, 1, 1)
                dots.masked_fill_(~mask, mask_value)
                del mask

            elif self.use_self_attention==True and self.use_cross_attention==True:
                mask = F.pad(mask.flatten(1), (self.num_cls_token, 0), value=True)
                assert mask.shape[-1] == dots.shape[-1], "mask has incorrect dimensions"
                mask = mask[:, None, :] * mask[:, :, None]
                mask = mask.unsqueeze(1).repeat(1, self.heads, 1, 1)
                dots.masked_fill_(~mask, mask_value)
                del mask


        attn = self.attend(dots)

        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return out, attn



class TransformerLayer(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        dim_head,
        mlp_dim,
        dropout,
        num_cls_token,
        attention_class=Attention,
        attention_args={},
    ):
        super().__init__()
        self.attn = attention_class(
            dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            num_cls_token=num_cls_token,
            **attention_args,
        )
        self.ff = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, t_audio, mask=None, query=None):
        if query is not None:
            query = self.norm(query)
        x_, attn = self.attn(self.norm(x), t_audio, context=query, mask=mask)
        x = x_ + x
        x_ = self.ff(self.norm(x))
        x = x_ + x
        return x, attn


class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout,
        num_cls_token,
        attention_class=Attention,
        attention_args={},
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                TransformerLayer(
                    dim,
                    heads,
                    dim_head,
                    mlp_dim,
                    dropout,
                    num_cls_token,
                    attention_class,
                    attention_args,
                )
            )

    def forward(self, x, t_audio, mask=None, query=None, return_attn=False):
        attn_outputs = []
        for layer in self.layers:
            x, attn = layer(x, t_audio, mask, query)
            attn_outputs.append(attn)
        if return_attn:
            return x, attn_outputs
        else:
            return x


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class AudioVisualEmbedding(nn.Module):
    def __init__(
        self,
        audio_feat_dim,
        video_feat_dim,
        time_len,
        embed_dim=128,
        out_dim=2048,
        embed_modality=True,
        num_modal=2,
        use_spatial=True,
        spatial_dim=7,
        dropout=0.1,
        audio_only=False,
        video_only=False,
        embed_cls_token_modality=False,
        embed_cls_token_time=False,
        time_embed_type="fixed",
        fourier_scale=10.0,
        embed_augment_position=False,
    ):
        super().__init__()
        self.use_spatial = use_spatial
        self.num_modal = num_modal
        self.time_len = time_len
        self.embed_dim = embed_dim
        self.spatial_dim = spatial_dim
        self.audio_only = audio_only
        self.video_only = video_only
        self.embed_cls_token_modality = embed_cls_token_modality
        self.embed_cls_token_time = embed_cls_token_time
        self.time_embed_type = time_embed_type
        self.embed_augment_position = embed_augment_position

        self.modal_embedding = None

        if embed_modality:
            if not (self.audio_only or self.video_only):
                # embedding for modality
                self.modal_embedding = nn.Embedding(num_modal, embed_dim)

        # embedding for time
        if self.time_embed_type == "fixed":
            self.time_embedding = nn.Embedding(time_len, embed_dim)
        elif self.time_embed_type == "sinusoid":
            # sinusoidal embeddings
            self.fourier_embeddings = nn.Linear(embed_dim, embed_dim)
            # positions = torch.tensor(
            #     np.linspace(0, 1, time_len, endpoint=False, dtype=np.float32)
            # ).unsqueeze(-1)
            # positions = torch.cat((positions, positions), dim=-1)

            randn = fourier_scale * torch.randn((2, embed_dim // 2))
            # position_proj = (2.0 * np.pi * positions) @ randn
            # position_proj = torch.cat(
            #     [torch.sin(position_proj), torch.cos(position_proj)], dim=-1
            # )
            self.register_buffer("fourier_randn", randn)
            # self.register_buffer("fourier_pos", position_proj)
        elif self.time_embed_type == 'none':
            pass
        else:
            raise NotImplementedError

        if use_spatial and not self.audio_only:
            # spatial 2D positional embedding
            self.pe_row = nn.Embedding(spatial_dim, embed_dim)
            self.pe_col = nn.Embedding(spatial_dim, embed_dim)

        # embedding for padding
        self.pad_embedding = nn.Embedding(1, embed_dim)

        self.dropout = nn.Dropout(dropout)

        if not self.video_only:
            self.out_proj_audio = nn.Conv1d(
                audio_feat_dim + embed_dim, out_dim, kernel_size=1
            )
        if not self.audio_only:
            self.out_proj_video = nn.Conv1d(
                video_feat_dim + embed_dim, out_dim, kernel_size=1
            )

        if self.embed_cls_token_time or embed_cls_token_modality:
            self.out_proj_cls_token = nn.Conv1d(
                out_dim + embed_dim, out_dim, kernel_size=1
            )

    def get_fourier_embeddings(self, time_tensor):
        positions = time_tensor.unsqueeze(-1)
        positions = torch.cat((positions, positions), dim=-1)
        position_proj = (2.0 * np.pi * positions) @ self.fourier_randn
        position_proj = torch.cat(
            [torch.sin(position_proj), torch.cos(position_proj)], dim=-1
        )
        fourier_embed = self.fourier_embeddings(position_proj)
        return fourier_embed

    def forward(
        self, audio_feat, video_feat, cls_token=None, mask_audio=None, mask_video=None, time_audio=None, time_video=None
    ):
        bs = audio_feat.size(0) if audio_feat is not None else video_feat.size(0)
        device = audio_feat.device if audio_feat is not None else video_feat.device
        attn_mask = None
        attn_mask_audio = None
        attn_mask_video = None

        if self.modal_embedding is not None:
            modal_tensor = (
                torch.arange(0, self.num_modal, device=device)
                .unsqueeze(0)
                .repeat(bs, 1)
            )
            modal_embeddings = self.modal_embedding(
                modal_tensor
            )  # [bs, num_modal, embed_dim]

        time_embed_start_time = 0
        if self.embed_augment_position and self.training:
            assert audio_feat is not None and video_feat is not None
            if audio_feat.size(1) < self.time_len:
                time_embed_start_time = torch.randint(
                    low=0, high=self.time_len - audio_feat.size(1), size=(1,)
                )

        if audio_feat is not None:
            assert audio_feat.dim() == 3
            # assert audio_feat.size(1) == self.time_len

            audio_time_len = audio_feat.size(1)

            if self.time_embed_type == "fixed":
                if time_audio is not None:
                    audio_time_tensor = torch.tensor(time_audio.round(), device=device, dtype=int)
                else:
                    audio_time_tensor = (
                        torch.arange(
                            time_embed_start_time,
                            audio_time_len + time_embed_start_time,
                            device=device,
                        )
                        .unsqueeze(0)
                        .repeat(bs, 1)
                    )
                audio_time_embeddings = self.time_embedding(
                    audio_time_tensor
                )  # [bs, time_len, embed_dim]
            elif self.time_embed_type == "sinusoid":
                if time_audio is not None:
                    # use embeddings according to given time steps
                    # audio_time_embeddings = self.fourier_embeddings(self.fourier_pos[time_audio])
                    audio_time_embeddings = self.get_fourier_embeddings(time_audio)
                else:
                    # use embeddings according to sequence
                    audio_time_tensor = torch.arange(
                        time_embed_start_time,
                        audio_time_len + time_embed_start_time,
                        device=device,
                    )
                    audio_time_embeddings = (
                        self.get_fourier_embeddings(audio_time_tensor).unsqueeze(0).repeat(bs, 1, 1)
                    )
            else:
                # no time embeddings
                audio_time_embeddings = torch.zeros((bs, audio_time_len, self.embed_dim), device=device, dtype=torch.float32)

            # audio padding embeddings
            audio_padding_embeddings = torch.zeros_like(audio_time_embeddings)
            audio_padding_embeddings[mask_audio] = self.pad_embedding.weight[0]

            # audio embedding
            if self.modal_embedding is None:
                audio_modal_embedding = torch.zeros((bs, audio_time_len, self.embed_dim), device=device, dtype=torch.float32)
            else:
                audio_modal_embedding = modal_embeddings[:, 0, :].unsqueeze(1).repeat(1, audio_time_len, 1)

            audio_embedding = (
                audio_modal_embedding
                + audio_time_embeddings
                + audio_padding_embeddings
            )

            # concat audio features with audio embeddings
            audio_feat = torch.cat([audio_feat, audio_embedding], dim=-1)

            # create audio attention mask (attend only to non-padded tokens)
            audio_attn_mask = ~mask_audio
            if attn_mask is None:
                attn_mask_audio = audio_attn_mask
            else:
                attn_mask_audio = torch.cat([attn_mask, audio_attn_mask], dim=-1)

            # out projection audio
            audio_feat = audio_feat.transpose(1, 2)
            audio_feat = self.out_proj_audio(audio_feat)
            audio_feat = self.dropout(audio_feat)

        if video_feat is not None:
            if self.use_spatial:
                assert video_feat.dim() == 5
            else:
                assert video_feat.dim() == 3
            # assert video_feat.size(1) == self.time_len

            video_time_len = video_feat.size(1)

            if self.time_embed_type == "fixed":
                if time_video is not None:
                    video_time_tensor = torch.tensor(time_video.round(), device=device, dtype=int)
                else:
                    video_time_tensor = (
                        torch.arange(
                            time_embed_start_time,
                            video_time_len + time_embed_start_time,
                            device=device,
                        )
                        .unsqueeze(0)
                        .repeat(bs, 1)
                    )
                video_time_embeddings = self.time_embedding(
                    video_time_tensor
                )  # [bs, time_len, embed_dim]
            elif self.time_embed_type == "sinusoid":
                if time_video is not None:
                    # use embeddings according to given time steps
                    video_time_embeddings = self.get_fourier_embeddings(time_video)
                else:
                    video_time_tensor = torch.arange(
                        time_embed_start_time,
                        video_time_len + time_embed_start_time,
                        device=device,
                    )
                    video_time_embeddings = (
                        self.get_fourier_embeddings(video_time_tensor).unsqueeze(0).repeat(bs, 1, 1)
                    )
            else:
                # no time embeddings
                video_time_embeddings = torch.zeros((bs, video_time_len, self.embed_dim), device=device, dtype=torch.float32)

            # video padding embeddings
            video_padding_embeddings = torch.zeros_like(video_time_embeddings)
            video_padding_embeddings[mask_video] = self.pad_embedding.weight[0]

            # video embedding
            if self.modal_embedding is None:
                video_modal_embedding = torch.zeros((bs, video_time_len, self.embed_dim), device=device, dtype=torch.float32)
            else:
                video_modal_embedding = modal_embeddings[:, 1, :].unsqueeze(1).repeat(1, video_time_len, 1)

            video_embedding = (
                video_modal_embedding
                + video_time_embeddings
                + video_padding_embeddings
            )

            if self.use_spatial:
                position_embeddings = self.pe_row.weight.unsqueeze(
                    1
                ) + self.pe_col.weight.unsqueeze(0)
                position_embeddings = position_embeddings.unsqueeze(0).repeat(
                    bs, 1, 1, 1
                )  # [bs, row, col, embed_dim]
                position_embeddings = position_embeddings.unsqueeze(1).repeat(
                    1, video_time_len, 1, 1, 1
                )  # [bs, time_len, row, col, embed_dim]

                video_embedding = (
                    video_embedding.unsqueeze(2)
                    .unsqueeze(2)
                    .repeat(1, 1, self.spatial_dim, self.spatial_dim, 1)
                )
                video_embedding = video_embedding + position_embeddings

            # concat video features with video embedding
            video_feat = torch.cat([video_feat, video_embedding], dim=-1)

            # create video attention mask (attend only to non-padded tokens)
            video_attn_mask = ~mask_video
            if self.use_spatial:
                video_attn_mask = video_attn_mask.repeat_interleave(
                    self.spatial_dim * self.spatial_dim, dim=1
                )
            if attn_mask is None:
                attn_mask_video = video_attn_mask
            else:
                attn_mask_video = torch.cat([attn_mask, video_attn_mask], dim=-1)

            if self.use_spatial:
                # flatten spatially
                video_feat = torch.flatten(
                    video_feat, start_dim=1, end_dim=3
                )  # [bs, time_len*row*col, embed_dim]

            # out projection video
            video_feat = video_feat.transpose(1, 2)
            video_feat = self.out_proj_video(video_feat)
            video_feat = self.dropout(video_feat)


        if video_feat is not None:
            video_feat = video_feat.transpose(1, 2).contiguous()
        if audio_feat is not None:
            audio_feat = audio_feat.transpose(1, 2).contiguous()


        cls_token_embedding = None
        # if class token is given, add time embedding (for localisation)
        if self.embed_cls_token_time and cls_token is not None:
            cls_token_time_len = cls_token.size(1)
            assert cls_token_time_len == self.time_len
            cls_token_time_tensor = (
                torch.arange(0, cls_token_time_len, device=device)
                .unsqueeze(0)
                .repeat(bs, 1)
            )
            cls_token_time_embeddings = self.time_embedding(
                cls_token_time_tensor
            )  # [bs, time_len, embed_dim]
            cls_token_embedding = cls_token_time_embeddings
        if self.embed_cls_token_modality and cls_token is not None:
            # assume that we can just devide cls token in two equal lengths: first audio, second visual
            num_cls_token = cls_token.size(1)
            num_cls_token_modal = int(num_cls_token / 2)

            cls_token_audio_modal_embedding = (
                modal_embeddings[:, 0, :].unsqueeze(1).repeat(1, num_cls_token_modal, 1)
            )
            cls_token_video_modal_embedding = (
                modal_embeddings[:, 1, :].unsqueeze(1).repeat(1, num_cls_token_modal, 1)
            )
            cls_token_modal_embedding = torch.cat(
                [cls_token_audio_modal_embedding, cls_token_video_modal_embedding],
                dim=1,
            )
            if cls_token_embedding is None:
                cls_token_embedding = cls_token_modal_embedding
            else:
                cls_token_embedding += cls_token_modal_embedding

        if cls_token_embedding is not None:
            cls_token = torch.cat([cls_token, cls_token_embedding], dim=-1)
            cls_token = self.out_proj_cls_token(cls_token.transpose(1, 2))
            cls_token = cls_token.transpose(1, 2).contiguous()

        return audio_feat, video_feat, attn_mask_audio, attn_mask_video, cls_token



class EmbeddingNet(nn.Module):
    def __init__(self, input_size, output_size, dropout, use_bn, hidden_size=-1):
        super(EmbeddingNet, self).__init__()
        modules = []

        if hidden_size > 0:
            modules.append(nn.Linear(in_features=input_size, out_features=hidden_size))
            if use_bn:
                modules.append(nn.BatchNorm1d(num_features=hidden_size))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(dropout))
            modules.append(nn.Linear(in_features=hidden_size, out_features=output_size))
            modules.append(nn.BatchNorm1d(num_features=output_size))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(dropout))
        else:
            modules.append(nn.Linear(in_features=input_size, out_features=output_size))
            modules.append(nn.BatchNorm1d(num_features=output_size))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(dropout))
        self.fc = nn.Sequential(*modules)

    def forward(self, x):
        output = self.fc(x)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class Multimodal_Sequence_Transformer(nn.Module):
    def __init__(self, params_model, input_size_audio, input_size_video):
        super(Multimodal_Sequence_Transformer, self).__init__()

        print('Initializing model variables...', end='')
        # Dimension of embedding
        self.dim_out = params_model['dim_out']
        self.input_dim_audio = input_size_audio
        self.input_dim_video = input_size_video
        self.hidden_size_encoder=params_model['encoder_hidden_size']
        self.hidden_size_decoder=params_model['decoder_hidden_size']
        self.drop_enc=params_model['dropout_encoder']
        self.drop_proj_o=params_model['dropout_decoder']
        self.reg_loss=params_model['reg_loss']
        self.cross_entropy_loss=params_model['cross_entropy_loss']
        self.drop_proj_w=params_model['additional_dropout']
        self.use_self_attention=params_model['transformer_attention_use_self_attention']
        self.use_cross_attention=params_model['transformer_attention_use_cross_attention']

        self.rec_loss = params_model['rec_loss']

        self.lr_scheduler = params_model['lr_scheduler']

        print('Initializing trainable models...', end='')

        self.average_features = params_model['transformer_average_features']
        self.use_embedding_net = params_model['transformer_use_embedding_net']
        self.audio_only = params_model['audio_only']
        self.video_only = params_model['video_only']

        if self.audio_only or self.video_only:
            assert self.use_cross_attention and self.use_self_attention

        if self.use_embedding_net:
            self.A_enc = EmbeddingNet(
                input_size=input_size_audio,
                hidden_size=self.hidden_size_encoder,
                output_size=params_model['transformer_dim'],
                dropout=self.drop_enc,
                use_bn=params_model['embeddings_batch_norm']
            )
            self.V_enc = EmbeddingNet(
                input_size=input_size_video,
                hidden_size=self.hidden_size_encoder,
                output_size=params_model['transformer_dim'],
                dropout=self.drop_enc,
                use_bn=params_model['embeddings_batch_norm']
            )
            input_dim_transformer_embed_audio = params_model['transformer_dim']
            input_dim_transformer_embed_video = params_model['transformer_dim']
        else:
            input_dim_transformer_embed_audio = self.input_dim_audio
            input_dim_transformer_embed_video = self.input_dim_video


        word_embedding_dim = 300
        self.W_proj= EmbeddingNet(
            input_size=word_embedding_dim,
            output_size=self.dim_out,
            dropout=self.drop_proj_w,
            use_bn=params_model['embeddings_batch_norm']
        )

        self.D_w = EmbeddingNet(
            input_size=self.dim_out,
            output_size=word_embedding_dim,
            dropout=self.drop_proj_w,
            use_bn=params_model['embeddings_batch_norm']
        )

        self.use_class_token = params_model['transformer_use_class_token']
        if self.use_class_token:
            self.num_cls_token = 1
            self.cls_token = nn.Parameter(torch.randn(1, self.num_cls_token, params_model['transformer_dim']))
        else:
            self.num_cls_token = 0

        self.position_encoder_block = self.audiovisual_embeddings = AudioVisualEmbedding(
            audio_feat_dim=input_dim_transformer_embed_audio,
            video_feat_dim=input_dim_transformer_embed_video,
            time_len=params_model['transformer_embedding_time_len'],
            embed_dim=params_model['transformer_embedding_dim'],
            out_dim=params_model['transformer_dim'],
            embed_modality=params_model['transformer_embedding_modality'],
            num_modal=2,
            use_spatial=False,
            spatial_dim=None,
            dropout=params_model['transformer_embedding_dropout'],
            audio_only=False,
            video_only=False,
            embed_cls_token_modality=False,
            embed_cls_token_time=False,
            time_embed_type=params_model['transformer_embedding_time_embed_type'],
            fourier_scale=params_model['transformer_embedding_fourier_scale'],
            embed_augment_position=params_model['transformer_embedding_embed_augment_position']
        )

        # fill with command line arguments for controlling attention
        attention_dict = {
            "use_self_attention": self.use_self_attention, # params_model['transformer_attention_use_self_attention']
            "use_cross_attention": self.use_cross_attention # params_model['transformer_attention_use_cross_attention']
        }

        self.Audio_visual_transformer = Transformer(
                dim=params_model['transformer_dim'],
                depth=params_model['transformer_depth'],
                heads=params_model['transformer_heads'],
                dim_head=params_model['transformer_dim_head'],
                mlp_dim=params_model['transformer_mlp_dim'],
                dropout=params_model['transformer_dropout'],
                num_cls_token=self.num_cls_token,
                attention_class=NewAttention,
                attention_args=attention_dict,
        )
        self.O_proj = EmbeddingNet(
            input_size=params_model['transformer_dim'],
            hidden_size=self.hidden_size_decoder,
            output_size=self.dim_out,
            dropout=self.drop_proj_o,
            use_bn=params_model['embeddings_batch_norm']
        )
        self.D_o = EmbeddingNet(
            input_size=self.dim_out,
            hidden_size=self.hidden_size_decoder,
            output_size=params_model['transformer_dim'],
            dropout=self.drop_proj_o,
            use_bn=params_model['embeddings_batch_norm']
        )

        # Optimizers
        print('Defining optimizers...', end='')
        self.lr = params_model['lr']

        optimizer = params_model['optimizer']
        self.is_sam_optim = False
        if optimizer == 'adam':
            self.optimizer_gen = optim.Adam(
                self.parameters(),
                lr=self.lr, weight_decay=1e-5
            )
            if self.lr_scheduler:
                self.scheduler_learning_rate =  optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_gen, 'max', patience=3, verbose=True)

        elif optimizer == 'adam-sam':
            self.optimizer_gen = SAM(self.parameters(), optim.Adam, lr=self.lr, weight_decay=1e-5)
            self.is_sam_optim = True
            if self.lr_scheduler:
                # lr scheduling on base optimizer
                self.scheduler_learning_rate =  optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_gen.base_optimizer, 'max', patience=3, verbose=True)
        else:
            raise NotImplementedError

        print('Done')

        # Loss function
        print('Defining losses...', end='')
        self.criterion_cyc = nn.MSELoss()
        self.criterion_cls = nn.CrossEntropyLoss()
        self.MSE_loss = nn.MSELoss()
        print('Done')

    def optimize_scheduler(self, value):
        if self.lr_scheduler:
            self.scheduler_learning_rate.step(value)

    def forward(self, a, v, w, masks, timesteps):
        b, _, _ = a.shape
        device = a.device

        a = a / 255.
        v = F.normalize(v)


        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)


        # get padding masks (True = padded)
        pos_a = ~(masks['audio'].bool().to(device))
        pos_v = ~(masks['video'].bool().to(device))

        # get timesteps for positional embeddings
        pos_t_audio = torch.tensor(timesteps['audio'], dtype=torch.float32, device=device)
        pos_t_video = torch.tensor(timesteps['video'], dtype=torch.float32, device=device)


        # get sequence length
        t_audio = a.shape[1]
        t_video = v.shape[1]


        if self.use_embedding_net:
            a = a.view(b * t_audio, -1)
            v = v.view(b * t_video, -1)

            a = self.A_enc(a)
            v = self.V_enc(v)

            a = a.view(b, t_audio, -1)
            v = v.view(b, t_video, -1)


        position_aware_audio, position_aware_visual, attn_mask_audio, attn_mask_video, cls_tokens = self.position_encoder_block(
            a, v, cls_tokens, pos_a, pos_v, pos_t_audio, pos_t_video
        )

        if attn_mask_audio is not None or attn_mask_video is not None:
                attn_mask = torch.cat((attn_mask_audio, attn_mask_video), dim=1)


        multi_modal_input = torch.cat((cls_tokens, position_aware_audio, position_aware_visual), dim=1)


        multi_modal_input = self.Audio_visual_transformer(multi_modal_input, t_audio, attn_mask)


        c_o = multi_modal_input[:, 0]


        theta_o = self.O_proj(c_o)

        rho_o = self.D_o(theta_o)

        theta_w = self.W_proj(w)

        rho_w=self.D_w(theta_w)

        output = {
            "theta_w": theta_w,
            "w": w,
            "rho_w": rho_w,
            "theta_o": theta_o,
            "rho_o": rho_o,
        }

        return output


    def compute_loss(self, outputs, embeddings_crossentropy, gt_cross_entropy):

        theta_w = outputs['theta_w']

        w = outputs['w']
        rho_w = outputs['rho_w']

        theta_o = outputs['theta_o']

        rho_o = outputs['rho_o']


        device = theta_w.device

        if self.cross_entropy_loss==True:
            embedding_cross_entropy=self.W_proj(embeddings_crossentropy)
            Cross_loss=nn.CrossEntropyLoss()
            scores=torch.matmul(theta_o, embedding_cross_entropy.t())
            l_ce=Cross_loss(scores, gt_cross_entropy)
        else:
            l_ce = torch.tensor(0., device=device)

        if self.reg_loss==True:
            l_reg = (
                self.MSE_loss(theta_o, theta_w)
            )
        else:
            l_reg = torch.tensor(0., device=device)


        if self.rec_loss == True:
            l_rec = (
                    self.MSE_loss(w, rho_o) +
                    self.MSE_loss(w, rho_w)
            )
        else:
            l_rec = torch.tensor(0., device=device)


        loss_total = l_rec+l_reg+l_ce
        loss_dict = {
            "Loss/total_loss": loss_total.detach().cpu(),
            "Loss/loss_reg": l_reg.detach().cpu(),
            "Loss/loss_cmd_rec": l_rec.detach().cpu(),
            "Loss/cross_entropy": l_ce.detach().cpu()

        }
        return loss_total, loss_dict


    def optimize_params(self, audio, video, cls_numeric, cls_embedding, masks, timesteps, embedding_crossentropy, optimize=False):
        if not self.is_sam_optim:
            # Forward pass
            outputs = self.forward(audio, video, cls_embedding, masks, timesteps)

            # Backward pass
            loss_numeric, loss = self.compute_loss(outputs, embedding_crossentropy,  cls_numeric)

            if optimize == True:
                self.optimizer_gen.zero_grad()
                loss_numeric.backward()
                self.optimizer_gen.step()

        else:
            # SAM optimizer requires two forward / backward

            enable_running_stats(self)
            outputs = self.forward(audio, video, cls_embedding, masks, timesteps)
            loss_numeric, loss = self.compute_loss(outputs, embedding_crossentropy,  cls_numeric)

            if optimize:
                # first forward-backward step
                # self.optimizer_gen.zero_grad()
                loss_numeric.backward()
                self.optimizer_gen.first_step(zero_grad=True)

                # second forward-backward step
                disable_running_stats(self)
                outputs_second = self.forward(audio, video, cls_embedding, masks, timesteps)
                second_loss, _ = self.compute_loss(outputs_second, embedding_crossentropy,  cls_numeric)
                second_loss.backward()
                self.optimizer_gen.second_step(zero_grad=True)

        return loss_numeric, loss

    def get_embeddings(self, a, v, w, masks, timesteps):
        a = a / 255.
        v = F.normalize(v)

        b, _, _ = a.shape
        device = a.device


        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)

        # get padding masks (True = padded)
        pos_a = ~(masks['audio'].bool().to(device))
        pos_v = ~(masks['video'].bool().to(device))


        t_audio = a.shape[1]
        t_video = v.shape[1]

        if self.use_embedding_net:
            a = a.view(b * t_audio, -1)
            v = v.view(b * t_video, -1)

            a = self.A_enc(a)
            v = self.V_enc(v)

            a = a.view(b, t_audio, -1)
            v = v.view(b, t_video, -1)
        position_aware_audio, position_aware_visual, attn_mask_audio, attn_mask_video, cls_tokens = self.position_encoder_block(
            a, v, cls_tokens, pos_a, pos_v
        )

        if attn_mask_audio is not None or attn_mask_video is not None:
                attn_mask = torch.cat((attn_mask_audio, attn_mask_video), dim=1)

        multi_modal_input = torch.cat((cls_tokens, position_aware_audio, position_aware_visual), dim=1)

        multi_modal_input = self.Audio_visual_transformer(multi_modal_input, t_audio, attn_mask)

        c_o = multi_modal_input[:, 0]

        # cls token to theta
        theta_o = self.O_proj(c_o)

        theta_w=self.W_proj(w)

        return theta_o, theta_o, theta_w

