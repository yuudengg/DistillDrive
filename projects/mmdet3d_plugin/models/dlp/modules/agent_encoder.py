import os

import torch
import torch.nn as nn
import math

from .common_layers import Transformer, build_mlp
from projects.mmdet3d_plugin.models.attention import gen_sineembed_for_position
from mmcv.cnn.bricks.registry import ATTENTION, POSITIONAL_ENCODING
from mmcv.utils import build_from_cfg

@ATTENTION.register_module()
class AgentEncoder(nn.Module):
    def __init__(
        self,
        n_agent_cls,
        agent_channel,
        agent_dim,
        future_steps,
        pc_range=None,
        agent_frame=0,
        agent_pos=True,
        pos_norm=False,
        fourier_embed_tag=False,
        fourier_embed_cfg=None,
    ) -> None:
        super().__init__()

        self.agent_cls = n_agent_cls
        self.agent_dim = agent_dim
        self.agent_channel = agent_channel
        self.future_steps = future_steps
        self.agent_pos = agent_pos
        self.pos_norm = pos_norm
        self.agent_frame = agent_frame
        self.fourier_embed_tag = fourier_embed_tag
        self.fourier_embed_cfg = fourier_embed_cfg

        self.agent_proj = build_mlp(agent_channel, [self.agent_dim] * 2)
        self.agent_time_embed = PositionalEncoding(dim=self.agent_dim, max_len=self.future_steps)
        self.agent_feat_embed = Transformer(
            d_model = self.agent_dim,
            nhead = 8,
            num_encoder_layers = 4,
            num_decoder_layers = None,
            dim_feedforward = self.agent_dim * 4,
        )
        
        if self.agent_pos is True:
            if self.fourier_embed_tag:
                self.agent_pos_embed = build_from_cfg(fourier_embed_cfg, POSITIONAL_ENCODING)
            else:
                self.pc_range = nn.Parameter(torch.tensor(pc_range), requires_grad=False)
                self.agent_pos_embed = nn.Sequential(
                        nn.Linear(self.agent_dim, self.agent_dim * 2),
                        nn.ReLU(),
                        nn.Linear(self.agent_dim * 2, self.agent_dim),
                    )

    @staticmethod
    def to_vector(feat, valid_mask):
        vec_mask = valid_mask[..., :-1] & valid_mask[..., 1:]

        while len(vec_mask.shape) < len(feat.shape):
            vec_mask = vec_mask.unsqueeze(-1)
        return torch.where(
            vec_mask,
            feat[:, :, 1:, ...] - feat[:, :, :-1, ...],
            torch.zeros_like(feat[:, :, 1:, ...]),
        )

    def forward(self, bbox, traj, mask=None, padding_mask=None):
        '''
        Args:
            bbox: [B, N, 9]: x, y, z, l, w, h, yaw, vx, vy
            traj: [B, N, T, 2]
        Return:
            x_agent:
            x_pos:
            agent_valid_mask
        '''
        T = self.future_steps
        B, N, _ = bbox.shape
        # agent center and position
        center = bbox[:, :, :3]  # [B, N ,3]
        position = traj[:, :, :T, ] # [B, N, T, 2]
        # agent velocity
        velocity = position.clone() / 0.1 # [B, N, T, 2]
        # agent rotation
        relative_displacement = position.clone() # [B, N, T, 2]
        relative_displacement_norm = torch.norm(relative_displacement, dim=-1) + 1e-6 # add small epsilon
        heading_c = relative_displacement[:, :, :, 0] / relative_displacement_norm # [B, N, T]
        heading_s = relative_displacement[:, :, :, 1] / relative_displacement_norm # [B, N, T]
        # agent dimension
        dimension = bbox[:, :, 3:5] # [B, N ,2]
        dimension = [dimension for _ in range(T)]
        dimension = torch.stack(dimension,dim=2) # [B, N, T, 2]

        # find nan and zeros
        agent_time_mask = mask.clone() # [B, N, T]
        agent_time_mask_vec = mask.clone() # [B, N, T]
        agent_feature = torch.cat(
            [
                position[:, :, :, :2] * agent_time_mask[:, :, :T].unsqueeze(-1).repeat(1, 1, 1, 2).float(), # [B, N, T, 2]
                velocity[:, :, :, :2] * agent_time_mask[:, :, :T].unsqueeze(-1).repeat(1, 1, 1, 2).float(),  # [B, N, T, 2]
                torch.stack([heading_c, heading_s], dim=-1) * agent_time_mask[:, :, :T].unsqueeze(-1).repeat(1, 1, 1, 2).float(),  # [B, N, T, 2]
                dimension, # [B, N, T, 2]
                agent_time_mask_vec.float().unsqueeze(-1),  # [B, N, T, 1]
            ],
            dim=-1,
        )
        bs, n_agent, n_time, n_feat = agent_feature.shape # [B, N, T, 9]
        agent_valid_mask = agent_time_mask.any(-1)  # [B, N, T] --> [B, N], to find valid agent
        # [B, N, T, 9] --> [N_Valid, T, 9]
        agent_hs = self.agent_proj(agent_feature[agent_valid_mask])   # [N_Valid, T, D]
        agent_hs = self.agent_time_embed(agent_hs) # [N_Valid, T, D], for time position embeding
        key_padding_mask = agent_time_mask_vec[agent_valid_mask] # [N_Valid, T]
        N_valid, _ = key_padding_mask.shape
        if N_valid == 0:
            return (
                torch.zeros(bs, n_agent, self.agent_dim, device=agent_feature.device),
                torch.zeros(bs, n_agent, self.agent_dim, device=agent_feature.device),
                ~agent_valid_mask,
            )
        
        agent_hs = self.agent_feat_embed(
            src=agent_hs, 
            tgt=None, 
            src_padding_mask=~key_padding_mask,
            tgt_padding_mask=None,
        )  # [N_Valid, T, D]
        if self.agent_pos is True:
            
            agent_pos = torch.zeros(N_valid, 3).to(agent_hs.device) # [N_Valid, 3]
            agent_pos[:, :] = center[agent_valid_mask][:, :]
            if self.pos_norm:
                agent_pos = (agent_pos[:,:3] - self.pc_range[:3]) / (self.pc_range[3:6] - self.pc_range[0:3]) # [N_Valid, 3]

            if self.fourier_embed_tag:
                # use the height dimension
                agent_embed = self.agent_pos_embed(agent_pos[:, :])
            else:
                # only care the BEV loaction
                agent_embed = self.agent_pos_embed(gen_sineembed_for_position(agent_pos[:, :2])) # [N_Valid, C]
        else:
            agent_embed = torch.zeros(N_valid, self.agent_dim).to(agent_hs.device)

        device = agent_feature.device
        
        x_agent = torch.zeros(bs, n_agent, self.agent_dim, device=device)
        x_agent[agent_valid_mask] = agent_hs[:,self.agent_frame,:]

        x_pos = torch.zeros(bs, n_agent, self.agent_dim, device=device)
        x_pos[agent_valid_mask] = agent_embed
        return x_agent, x_pos, ~agent_valid_mask

    

class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=20):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x
