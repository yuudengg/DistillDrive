import torch
import torch.nn as nn

# scene_type ids (defined in packet.py): 0 ego, 1 agent, 2 map


class SceneProjector(nn.Module):
    """Driving features (D=256) -> LM embeddings (H, e.g. 2048 for Qwen1.5-1.8B-Chat).

    Output sequence = [ traj token (1) | ego (1) | agents (K) | map (M) ]  ->  1 + L tokens.
      * scene tokens: LayerNorm -> + type embedding -> 2-layer MLP
      * traj token:   the selected trajectory (T x 2 positions, metres) -> MLP
    """

    def __init__(self, in_dim=256, lm_dim=2048, hidden_dim=1024, num_types=3, traj_steps=6, traj_scale=0.1):
        super().__init__()
        self.in_dim, self.lm_dim = in_dim, lm_dim
        self.traj_scale = traj_scale
        self.norm = nn.LayerNorm(in_dim)
        self.type_embed = nn.Embedding(num_types, in_dim)
        self.mlp = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, lm_dim))
        self.traj_mlp = nn.Sequential(nn.Linear(traj_steps * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, lm_dim))
        # small output scale: LM token embeddings have a small norm, a large prefix would swamp them at the start
        for seq in (self.mlp, self.traj_mlp):
            nn.init.normal_(seq[-1].weight, std=0.02)
            nn.init.zeros_(seq[-1].bias)

    def forward(self, scene, scene_type, traj, scene_mask):
        """
        scene [B, L, D], scene_type [B, L] long, traj [B, T, 2], scene_mask [B, L] bool
        returns embeds [B, 1+L, H] (float32) and mask [B, 1+L] bool
        """
        x = self.norm(scene.float()) + self.type_embed(scene_type)
        scene_emb = self.mlp(x)
        traj_emb = self.traj_mlp(traj.float().flatten(1) * self.traj_scale)[:, None]
        embeds = torch.cat([traj_emb, scene_emb], dim=1)
        mask = torch.cat([scene_mask.new_ones(scene_mask.shape[0], 1), scene_mask], dim=1)
        return embeds, mask
