# adapted from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
import torch
from torch import nn
from einops import rearrange, repeat
import math

# helpers
NUM_FRAMES = 1
NUM_PATCHES = 1

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def generate_mask_matrix(npatch, nwindow):
    zeros = torch.zeros(npatch, npatch)
    ones = torch.ones(npatch, npatch)
    rows = []
    for i in range(nwindow):
        row = torch.cat([ones] * (i+1) + [zeros] * (nwindow - i-1), dim=1)
        rows.append(row)
    mask = torch.cat(rows, dim=0).unsqueeze(0).unsqueeze(0)
    return mask

class RoPEPositionalEncoding1D(nn.Module):
    """1D RoPE for temporal dimension"""
    def __init__(self, dim, max_seq_len=10000):
        super().__init__()
        assert dim % 2 == 0, "Embedding dimension must be even for RoPE"
        self.dim = dim
        
        # Compute frequency for each dimension pair
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Cache for computed embeddings
        self.register_buffer('cached_freqs', None)
        self.cached_seq_len = 0

    def forward(self, seq_len, device):
        if seq_len > self.cached_seq_len or self.cached_freqs is None:
            # Generate position indices
            t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
            # Compute frequencies
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            # Create sin and cos embeddings - repeat for interleaving
            emb = torch.cat((freqs, freqs), dim=-1)
            self.register_buffer('cached_freqs', emb)
            self.cached_seq_len = seq_len
        return self.cached_freqs[:seq_len]

class RoPEPositionalEncoding2D(nn.Module):
    """2D RoPE for spatial dimensions"""
    def __init__(self, dim, height, width):
        super().__init__()
        assert dim % 4 == 0, "Embedding dimension must be divisible by 4 for 2D RoPE"
        self.dim = dim
        self.height = height
        self.width = width
        
        # Compute frequency for each dimension pair (half for height, half for width)
        dim_quarter = dim // 4  # Quarter dimension for each height/width component
        inv_freq_h = 1.0 / (10000 ** (torch.arange(0, dim_quarter, 1).float() / dim_quarter))
        inv_freq_w = 1.0 / (10000 ** (torch.arange(0, dim_quarter, 1).float() / dim_quarter))
        
        self.register_buffer('inv_freq_h', inv_freq_h)
        self.register_buffer('inv_freq_w', inv_freq_w)
        
        # Pre-compute spatial embeddings
        self._compute_spatial_embeddings()
    
    def _compute_spatial_embeddings(self):
        # Create coordinate grids
        h = torch.arange(self.height).type_as(self.inv_freq_h)
        w = torch.arange(self.width).type_as(self.inv_freq_w)
        
        # Compute frequencies for height and width
        freqs_h = torch.einsum('i,j->ij', h, self.inv_freq_h)
        freqs_w = torch.einsum('i,j->ij', w, self.inv_freq_w)
        
        # Create 2D frequency maps
        freqs_h = repeat(freqs_h, 'h d -> h w d', w=self.width)
        freqs_w = repeat(freqs_w, 'w d -> h w d', h=self.height)
        
        # Combine height and width frequencies - repeat for interleaving
        freqs_h = torch.cat((freqs_h, freqs_h), dim=-1)
        freqs_w = torch.cat((freqs_w, freqs_w), dim=-1)
        
        # Concatenate to form full 2D embedding
        spatial_emb = torch.cat((freqs_h, freqs_w), dim=-1)
        spatial_emb = rearrange(spatial_emb, 'h w d -> (h w) d')
        
        self.register_buffer('spatial_emb', spatial_emb)
    
    def forward(self):
        return self.spatial_emb

def apply_rope_1d(q, k, freqs):
    """Apply 1D RoPE to query and key tensors"""
    cos_freqs = torch.cos(freqs)
    sin_freqs = torch.sin(freqs)
    
    # Split into pairs for rotation
    q_even, q_odd = q[..., 0::2], q[..., 1::2]
    k_even, k_odd = k[..., 0::2], k[..., 1::2]
    
    cos_freqs_even = cos_freqs[..., 0::2]
    sin_freqs_even = sin_freqs[..., 0::2]
    cos_freqs_odd = cos_freqs[..., 1::2]
    sin_freqs_odd = sin_freqs[..., 1::2]
    
    # Apply rotation
    q_even_new = q_even * cos_freqs_even - q_odd * sin_freqs_even
    q_odd_new = q_even * sin_freqs_odd + q_odd * cos_freqs_odd
    
    k_even_new = k_even * cos_freqs_even - k_odd * sin_freqs_even
    k_odd_new = k_even * sin_freqs_odd + k_odd * cos_freqs_odd
    
    # Interleave back
    q_out = torch.stack([q_even_new, q_odd_new], dim=-1).flatten(-2)
    k_out = torch.stack([k_even_new, k_odd_new], dim=-1).flatten(-2)
    
    return q_out, k_out

def apply_rope_2d(q, k, freqs):
    """Apply 2D RoPE to query and key tensors"""
    cos_freqs = torch.cos(freqs)
    sin_freqs = torch.sin(freqs)
    
    # Split into pairs for rotation  
    q_even, q_odd = q[..., 0::2], q[..., 1::2]
    k_even, k_odd = k[..., 0::2], k[..., 1::2]
    
    cos_freqs_even = cos_freqs[..., 0::2]
    sin_freqs_even = sin_freqs[..., 0::2]
    cos_freqs_odd = cos_freqs[..., 1::2]
    sin_freqs_odd = sin_freqs[..., 1::2]
    
    # Apply rotation
    q_even_new = q_even * cos_freqs_even - q_odd * sin_freqs_even
    q_odd_new = q_even * sin_freqs_odd + q_odd * cos_freqs_odd
    
    k_even_new = k_even * cos_freqs_even - k_odd * sin_freqs_even
    k_odd_new = k_even * sin_freqs_odd + k_odd * cos_freqs_odd
    
    # Interleave back
    q_out = torch.stack([q_even_new, q_odd_new], dim=-1).flatten(-2)
    k_out = torch.stack([k_even_new, k_odd_new], dim=-1).flatten(-2)
    
    return q_out, k_out

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., num_frames = 4, visual_patches = 196, additional_patches = 2):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.num_frames = num_frames
        self.visual_patches = visual_patches
        self.additional_patches = additional_patches
        self.total_patches = visual_patches + additional_patches
        
        # Calculate visual spatial dimensions (should be perfect square)
        self.height = int(math.sqrt(visual_patches))
        self.width = int(math.sqrt(visual_patches))
        assert self.height * self.width == visual_patches, f"Visual patches ({visual_patches}) must be perfect square"

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        # Split dimension for temporal and spatial RoPE (per head)
        assert dim_head % 2 == 0, "dim_head must be even to split for temporal/spatial RoPE"
        self.dim_temporal_per_head = dim_head // 2
        self.dim_spatial_per_head = dim_head // 2
        
        # RoPE encodings (using per-head dimensions)
        self.rope_temporal = RoPEPositionalEncoding1D(self.dim_temporal_per_head, max_seq_len=num_frames)
        self.rope_spatial = RoPEPositionalEncoding2D(self.dim_spatial_per_head, self.height, self.width)
        # Additional learnable positional embedding for non-visual patches (per head)
        self.additional_pos_emb = nn.Parameter(torch.randn(1, 1, additional_patches, dim_head))

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        
        # Don't hardcode device for bias - it will be moved with the model
        self.register_buffer('bias', generate_mask_matrix(NUM_PATCHES, NUM_FRAMES))

    def forward(self, x):
        (
            B,
            T,
            C,
        ) = x.size()
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        # Calculate head dimension for splitting
        head_dim = q.shape[-1]
        dim_per_head = head_dim // 2
        
        # Separate visual patches from additional patches
        # Input format: (b, h, num_frames * (visual_patches + additional_patches), d)
        visual_seq_len = self.num_frames * self.visual_patches
        additional_seq_len = self.num_frames * self.additional_patches
        
        # Split into visual and additional patches
        q_visual = q[:, :, :visual_seq_len, :]
        k_visual = k[:, :, :visual_seq_len, :]
        v_visual = v[:, :, :visual_seq_len, :]
        
        q_additional = q[:, :, visual_seq_len:, :]
        k_additional = k[:, :, visual_seq_len:, :]
        v_additional = v[:, :, visual_seq_len:, :]
        
        # ===== Process visual patches with RoPE =====
        # Split q, k into temporal and spatial parts for visual patches
        q_vis_temporal = q_visual[..., :dim_per_head]
        q_vis_spatial = q_visual[..., dim_per_head:]
        k_vis_temporal = k_visual[..., :dim_per_head]
        k_vis_spatial = k_visual[..., dim_per_head:]
        
        # Apply RoPE to temporal part (1D) for visual patches
        q_temp_reshaped = rearrange(q_vis_temporal, 'b h (p t) d -> b h p t d', t=self.num_frames)
        k_temp_reshaped = rearrange(k_vis_temporal, 'b h (p t) d -> b h p t d', t=self.num_frames)
        
        # Get temporal frequencies and expand to match tensor dimensions
        temp_freqs = self.rope_temporal(self.num_frames, device=x.device)
        # temp_freqs shape: (num_frames, dim_temporal_per_head)
        # Expand to match q_temp_reshaped: (B, heads, visual_patches, num_frames, dim_per_head)
        temp_freqs = temp_freqs.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, t, d)
        temp_freqs = temp_freqs.expand(B, self.heads, self.visual_patches, -1, -1)
        
        # Apply 1D RoPE to temporal dimension
        q_temp_rope, k_temp_rope = apply_rope_1d(q_temp_reshaped, k_temp_reshaped, temp_freqs)
        q_temp_rope = rearrange(q_temp_rope, 'b h p t d -> b h (p t) d')
        k_temp_rope = rearrange(k_temp_rope, 'b h p t d -> b h (p t) d')
        
        # Apply RoPE to spatial part (2D) for visual patches
        q_spat_reshaped = rearrange(q_vis_spatial, 'b h (t p) d -> b h t p d', t=self.num_frames)
        k_spat_reshaped = rearrange(k_vis_spatial, 'b h (t p) d -> b h t p d', t=self.num_frames)
        
        # Get spatial frequencies and expand to match tensor dimensions
        spat_freqs = self.rope_spatial()
        # spat_freqs shape: (visual_patches, dim_spatial_per_head)
        # Expand to match q_spat_reshaped: (B, heads, num_frames, visual_patches, dim_per_head)
        spat_freqs = spat_freqs.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, p, d)
        spat_freqs = spat_freqs.expand(B, self.heads, self.num_frames, -1, -1)
        
        # Apply 2D RoPE to spatial dimension
        q_spat_rope, k_spat_rope = apply_rope_2d(q_spat_reshaped, k_spat_reshaped, spat_freqs)
        q_spat_rope = rearrange(q_spat_rope, 'b h t p d -> b h (t p) d')
        k_spat_rope = rearrange(k_spat_rope, 'b h t p d -> b h (t p) d')
        
        # Concatenate temporal and spatial parts back for visual patches
        q_visual_rope = torch.cat([q_temp_rope, q_spat_rope], dim=-1)
        k_visual_rope = torch.cat([k_temp_rope, k_spat_rope], dim=-1)
        
        # ===== Process additional patches with learned embeddings =====
        # Apply learned positional embeddings to additional patches
        additional_pos = self.additional_pos_emb.expand(B, self.heads, -1, -1)
        additional_pos = repeat(additional_pos, 'b h p d -> b h (t p) d', t=self.num_frames)
        
        q_additional_pos = q_additional + additional_pos
        k_additional_pos = k_additional + additional_pos
        
        # Combine visual and additional patches back
        q_rope = torch.cat([q_visual_rope, q_additional_pos], dim=2)
        k_rope = torch.cat([k_visual_rope, k_additional_pos], dim=2)
        v_combined = torch.cat([v_visual, v_additional], dim=2)

        dots = torch.matmul(q_rope, k_rope.transpose(-1, -2)) * self.scale
        # apply causal mask
        dots = dots.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v_combined)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., num_frames = 4, visual_patches = 196, additional_patches = 2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout, 
                         num_frames = num_frames, visual_patches = visual_patches, additional_patches = additional_patches),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)
    
class ViTPredictor(nn.Module):
    def __init__(self, *, num_patches, num_frames, dim, depth, heads, mlp_dim, pool='cls', dim_head=64, dropout=0., emb_dropout=0., visual_patches=196, additional_patches=2):
        super().__init__()
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        
        # Verify input consistency
        assert num_patches == visual_patches + additional_patches, f"num_patches ({num_patches}) should equal visual_patches ({visual_patches}) + additional_patches ({additional_patches})"
        
        # update params for adding causal attention masks
        global NUM_FRAMES, NUM_PATCHES
        NUM_FRAMES = num_frames
        NUM_PATCHES = num_patches

        # Store patch information
        self.num_frames = num_frames
        self.num_patches = num_patches
        self.visual_patches = visual_patches
        self.additional_patches = additional_patches
        
        # Calculate visual spatial dimensions (should be perfect square)
        self.height = int(math.sqrt(visual_patches))
        self.width = int(math.sqrt(visual_patches))
        
        # Verify that visual patches form a square grid
        assert self.height * self.width == visual_patches, f"visual_patches ({visual_patches}) must be a perfect square for 2D RoPE"

        # Remove traditional positional embedding since we're using RoPE
        # self.pos_embedding = nn.Parameter(torch.randn(1, num_frames * (num_patches), dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout,
                                     num_frames=num_frames, visual_patches=visual_patches, additional_patches=additional_patches)
        self.pool = pool

    def forward(self, x): # x: (b, window_size * n_patches, 384)
        # print(f"x shape in ViTPredictor: {x.shape}")
        b, n, _ = x.shape
        
        # No positional embedding addition since RoPE is applied in attention
        # x = x + self.pos_embedding[:, :n]
        x = self.dropout(x) 
        x = self.transformer(x) 
        return x