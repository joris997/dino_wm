# adapted from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
import torch
from torch import nn
from einops import rearrange, repeat

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
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        self.bias = generate_mask_matrix(NUM_PATCHES, NUM_FRAMES).to('cuda')

    def forward(self, x):
        (
            B,
            T,
            C,
        ) = x.size()
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # apply causal mask
        dots = dots.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
        
class SplitAttention(nn.Module):
    def __init__(self, dim, heads_f=4, heads_g=4, dim_head = 64, dropout = 0.):
        super().__init__()
        self.heads_f = heads_f
        self.heads_g = heads_g
        self.total_heads = self.heads_f + self.heads_g
        self.dim_head = dim_head
        inner_dim = self.dim_head *  self.total_heads


        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_k = nn.Linear(dim, inner_dim,  bias = False)
        self.to_v = nn.Linear(dim, inner_dim,  bias = False)
        self.norm = nn.LayerNorm(dim)

        self.scale = dim_head ** -0.5
        self.dropout = nn.Dropout(dropout)
        
        self.to_out_f = nn.Linear(dim_head * self.heads_f, dim)
        self.to_out_g = nn.Linear(dim_head * self.heads_g, dim)
        
        # causal mask
        self.register_buffer('bias', generate_mask_matrix(NUM_PATCHES, NUM_FRAMES))

    def forward(self, x):
        B, T, C = x.size()
        x = self.norm(x)

        q = rearrange(self.to_q(x), 'b n (h d) -> b h n d', h = self.total_heads)
        k = rearrange(self.to_k(x), 'b n (h d) -> b h n d', h = self.total_heads)
        v = rearrange(self.to_v(x), 'b n (h d) -> b h n d', h = self.total_heads)

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        # avoid all -inf rows
        causal_mask = self.bias[:, :, :T, :T]
        dots = dots.masked_fill(causal_mask == 0, float("-inf"))

        attn = torch.softmax(dots, dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)

        out = torch.matmul(attn, v)
        out_f = out[:,:self.heads_f]
        out_g = out[:,self.heads_f:]
        
        out_f = rearrange(out_f, 'b h n d -> b n (h d)')
        out_g = rearrange(out_g, 'b h n d -> b n (h d)')
        return self.to_out_f(out_f), self.to_out_g(out_g)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)
        
class SplitTransformer(nn.Module):
    def __init__(self, dim, depth, heads_f, heads_g, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                SplitAttention(dim, heads_f=heads_f, heads_g=heads_g, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
        self.norm_f = nn.LayerNorm(dim)
        self.norm_g = nn.LayerNorm(dim)

    def forward(self, x):
        f,g = x, x      
        for attn, ff_f, ff_g in self.layers:
            f_attn, g_attn = attn(x)
            f = f + 0.5*f_attn
            g = g + 0.5*g_attn
            f = f + 0.5*ff_f(f)
            g = g + 0.5*ff_g(g)
        return self.norm_f(f), self.norm_g(g)
    
class ViTPredictor(nn.Module):
    def __init__(self, *, num_patches, num_frames, dim, action_dim,
                depth, heads, mlp_dim, pool='cls', dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.debug = False
        
        # update params for adding causal attention masks
        global NUM_FRAMES, NUM_PATCHES
        NUM_FRAMES = num_frames
        NUM_PATCHES = num_patches
        self.print(f"ViTPredictor constructor:")
        self.print(f"num_frames={num_frames}, num_patches={num_patches}, dim={dim}, depth={depth}, heads={heads}, mlp_dim={mlp_dim}")
        self.dim = dim
        self.action_dim = action_dim

        self.pos_embedding = nn.Parameter(0.05*torch.randn(1, num_frames * (num_patches), dim)) # dim for the pos encodings
        self.dropout = nn.Dropout(emb_dropout)

        heads_f = heads // 2
        heads_g = heads - heads_f
        self.transformer = SplitTransformer(dim, depth, heads_f, heads_g, dim_head, mlp_dim, dropout)
        
        # fz of size dim, gz matrix of size (dim, u_dim)
        hidden = mlp_dim
        self.fz_net = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.dim)
        )
        self.gz_net = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.dim * self.action_dim)
        )
        self.pool = pool

        # old pos emb
        self.pos_embedding_old = nn.Parameter(torch.randn(1, num_frames * (num_patches), dim+10)) # dim for the pos encodings
        self.dropout_old = nn.Dropout(emb_dropout)
        self.transformer_old = Transformer(dim+10, depth, heads, dim_head, mlp_dim, dropout)
        self.pool_old = pool
    
    def print(self, *args):
        if self.debug:
            print(*args)

    def forward(self, x, u_now): # x: (b, window_size * H/patch_size * W/patch_size, 384)
        # """
        # input: x: (b, num_hist * num_patches per img, emb_dim)
        #        u_now: control input of last frame (b, 1, action_dim)
        # output: dz: (b, num_hist * num_patches per img, emb_dim)
        # """
        # b, n, e = x.shape
        # p = NUM_PATCHES
        # t = n // p

        # # print(f"\n\nViTPredictor forward: x.shape={x.shape}, u_now.shape={u_now.shape}")
        # #? x size: [b, num_hist * num_patches, embedding dim + actions_hist]
        # x = x + self.pos_embedding[:, :n]
        # x = self.dropout(x) 
        
        # xf, xg = self.transformer(x)

        # self.print(f"x.shape (after transformer): {x.shape}")
        # #? size: [b, num_hist * num_patches, embedding dim + actions_hist]
        # # plot x into two to get f(z) and g(z) so later we can compute
        # # dz = f(z) + g(z)*u_now
        # fz = self.fz_net(xf)
        # gz = self.gz_net(xg)
        # # repeat u_now 3 times to go from [16,196,10] to [16,588,10]
        # u_now = repeat(u_now, 'b p d -> b (t p) d', t=t, p=p)  # (b, num_hist * num_patches per img, action_dim)
        # self.print(f"fz.shape: {fz.shape}, gz.shape: {gz.shape}, u_now.shape: {u_now.shape}")
        # # gz: [16, 588, 404*10], u_now: [16, 588, 10]
        # # we want gz@u_now to be [16, 588, 404]
        # gz = rearrange(gz, 'b n (d u) -> b n d u', u=self.action_dim)  # (b, num_hist * num_patches per img, dim, action_dim)
        # self.print(f"gz.shape (after rearrange): {gz.shape}")
        # gz_u = torch.einsum('bndu,bnu->bnd', gz, u_now)  # (b, num_hist * num_patches per img, dim)
        # self.print(f"gz_u.shape (after einsum): {gz_u.shape}")
        # dz = fz + gz_u  # (b, num_hist * num_patches per img, dim)
        # self.print(f"dz.shape (final output): {dz.shape}\n")
        
        # old ViT implementation
        b, n, e = x.shape
        # append u_now to each patch embedding
        print(f"ViTPredictor forward: x.shape={x.shape}, u_now.shape={u_now.shape}")
        # repeat u_now 3 times in the second dimension
        u_now_expanded = repeat(u_now, 'b n d -> b (repeat n) d', repeat=3)
        print(f"u_now_expanded.shape={u_now_expanded.shape}")
        x = torch.cat([x, u_now_expanded], dim=-1)  # (b, n, e + action_dim
        # pass throuhgh old ViT
        x = x + self.pos_embedding_old[:, :n]
        x = self.dropout_old(x)
        x = self.transformer_old(x)
        return x[...,:-10]
    
    def get_fz(self, x):
        b, n, e = x.shape

        x = x + self.pos_embedding[:, :n]
        x = self.dropout(x)
        xf, xg = self.transformer(x)

        fz = self.fz_net(xf)
        return fz
    
    def get_gz(self, x):
        b, n, e = x.shape

        x = x + self.pos_embedding[:, :n]
        x = self.dropout(x)
        xf, xg = self.transformer(x)

        gz = self.gz_net(xg)
        gz = rearrange(gz, 'b n (d u) -> b n d u', u=self.action_dim)  # (b, num_hist * num_patches per img, dim, action_dim)
        return gz
