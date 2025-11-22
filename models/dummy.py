import torch.nn as nn
import torch

class DummyModel(nn.Module):
    def __init__(self, emb_dim, **kwargs):
        super().__init__()
        self.name = "dummy"
        self.latent_ndim = 1
        self.emb_dim = emb_dim
        self.fc = nn.Linear(emb_dim, 1)  # not used

    def forward(self, x):
        b, dim = x.shape
        num_repeat = self.emb_dim // dim
        processed_x = torch.zeros([b, self.emb_dim]).to(x.device)
        x_repeated = x.repeat(1, num_repeat)
        processed_x[:, :num_repeat * dim] = x_repeated
        return x.unsqueeze(1)  # return shape: (b, 1(or # patches), num_features)

class DummyRepeatActionEncoder(nn.Module):
    def __init__(self, in_chans, emb_dim, **kwargs):
        super().__init__()
        self.name = "dummy_repeat"
        self.latent_ndim = 1
        self.in_chans = in_chans
        self.emb_dim = emb_dim
        self.fc = nn.Linear(in_chans, 1)  # not used

    def forward(self, act):
        '''
        (b, t, act_dim) --> (b, t, action_emb_dim)
        '''
        b, t, act_dim = act.shape
        num_repeat = self.emb_dim // act_dim
        processed_act = torch.zeros([b, t, self.emb_dim]).to(act.device)
        act_repeated = act.repeat(1, 1, num_repeat)
        processed_act[:, :, :num_repeat * act_dim] = act_repeated
        return processed_act

class DummyRepeatActionDecoder(nn.Module):
    def __init__(self, out_chans, emb_dim, **kwargs):
        super().__init__()
        self.name = "dummy_repeat"
        self.latent_ndim = 1
        self.out_chans = out_chans
        self.emb_dim = emb_dim
        self.fc = nn.Linear(emb_dim, 1)  # not used

    def forward(self, act_emb):
        '''
        (b, t, action_emb_dim) --> (b, t, act_dim)
        '''
        b, t, emb_dim = act_emb.shape
        num_repeat = emb_dim // self.out_chans
        processed_act = torch.zeros([b, t, self.out_chans]).to(act_emb.device)
        act_repeated = act_emb.repeat(1, 1, num_repeat)
        processed_act[:, :, :num_repeat * self.out_chans] = act_repeated
        return processed_act