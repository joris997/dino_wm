import torch
import torch.nn as nn
from torchvision import transforms
from einops import rearrange, repeat

class VWorldModel(nn.Module):
    def __init__(
        self,
        image_size,  # 224
        num_hist,
        num_pred,
        encoder,
        proprio_encoder,
        action_encoder,
        action_decoder,
        decoder,
        predictor,
        proprio_dim=0,
        action_dim=0,
        num_action_repeat=7,
        num_proprio_repeat=7,
        cfg_dict=None,
        train_encoder=True,
        train_predictor=False,
        train_decoder=True,
        train_action_decoder=True,
    ):
        super().__init__()
        self.cfg_dict = cfg_dict

        self.num_hist = num_hist
        self.local_hist = num_hist - 1 # local history excludes first frame
        self.encoder = encoder
        self.proprio_encoder = proprio_encoder
        self.action_encoder = action_encoder
        self.action_decoder = action_decoder
        self.decoder = decoder  # decoder could be None
        self.predictor = predictor  # predictor could be None
        self.train_encoder = train_encoder
        self.train_predictor = train_predictor
        self.train_decoder = train_decoder
        self.train_action_decoder = train_action_decoder
        self.num_action_repeat = num_action_repeat
        self.num_proprio_repeat = num_proprio_repeat
        self.proprio_dim = proprio_dim * num_proprio_repeat 
        self.action_dim = action_dim * num_action_repeat 
        self.emb_dim = self.encoder.emb_dim + (self.action_dim + self.proprio_dim) # Not used

        self.Ts = self.cfg_dict.frameskip/30.0 # TODO: assume 30Hz data for now
        self.print(f"num_action_repeat: {self.num_action_repeat}")
        self.print(f"num_proprio_repeat: {self.num_proprio_repeat}")
        self.print(f"proprio encoder: {proprio_encoder}")
        self.print(f"action encoder: {action_encoder}")
        self.print(f"action decoder: {action_decoder}")
        self.print(f"proprio_dim: {proprio_dim}, after repeat: {self.proprio_dim}")
        self.print(f"action_dim: {action_dim}, after repeat: {self.action_dim}")
        self.print(f"emb_dim: {self.emb_dim}")

        if "dino" in self.encoder.name:
            decoder_scale = 16  # from vqvae
            num_side_patches = image_size // decoder_scale
            self.encoder_image_size = num_side_patches * encoder.patch_size
            self.encoder_transform = transforms.Compose(
                [transforms.Resize(self.encoder_image_size)]
            )
        else:
            # set self.encoder_transform to identity transform
            self.encoder_transform = lambda x: x

        self.decoder_criterion = nn.MSELoss()
        self.decoder_latent_loss_weight = 0.25
        self.emb_criterion = nn.MSELoss()
        
    def print(self, *args):
        if self.cfg_dict.debug:
            print(*args)

    def train(self, mode=True):
        super().train(mode)
        if self.train_encoder:
            self.encoder.train(mode)
        if self.predictor is not None and self.train_predictor:
            self.predictor.train(mode)
        self.proprio_encoder.train(mode)
        self.action_encoder.train(mode)
        if self.decoder is not None and self.train_decoder:
            self.decoder.train(mode)
        if self.action_decoder is not None and self.train_action_decoder:
            self.action_decoder.train(mode)

    def eval(self):
        super().eval()
        self.encoder.eval()
        if self.predictor is not None:
            self.predictor.eval()
        self.proprio_encoder.eval()
        self.action_encoder.eval()
        if self.decoder is not None:
            self.decoder.eval()
        if self.action_decoder is not None:
            self.action_decoder.eval()

    def encode(self, obs, act): 
        """
        This method encodes the observation and actions seperately, and appends
        the proprioception to the visual embeddings. It does not append the action
        history to the embeddings, this is done in self.forward!

        input :  obs (dict): "visual", "proprio", (b, num_frames, num_hist, img_size, img_size) 
                 act: (b, num_frames, action_dim)
        output:  o (tensor): (b, num_frames, num_patches, emb_dim)
                 z (tensor): (b, num_frames, num_patches, emb_dim)
                 u (tensor): (b, num_frames, num_patches, action_emb_dim)
        """
        self.print(f"\n\tVWorldModel encode:")
        o_dct = self.encode_obs(obs)
        o, p = o_dct['visual'], o_dct['proprio']
        act_emb = self.encode_act(act)
        self.print(f"act_emb.shape: {act_emb.shape}")

        # now we create z which is [obs, pro, history of actions]. To keep the history dimension
        # consistent we remove the first state in obs and pro (as the last act_emb) is the control
        # input at this time frame which we later want to optimize for and not append to z!
        z_hist = self.local_hist    # num_hist - 1
        self.print(f"o.shape: {o.shape}, p.shape: {p.shape}, act_emb.shape: {act_emb.shape}")
        z = torch.cat([
            o, 
            p,
            act_emb[:, :-1, ...]
        ], dim=-1
        )  # (b, num_frames, num_patches, dim + proprio_dim + action_dim)
        
        # the u is now only the current u, the history of u is now part of the latent state!
        u = act_emb[:, -1:, ...]
        return o, z, u
    
    def encode_act(self, act):
        self.print(f"Encoding action shape: {act.shape}")
        act = self.action_encoder(act) # (b, num_frames, action_emb_dim)
        act = repeat(act.unsqueeze(2), "b t 1 a -> b t f a", f=196)
        act = act.repeat(1, 1, 1, self.num_action_repeat)
        self.print(f"act.shape: {act.shape}")
        return act

    def decode_act(self, act_emb):
        self.print(f"Decoding action emb shape: {act_emb.shape}")
        act = self.action_decoder(act_emb) # (b, num_frames, action_dim)
        self.print(f"act.shape: {act.shape}")
        return act
    
    def encode_proprio(self, proprio):
        self.print(f"Encoding proprio shape: {proprio.shape}")
        proprio = self.proprio_encoder(proprio)
        self.print(f"proprio.shape: {proprio.shape}")
        return proprio

    def encode_obs(self, obs):
        """
        input : obs (dict): "visual", "proprio" (b, t, 3, img_size, img_size)
        output:   z (dict): "visual", "proprio" (b, t, num_patches, encoder_emb_dim)
        """
        self.print(f"\n\tVWorldModel encode_obs:")
        vis, proprio = obs['visual'], obs['proprio']

        vis = rearrange(vis, "b t ... -> (b t) ...")
        vis = self.encoder_transform(vis)
        vis_emb = self.encoder.forward(vis)
        vis_emb = rearrange(vis_emb, "(b t) p d -> b t p d", t=obs['visual'].shape[1])

        proprio_emb = self.encode_proprio(proprio)
        proprio_emb = repeat(proprio_emb.unsqueeze(2), "b t 1 a -> b t f a", f=vis_emb.shape[2])
        proprio_emb = proprio_emb.repeat(1, 1, 1, self.num_proprio_repeat)


        o_dct = {'visual': vis_emb, 'proprio': proprio_emb}
        self.print(f"obs['visual'].shape: {obs['visual'].shape}")
        self.print(f"obs['proprio'].shape: {obs['proprio'].shape}")
        self.print(f"o_dct['visual'].shape: {o_dct['visual'].shape}")
        self.print(f"o_dct['proprio'].shape: {o_dct['proprio'].shape}")
        return o_dct

    def predict(self, z, u):  # in embedding space
        """
        input : z: (b, num_hist, num_patches, emb_dim), u: (b, 1, num_patches, action_emb_dim)
        output: z: (b, num_hist, num_patches, emb_dim)
        """
        self.print(f"\n\tVWorldModel predict:")
        self.print(f"z.shape (before): {z.shape}, u.shape: {u.shape}")

        # reshape to a batch of windows of inputs
        z_rshp = rearrange(z, "b t p d -> b (t p) d")
        u_rshp = rearrange(u, "b t p d -> b (t p) d")
        self.print(f"z_rshp.shape: {z_rshp.shape}, u_rshp.shape: {u_rshp.shape}")

        dz_rshp = self.predictor(z_rshp, u_rshp)
        dz = rearrange(dz_rshp, "b (t p) d -> b t p d", t=z.shape[1])
        zp1 = z + self.Ts*dz

        # reshape back to (b, num_hist, num_patches, emb_dim)
        self.print(f"dz.shape: {dz.shape}, zp1.shape: {zp1.shape}\n")
        return zp1, dz

    def decode(self, z, u):
        """
        input :   z: (b, num_frames, num_patches, emb_dim)
                  u: (b, num_frames, num_patches, action_emb_dim)
        output: obs: (b, num_frames, 3, img_size, img_size)
                diff: (tensor)
                act: (b, num_frames, action_dim)
        """
        self.print(f"\n\tVWorldModel decode:")
        o, p, u_hist = self.separate_emb(z)
        obs, diff = self.decode_obs(o, p)
        act_hist = self.decode_act(u_hist[:, :, -1, :])
        # u is increased in size for the patches, but they're all the same, so just take one patch
        # take the last action in the history (there is only one action actually) and add a None
        # dimension to match the desired input shape (b, num_frames, action_dim)
        act = self.decode_act(u[:, -1:, -1, :]) # (b, num_frames, action_dim)
        return obs, diff, act

    def decode_obs(self, o, p=None):
        """
        input :   z: (b, num_frames, num_patches, emb_dim)
        output: obs: (b, num_frames, 3, img_size, img_size)
        """
        # if o is a dict and p is None, extract o and p from the dict
        if isinstance(o, dict) and p is None:
            o, p = o['visual'], o['proprio']
            
        self.print(f"\n\tVWorldModel decode_obs:")
        visual, diff = self.decoder(o)  # (b*num_frames, 3, 224, 224)
        self.print(f"o.shape: {o.shape}")
        self.print(f"visual.shape (before rearrange): {visual.shape}")
        visual = rearrange(visual, "(b t) c h w -> b t c h w", t=o.shape[1])
        self.print(f"visual.shape (after rearrange): {visual.shape}")
        obs = {
            "visual": visual,
            "proprio": p,  # Note: no decoder for proprio for now!
        }
        return obs, diff
    
    def separate_emb(self, z):
        """
        input: z (b, num_frames, num_patches, emb_dim + pro_emb_dim + action_emb_dim
        output: z_obs (dict), z_act (tensor)
        """
        self.print(f"\n\tVWorldModel separate_emb:")
        self.print(f"z.shape: {z.shape}")
        o, p, u_hist = z[..., :-(self.proprio_dim*self.num_proprio_repeat + self.action_dim*self.num_action_repeat)], \
                       z[..., -(self.proprio_dim*self.num_proprio_repeat + self.action_dim*self.num_action_repeat) :-self.action_dim*self.num_action_repeat],  \
                       z[..., -self.action_dim*self.num_action_repeat:]
        self.print(f"o.shape: {o.shape}")
        self.print(f"p.shape: {p.shape}")
        self.print(f"u_hist.shape: {u_hist.shape}")
        return o, p, u_hist

    def forward(self, obs, act):
        """
        input:  obs (dict):  "visual", "proprio" (b, num_frames, 3, img_size, img_size)
                act: (b, num_frames, action_dim)
        output: z_pred: (b, num_hist, num_patches, emb_dim)
                    the latent state predictions for the next num_hist frames
                visual_pred: (b, num_hist, 3, img_size, img_size)
                    the predicted visual observations for the next num_hist frames
                visual_reconstructed: (b, num_frames, 3, img_size, img_size)
                    the reconstructed visual observations for the input frames
        """
        self.print(f"\n\nVWorldModel forward:")
        self.print(f"obs['visual'].shape: {obs['visual'].shape}, obs['proprio'].shape: {obs['proprio'].shape}, act.shape: {act.shape}")

        #! remove the first observation such that later we can append n_hist-1 actions
        #! to the observations to create a consistent latent state
        obs['visual'] = obs['visual'][:, 1:, ...]
        obs['proprio'] = obs['proprio'][:, 1:, ...]
        self.print(f"obs['visual'].shape: {obs['visual'].shape}, obs['proprio'].shape: {obs['proprio'].shape}, act.shape: {act.shape}")

        loss = 0
        loss_components = {}
        o, z, u = self.encode(obs, act)
        self.print(f"act: {act}")
        # for the targets, we remove 1 index as the last observation is removed to align it with the action history
        o_src = o[:, : self.local_hist, :, :]  # (b, num_hist, num_patches, dim)
        o_tgt = o[:, 1:1 + self.local_hist, :, :]  # (b, num_hist, num_patches, dim)
        z_src = z[:, : self.local_hist, :, :]  # (b, num_hist, num_patches, dim)
        z_tgt = z[:, 1:1 + self.local_hist, :, :]  # (b, num_hist, num_patches, dim)
        u_src = u[:, :1, :, :]  # (b, num_hist, action_dim)
        u_tgt = u[:, :1, :, :]  # (b, num_hist, action_dim)
        visual_src = obs['visual'][:, :self.local_hist, ...]  # (b, num_hist, 3, img_size, img_size)
        visual_tgt = obs['visual'][:, 1:1 + self.local_hist, ...]  # (b, num_hist, 3, img_size, img_size)
        self.print(f"\no_src.shape: {o_src.shape}, o_tgt.shape: {o_tgt.shape}")
        self.print(f"u_src.shape: {u_src.shape}, u_tgt.shape: {u_tgt.shape}")
        self.print(f"z_src.shape: {z_src.shape}, z_tgt.shape: {z_tgt.shape}")
        self.print(f"visual_src.shape: {visual_src.shape}, visual_tgt.shape: {visual_tgt.shape}")

        if self.predictor is not None:
            z_pred, dz_pred = self.predict(z_src, u_src)
            if self.decoder is not None:
                self.print(f"GOING DECODING")
                self.print(f"z_src.shape: {z_src.shape},z_pred.shape: {z_pred.shape}, u_src.shape: {u_src.shape}")
                obs_pred, diff_pred, _ = self.decode(
                    z_pred.detach(),
                    u_src.detach()
                )  # recon loss should only affect decoder
                visual_pred = obs_pred['visual']
                self.print(f"visual_pred.shape: {visual_pred.shape}, visual_tgt.shape: {visual_tgt.shape}")
                recon_loss_pred = self.decoder_criterion(visual_pred, visual_tgt)
                decoder_loss_pred = (
                    recon_loss_pred + self.decoder_latent_loss_weight * diff_pred
                )
                loss_components["decoder_recon_loss_pred"] = recon_loss_pred
                loss_components["decoder_vq_loss_pred"] = diff_pred
                loss_components["decoder_loss_pred"] = decoder_loss_pred
            else:
                visual_pred = None

            # Compute loss for visual, proprio, action dims
            self.print(f"z_pred.shape: {z_pred.shape}, z_tgt.shape: {z_tgt.shape}")
            z_visual_loss = self.emb_criterion(
                z_pred[:, :, :, :-self.action_dim-self.proprio_dim],
                z_tgt[:, :, :, :-self.action_dim-self.proprio_dim].detach()
            )
            z_proprio_loss = self.emb_criterion(
                z_pred[:, :, :, -self.proprio_dim-self.action_dim:-self.action_dim],
                z_tgt[:, :, :, -self.proprio_dim-self.action_dim:-self.action_dim].detach()
            )
            z_act_history_loss = self.emb_criterion(
                z_pred[:, :, :, -self.action_dim:],
                z_tgt[:, :, :, -self.action_dim:].detach()
            )
            z_loss = self.emb_criterion(
                z_pred[:, :, :, :], 
                z_tgt[:, :, :, :].detach()
            )

            loss = loss + z_loss
            loss_components["z_loss"] = z_loss
            loss_components["z_visual_loss"] = z_visual_loss
            loss_components["z_proprio_loss"] = z_proprio_loss
            loss_components["z_act_history_loss"] = z_act_history_loss
        else:
            visual_pred = None
            z_pred = None

        if self.decoder is not None:
            self.print(f"GOING DECODING FULL")
            self.print(f"z.shape: {z.shape}")
            obs_reconstructed, diff_reconstructed, act_reconstructed = self.decode(
                z.detach(),
                u.detach()
            )  # recon loss should only affect decoder
            # Latent reconstruction losses
            visual_reconstructed = obs_reconstructed["visual"]
            self.print(f"visual_reconstructed.shape: {visual_reconstructed.shape}, obs['visual'].shape: {obs['visual'].shape}")
            recon_loss_reconstructed = self.decoder_criterion(visual_reconstructed, obs['visual'])
            decoder_loss_reconstructed = (
                recon_loss_reconstructed
                + self.decoder_latent_loss_weight * diff_reconstructed
            )

            loss_components["decoder_recon_loss_reconstructed"] = recon_loss_reconstructed
            loss_components["decoder_vq_loss_reconstructed"] = diff_reconstructed
            loss_components["decoder_loss_reconstructed"] = decoder_loss_reconstructed
            loss = loss + decoder_loss_reconstructed

            # Control reconstruction loss
            self.print(f"act_reconstructed.shape: {act_reconstructed.shape}, act.shape: {act.shape}")
            act_loss = self.emb_criterion(act_reconstructed, act[:, :-1, :])
            loss_components["act_loss"] = act_loss
            loss = loss + act_loss
        else:
            visual_reconstructed = None

        loss_components["loss"] = loss
        return z_pred, visual_pred, visual_reconstructed, loss, loss_components

    def rollout(self, obs_0, act):
        """
        input:  obs_0 (dict): (b, n, 3, img_size, img_size)
                  act: (b, t+n, action_dim)
        output: embeddings of rollout obs
                visuals: (b, t+n+1, 3, img_size, img_size)
                z: (b, t+n+1, num_patches, emb_dim)
        """
        self.print("\n\nVWorldModel rollout:")
        #! remove the first observation such that later we can append n_hist-1 actions
        #! to the observations to create a consistent latent state
        obs_0['visual'] = obs_0['visual'][:, 1:, ...]
        obs_0['proprio'] = obs_0['proprio'][:, 1:, ...]
        
        self.print(f"obs_0['visual'].shape: {obs_0['visual'].shape}, obs_0['proprio'].shape: {obs_0['proprio'].shape}, act.shape: {act.shape}")
        num_obs_init = obs_0['visual'].shape[1]
        act_0 = act[:, :num_obs_init+1]
        action = act[:, num_obs_init+1:]
        # prepend the last action of act_0 as this is the action at the current time step
        # and therefore the first action that needs to be taken!
        action = torch.cat([act_0[:, -1:], action], dim=1)
        self.print(f"obs_0['visual'].shape: {obs_0['visual'].shape}, obs_0['proprio'].shape: {obs_0['proprio'].shape}, act_0.shape: {act_0.shape}")
        o, z, u = self.encode(obs_0, act_0)
        self.print(f"Initial o.shape: {o.shape}, z.shape: {z.shape}, u.shape: {u.shape}, action.shape: {action.shape}")
        t = 0
        while t < action.shape[1]:
            self.print(f"\nRollout at time step {t}/{action.shape[1]}")
            u_now = self.encode_act(action[:, t : t + 1, :])
            self.print(f"what goes in: z.shape: {z[:, -self.num_hist:].shape}, u_now.shape: {u_now.shape}")
            self.print(f"u_now: {u_now}")
            z_pred, _ = self.predict(z[:, -self.num_hist:], u_now)
            z_new = z_pred[:, -1:, ...]

            z = torch.cat([z, z_new], dim=1)
            self.print(f"Rollout step {t}: z.shape: {z.shape}")
            t += 1

        self.print(f"Final z.shape: {z.shape}")
        # z requires u appended in separate_emb because of the concatenated u_hist
        o, p, u_hist = self.separate_emb(z)
        obss = {'visual': o, 'proprio': p}

        # raise NotImplementedError("rollout decoding not implemented yet")
        return obss, z # TODO: add u to z_obses to analyze it later..
    
    def take_step(self, obs, act):
        """
        Take a step in the environment. The final action in act
        is considered to be the current control input.
        """
        self.print(f"\n\nVWorldModel take_step:")
        self.print(f"obs['visual'].shape: {obs['visual'].shape}, obs['proprio'].shape: {obs['proprio'].shape}, act.shape: {act.shape}")
        if obs['visual'].shape[1] == act.shape[1]:
            # if obs and act dimension match, remove the first observation
            # such that later we can append n_hist-1 actions
            obs['visual'] = obs['visual'][:, 1:, ...]
            obs['proprio'] = obs['proprio'][:, 1:, ...]
        act_0 = act[:, :-1, ...]
        act_now = act[:, -1:, ...]
        self.print(f"obs['visual'].shape: {obs['visual'].shape}, obs['proprio'].shape: {obs['proprio'].shape}, act_0.shape: {act_0.shape}, act_now.shape: {act_now.shape}")

        # Encode the observations, action history, and current action
        o, z, u = self.encode(obs, act)
        u_now = self.encode_act(act_now)

        # Take the step in the latent space
        z_pred, dz_pred = self.predict(z[:, -self.num_hist :], u_now)
        obs_pred, _, _ = self.decode(
            z_pred,
            u_now
        )  # recon loss should only affect decoder
        return obs_pred, z_pred, dz_pred
    
    def get_fz_gz(self, obs, act):
        assert obs['visual'].shape[1] == act.shape[1], "obs and act must have the same number of frames"
        act_0 = torch.zeros_like(act[:, -1:, ...])
        o, z, u = self.encode(obs, act_0)

        fz = self.predictor.get_fz(z)
        gz = self.predictor.get_gz(z)
        return fz, gz
