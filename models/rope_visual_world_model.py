import torch
import torch.nn as nn
from torchvision import transforms
from einops import rearrange, repeat
from models.behavioral_cloner import mdn_loss

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
        proprio_decoder,
        behavioral_cloner,
        decoder,
        predictor,
        proprio_dim=0,
        action_dim=0,
        cfg_dict=None,
        info_dict = None,
        train_encoder=True,
        train_predictor=False,
        train_decoder=True,
        train_action_decoder=True,
        train_proprio_decoder=True,
        train_behavioral_cloner=True,
    ):
        super().__init__()
        self.cfg_dict = cfg_dict
        self.info_dict = info_dict # if already trained, contrains info on mean and std of z

        self.num_hist = num_hist
        self.encoder = encoder
        self.proprio_encoder = proprio_encoder
        self.action_encoder = action_encoder
        self.action_decoder = action_decoder
        self.proprio_decoder = proprio_decoder
        self.behavioral_cloner = behavioral_cloner  # could be None
        self.decoder = decoder  # decoder could be None
        self.predictor = predictor  # predictor could be None
        self.train_encoder = train_encoder
        self.train_predictor = train_predictor
        self.train_decoder = train_decoder
        self.train_action_decoder = train_action_decoder
        self.train_proprio_decoder = train_proprio_decoder
        self.train_behavioral_cloner = train_behavioral_cloner
        self.proprio_dim = proprio_dim 
        self.action_dim = action_dim 
        self.emb_dim = self.encoder.emb_dim + (self.action_dim + self.proprio_dim) # Not used

        self.Ts = self.cfg_dict.frameskip/30.0 # TODO: assume 30Hz data for now
        #self.print(f"proprio encoder: {proprio_encoder}")
        #self.print(f"action encoder: {action_encoder}")
        #self.print(f"action decoder: {action_decoder}")
        #self.print(f"proprio decoder: {proprio_decoder}")
        #self.print(f"behavioral_cloner: {behavioral_cloner}")
        #self.print(f"proprio_dim: {proprio_dim}, after repeat: {self.proprio_dim}")
        #self.print(f"action_dim: {action_dim}, after repeat: {self.action_dim}")
        #self.print(f"emb_dim: {self.emb_dim}")

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
        self.BC_criterion = mdn_loss
        
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
        if self.proprio_decoder is not None and self.train_proprio_decoder:
            self.proprio_decoder.train(mode)
        if self.behavioral_cloner is not None and self.train_behavioral_cloner:
            self.behavioral_cloner.train(mode)

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
        if self.proprio_decoder is not None:
            self.proprio_decoder.eval()
        if self.behavioral_cloner is not None:
            self.behavioral_cloner.eval()

    def encode(self, obs, act, with_pred=False): 
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
        #self.print(f"\n\tVWorldModel encode:")
        o_dct = self.encode_obs(obs)
        o, p = o_dct['visual'], o_dct['proprio']
        #self.print(f"act.shape: {act.shape}")
        u = self.encode_act(act)
        #self.print(f"act_emb.shape: {act_emb.shape}")

        # Create latent state by appending the new proprio and action patch
        #self.print(f"o.shape: {o.shape}, p.shape: {p.shape}, u.shape: {u.shape}")
        z = torch.cat([
            o, 
            p,
            u
        ], dim=-2
        )  # (b, num_frames, num_patches, dim + proprio_dim + action_dim)
        
        return o, z
    
    def encode_act(self, act):
        """
        input : act (b, num_frames, num_skip_frames*action_dim)
        """
        #TODO: why do I need to do this for encode_act but not encode_proprio?
        #self.print(f"Encoding action shape: {act.shape}")
        act = self.action_encoder(act) # (b, num_frames, 1, 384)
        # act = rearrange(act, "b t d -> b t 1 d") # (b, num_frames, 1, 384)
        #self.print(f"act.shape: {act.shape}")
        return act

    def decode_act(self, act_emb):
        #self.print(f"Decoding action emb shape: {act_emb.shape}")
        act = self.action_decoder(act_emb) # (b, num_frames, action_dim)
        #self.print(f"act.shape: {act.shape}")
        return act
    
    def decode_proprio(self, proprio_emb):
        #self.print(f"Decoding proprio emb shape: {proprio_emb.shape}")
        proprio = self.proprio_decoder(proprio_emb) # (b, num_frames, proprio_dim)
        #self.print(f"proprio.shape: {proprio.shape}")
        return proprio
    
    def encode_proprio(self, proprio):
        #self.print(f"Encoding proprio shape: {proprio.shape}")
        proprio = self.proprio_encoder(proprio) # (b, num_frames, 384)
        # proprio = rearrange(proprio, "b t d -> b t 1 d") # (b, num_frames, 1, 384)
        #self.print(f"proprio.shape: {proprio.shape}")
        return proprio

    def encode_obs(self, obs):
        """
        input : obs (dict): "visual"  (b, t, 3, img_size, img_size), 
                            "proprio" (b, t, frameskip*proprio_dim)
        output:   z (dict): "visual", "proprio" (b, t, num_patches, encoder_emb_dim)
        """
        #self.print(f"\n\tVWorldModel encode_obs:")
        vis, proprio = obs['visual'], obs['proprio']

        # Visualization
        vis = rearrange(vis, "b t ... -> (b t) ...")
        vis = self.encoder_transform(vis)
        #self.print(f"vis.shape (after transform): {vis.shape}")
        vis_emb = self.encoder.forward(vis)
        vis_emb = rearrange(vis_emb, "(b t) p d -> b t p d", t=obs['visual'].shape[1])

        # Proprioception
        proprio_emb = self.encode_proprio(proprio) # (b, t, 384)
        # proprio_emb = rearrange(proprio_emb, "b t d -> b t 1 d") # (b, t, 1, 384)

        o_dct = {'visual': vis_emb, 'proprio': proprio_emb}
        #self.print(f"obs['visual'].shape: {obs['visual'].shape}")
        #self.print(f"obs['proprio'].shape: {obs['proprio'].shape}")
        #self.print(f"o_dct['visual'].shape: {o_dct['visual'].shape}")
        #self.print(f"o_dct['proprio'].shape: {o_dct['proprio'].shape}")
        return o_dct

    def predict(self, z):  # in embedding space
        """
        input : z: (b, num_hist, num_patches, emb_dim), u: (b, 1, num_patches, action_emb_dim)
        output: z: (b, num_hist, num_patches, emb_dim)
        """
        #self.print(f"\n\tVWorldModel predict:")
        #self.print(f"z.shape (before): {z.shape}")

        # reshape to a batch of windows of inputs
        z_rshp = rearrange(z, "b t p d -> b (t p) d")
        #self.print(f"z_rshp.shape: {z_rshp.shape}")
        zp1_rshp = self.predictor(z_rshp)
        zp1 = rearrange(zp1_rshp, "b (t p) d -> b t p d", t=z.shape[1])

        # reshape back to (b, num_hist, num_patches, emb_dim)
        #self.print(f"zp1.shape: {zp1.shape}\n")
        return zp1

    def decode(self, z):
        """
        input :   z: (b, num_frames, num_patches, emb_dim)
                  u: (b, num_frames, num_patches, action_emb_dim)
        output: obs: (b, num_frames, 3, img_size, img_size)
                diff: (tensor)
                act: (b, num_frames, action_dim)
        """
        #self.print(f"\n\tVWorldModel decode:")
        #self.print(f"z.shape: {z.shape}")
        o, p, u = self.separate_emb(z)
        obs, diff = self.decode_obs(o, p)
        act = self.decode_act(u)
        return obs, diff, act

    def decode_obs(self, o, p=None):
        """
        input :   z: (b, num_frames, num_patches, emb_dim)
        output: obs: (b, num_frames, 3, img_size, img_size)
        """
        # if o is a dict and p is None, extract o and p from the dict
        if isinstance(o, dict) and p is None:
            o, p = o['visual'], o['proprio']
            
        #self.print(f"\n\tVWorldModel decode_obs:")
        visual, diff = self.decoder(o)  # (b*num_frames, 3, 224, 224)
        #self.print(f"o.shape: {o.shape}")
        #self.print(f"visual.shape (before rearrange): {visual.shape}")
        visual = rearrange(visual, "(b t) c h w -> b t c h w", t=o.shape[1])
        #self.print(f"visual.shape (after rearrange): {visual.shape}")

        proprio = self.decode_proprio(p) if p is not None else None

        obs = {
            "visual": visual,
            "proprio": proprio,  # Note: no decoder for proprio for now!
        }
        return obs, diff
    
    def separate_emb(self, z):
        """
        input: z (b, num_frames, num_patches, emb_dim + pro_emb_dim + action_emb_dim
        output: z_obs (dict), z_act (tensor)
        """
        #self.print(f"\n\tVWorldModel separate_emb:")
        #self.print(f"z.shape: {z.shape}")
        o, p, u = z[:, :, :196, :], \
                  z[:, :, 196:197, :], \
                  z[:, :, 197:, :]
        #self.print(f"o.shape: {o.shape}")
        #self.print(f"p.shape: {p.shape}")
        #self.print(f"u.shape: {u.shape}")
        return o, p, u

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
        #self.print(f"\n\nVWorldModel forward:")
        #self.print(f"obs['visual'].shape: {obs['visual'].shape}, obs['proprio'].shape: {obs['proprio'].shape}, act.shape: {act.shape}")
        #self.print(f"obs['visual'].shape: {obs['visual'].shape}, obs['proprio'].shape: {obs['proprio'].shape}, act.shape: {act.shape}")
        
        # add patch dimension for proprio and act
        obs['proprio'] = rearrange(obs['proprio'], "b t d -> b t 1 d")
        act = rearrange(act, "b t d -> b t 1 d")

        loss = 0
        loss_components = {}
        o, z = self.encode(obs, act, with_pred=True)
        #self.print(f"act: {act}")
        #self.print(f"o.shape: {o.shape}, z.shape: {z.shape}, u.shape: {u.shape}")
        # for the targets, we remove 1 index as the last observation is removed to align it with the action history
        o_src = o[:, : self.num_hist, :, :]  # (b, num_hist, num_patches, dim)
        o_tgt = o[:, 1:1 + self.num_hist, :, :]  # (b, num_hist, num_patches, dim)
        z_src = z[:, : self.num_hist, :, :]  # (b, num_hist, num_patches, dim)
        z_tgt = z[:, 1:1 + self.num_hist, :, :]  # (b, num_hist, num_patches, dim)
        visual_src = obs['visual'][:, :self.num_hist, ...]  # (b, num_hist, 3, img_size, img_size)
        visual_tgt = obs['visual'][:, 1:1 + self.num_hist, ...]  # (b, num_hist, 3, img_size, img_size)
        proprio_src = obs['proprio'][:, :self.num_hist, ...]  # (b, num_hist, proprio_dim)
        proprio_tgt = obs['proprio'][:, 1:1 + self.num_hist, ...]  # (b, num_hist, proprio_dim)

        #self.print(f"\no_src.shape: {o_src.shape}, o_tgt.shape: {o_tgt.shape}")
        #self.print(f"u_src.shape: {u_src.shape}, u_tgt.shape: {u_tgt.shape}")
        #self.print(f"z_src.shape: {z_src.shape}, z_tgt.shape: {z_tgt.shape}")
        #self.print(f"visual_src.shape: {visual_src.shape}, visual_tgt.shape: {visual_tgt.shape}")
        #self.print(f"proprio_src.shape: {proprio_src.shape}, proprio_tgt.shape: {proprio_tgt.shape}")


        if self.predictor is not None:
            z_pred = self.predict(z_src)
            if self.decoder is not None:
                #self.print(f"GOING DECODING")
                #self.print(f"z_src.shape: {z_src.shape},z_pred.shape: {z_pred.shape}, u_src.shape: {u_src.shape}")
                obs_pred, diff_pred, _ = self.decode(
                    z_pred.detach(),
                )  # recon loss should only affect decoder
                visual_pred = obs_pred['visual']
                #self.print(f"visual_pred.shape: {visual_pred.shape}, visual_tgt.shape: {visual_tgt.shape}")
                recon_loss_pred = self.decoder_criterion(visual_pred, visual_tgt)
                decoder_loss_pred = (
                    recon_loss_pred + self.decoder_latent_loss_weight * diff_pred
                )
                loss_components["decoder_recon_loss_pred"] = recon_loss_pred
                loss_components["decoder_vq_loss_pred"] = diff_pred
                loss_components["decoder_loss_pred"] = decoder_loss_pred

                if self.proprio_decoder is not None:
                    proprio_pred = obs_pred['proprio']
                    #self.print(f"proprio_pred.shape: {proprio_pred.shape}, proprio_tgt.shape: {proprio_tgt.shape}")
                    recon_loss_proprio_pred = self.decoder_criterion(proprio_pred, proprio_tgt)
                    loss_components["decoder_recon_loss_proprio_pred"] = recon_loss_proprio_pred
                    decoder_loss_pred = decoder_loss_pred

            else:
                visual_pred = None

            # Compute loss for visual, proprio, action dims
            #self.print(f"z_pred.shape: {z_pred.shape}, z_tgt.shape: {z_tgt.shape}")
            z_visual_loss = self.emb_criterion(
                z_pred[:, :, :196, :],
                z_tgt[:, :, :196, :].detach()
            )
            z_proprio_loss = self.emb_criterion(
                z_pred[:, :, 196:197, :],
                z_tgt[:, :, 196:197, :].detach()
            )
            # z_act_loss = self.emb_criterion(
            #     z_pred[:, :, 197:198, :],
            #     z_tgt[:, :, 197:198, :].detach()
            # )
            z_loss = self.emb_criterion(
                z_pred[:, :, :, :], 
                z_tgt[:, :, :, :].detach()
            )

            loss = loss + z_loss
            loss_components["z_loss"] = z_loss
            loss_components["z_visual_loss"] = z_visual_loss
            loss_components["z_proprio_loss"] = z_proprio_loss
            # loss_components["z_act_loss"] = z_act_loss
        else:
            visual_pred = None
            z_pred = None

        if self.behavioral_cloner is not None and self.train_behavioral_cloner:
            # TODO: implement behavioral cloning loss with new structure
            #self.print(f"GOING BEHAVIORAL CLONING")
            # remove the history- and patch embedding dimension of u here because
            # u_src and u_pred only have one time step and all patches are the same
            #self.print(f"z_src.shape: {z_src.shape}, u_src.shape: {u_src.shape}, u_tgt.shape: {u_tgt.shape}")
            logits, means, logstds = self.behavioral_cloner(z_src) # u is already present in z_src
            #self.print(f"logits.shape: {logits.shape}, means.shape: {means.shape}, logstds.shape: {logstds.shape}, act.shape: {act.shape}")
            bc_loss = self.BC_criterion(logits, means, logstds)
            loss_components["bc_loss"] = bc_loss
            loss = loss + bc_loss

        if self.decoder is not None:
            #self.print(f"GOING DECODING FULL")
            #self.print(f"z.shape: {z.shape}")
            obs_reconstructed, diff_reconstructed, act_reconstructed = self.decode(
                z.detach(),
            )  # recon loss should only affect decoder
            # Latent reconstruction losses
            visual_reconstructed = obs_reconstructed["visual"]
            #self.print(f"visual_reconstructed.shape: {visual_reconstructed.shape}, obs['visual'].shape: {obs['visual'].shape}")
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
            #self.print(f"act_reconstructed.shape: {act_reconstructed.shape}, act.shape: {act.shape}")
            act_loss = self.emb_criterion(act_reconstructed, act)
            loss_components["act_loss"] = act_loss
            loss = loss + act_loss

            # Proprioception reconstruction loss
            if self.proprio_decoder is not None:
                proprio_reconstructed = obs_reconstructed["proprio"]
                #self.print(f"proprio_reconstructed.shape: {proprio_reconstructed.shape}, obs['proprio'].shape: {obs['proprio'].shape}")
                proprio_recon_loss = self.decoder_criterion(proprio_reconstructed, obs['proprio'])
                loss_components["proprio_recon_loss"] = proprio_recon_loss
                loss = loss + proprio_recon_loss
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
        # add patch dimension for proprio and act
        obs_0['proprio'] = rearrange(obs_0['proprio'], "b t d -> b t 1 d")
        act = rearrange(act, "b t d -> b t 1 d")

        #self.print(f"obs_0['visual'].shape: {obs_0['visual'].shape}, obs_0['proprio'].shape: {obs_0['proprio'].shape}, act.shape: {act.shape}")
        num_obs_init = obs_0['visual'].shape[1]
        act_0 = act[:, :num_obs_init]
        action = act[:, num_obs_init:]
        # prepend the last action of act_0 as this is the action at the current time step
        # and therefore the first action that needs to be taken!
        action = torch.cat([act_0[:, -1:], action], dim=1)
        #self.print(f"obs_0['visual'].shape: {obs_0['visual'].shape}, obs_0['proprio'].shape: {obs_0['proprio'].shape}, act_0.shape: {act_0.shape}, action.shape: {action.shape}")
        o, z = self.encode(obs_0, act_0)
        #self.print(f"Initial o.shape: {o.shape}, z.shape: {z.shape}, u.shape: {u.shape}, action.shape: {action.shape}")
        t = 0
        while t < action.shape[1]:
            #self.print(f"\nRollout at time step {t}/{action.shape[1]}")
            z_pred = self.predict(z[:, -self.num_hist:])
            z_new = z_pred[:, -1:, ...]

            # replace the action embedding in z_new with the current action
            u_now = self.encode_act(action[:, t:t+1, ...])  # (b, 1, 1, 384)
            z_new = torch.cat([
                z_new[:, :, :196, :],
                z_new[:, :, 196:197, :],
                u_now
            ], dim=-2)

            z = torch.cat([z, z_new], dim=1)
            #self.print(f"Rollout step {t}: z.shape: {z.shape}")
            t += 1

        #self.print(f"Final z.shape: {z.shape}")
        # z requires u appended in separate_emb because of the concatenated u_hist
        o, p, u = self.separate_emb(z)
        obss = {'visual': o, 'proprio': p}

        # raise NotImplementedError("rollout decoding not implemented yet")
        return obss, z # TODO: add u to z_obses to analyze it later..
    
    def take_step(self, obs, act):
        """
        Take a step in the environment. The final action in act
        is considered to be the current control input.
        """
        #self.print(f"\n\nVWorldModel take_step:")
        #self.print(f"obs['visual'].shape: {obs['visual'].shape}, obs['proprio'].shape: {obs['proprio'].shape}, act.shape: {act.shape}")
        #self.print(f"obs['visual'].shape: {obs['visual'].shape}, obs['proprio'].shape: {obs['proprio'].shape}, act_0.shape: {act_0.shape}, act_now.shape: {act_now.shape}")

        # Encode the observations, action history, and current action
        o, z, u = self.encode(obs, act)

        # Take the step in the latent space
        z_pred = self.predict(z[:, -self.num_hist :])
        obs_pred, _, _ = self.decode(
            z_pred,
        )  # recon loss should only affect decoder

        # Just return the current observation prediction
        obs_now, _, _ = self.decode(
            z_pred[:, -1:, :],
        )  # recon loss should only affect decoder
        return obs_pred, z_pred, obs_now
    