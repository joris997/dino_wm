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
        concat_dim=0,
        num_action_repeat=7,
        num_proprio_repeat=7,
        train_encoder=True,
        train_predictor=False,
        train_decoder=True,
        train_action_decoder=True,
    ):
        super().__init__()
        self.debug = False
        
        self.num_hist = num_hist
        self.num_pred = num_pred
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
        self.emb_dim = self.encoder.emb_dim + (self.action_dim + self.proprio_dim) * (concat_dim) # Not used

        self.Ts = 1/30.0 # TODO: assume 30Hz for now, make it configurable later
        print(f"num_action_repeat: {self.num_action_repeat}")
        print(f"num_proprio_repeat: {self.num_proprio_repeat}")
        print(f"proprio encoder: {proprio_encoder}")
        print(f"action encoder: {action_encoder}")
        print(f"action decoder: {action_decoder}")
        print(f"proprio_dim: {proprio_dim}, after repeat: {self.proprio_dim}")
        print(f"action_dim: {action_dim}, after repeat: {self.action_dim}")
        print(f"emb_dim: {self.emb_dim}")

        self.concat_dim = concat_dim # 0 or 1
        assert concat_dim == 0 or concat_dim == 1, f"concat_dim {concat_dim} not supported."
        print("Model emb_dim: ", self.emb_dim)

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
        if self.debug:
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
        output:  z (tensor): (b, num_frames, num_patches, emb_dim)
        """
        z_dct = self.encode_obs(obs)
        act_emb = self.encode_act(act)
        self.print(f"act_emb.shape: {act_emb.shape}")
        if self.concat_dim == 0:
            z = torch.cat(
                    [z_dct['visual'], z_dct['proprio'].unsqueeze(2)], dim=2 # add as an extra token
                )  # (b, num_frames, num_patches + 2, dim)
            u = act_emb.unsqueeze(2)
        if self.concat_dim == 1:
            proprio_tiled = repeat(z_dct['proprio'].unsqueeze(2), "b t 1 a -> b t f a", f=z_dct['visual'].shape[2])
            proprio_repeated = proprio_tiled.repeat(1, 1, 1, self.num_proprio_repeat)
            act_tiled = repeat(act_emb.unsqueeze(2), "b t 1 a -> b t f a", f=z_dct['visual'].shape[2])
            z = torch.cat(
                [z_dct['visual'], proprio_repeated], dim=3
            )  # (b, num_frames, num_patches, dim + action_dim)
            act_repeated = act_tiled.repeat(1, 1, 1, self.num_action_repeat)
            u = act_repeated
        return z, u
    
    def encode_act(self, act):
        self.print(f"Encoding action shape: {act.shape}")
        act = self.action_encoder(act) # (b, num_frames, action_emb_dim)
        return act

    def decode_act(self, act_emb):
        self.print(f"Decoding action emb shape: {act_emb.shape}")
        act = self.action_decoder(act_emb) # (b, num_frames, action_dim)
        return act
    
    def encode_proprio(self, proprio):
        self.print(f"Encoding proprio shape: {proprio.shape}")
        proprio = self.proprio_encoder(proprio)
        return proprio

    def encode_obs(self, obs):
        """
        input : obs (dict): "visual", "proprio" (b, t, 3, img_size, img_size)
        output:   z (dict): "visual", "proprio" (b, t, num_patches, encoder_emb_dim)
        """
        visual = obs['visual']
        b = visual.shape[0]
        visual = rearrange(visual, "b t ... -> (b t) ...")
        visual = self.encoder_transform(visual)
        visual_embs = self.encoder.forward(visual)
        visual_embs = rearrange(visual_embs, "(b t) p d -> b t p d", b=b)

        proprio = obs['proprio']
        proprio_emb = self.encode_proprio(proprio)
        self.print(f"obs['visual'].shape: {obs['visual'].shape}")
        self.print(f"obs['proprio'].shape: {obs['proprio'].shape}")
        self.print(f"visual_embs.shape: {visual_embs.shape}")
        self.print(f"proprio_emb.shape: {proprio_emb.shape}")
        return {"visual": visual_embs, "proprio": proprio_emb}

    def predict(self, z, u):  # in embedding space
        """
        input : z: (b, num_hist, num_patches, emb_dim), u: (b, num_hist, num_patches, action_emb_dim)
        output: z: (b, num_hist, num_patches, emb_dim)
        """
        self.print(f"z.shape (before): {z.shape}, u.shape: {u.shape}")

        # split u
        z = z[:, 1:, :, :] # only use N-1 frames to make it equal to size u_hist
        u_hist, u_now = u[:, :-1, :, :], u[:, -1:, :, :]

        # reshape to a batch of windows of inputs
        z      = rearrange(z, "b t p d -> b (t p) d")
        u_hist = rearrange(u_hist, "b t p d -> b (t p) d")
        u_now  = rearrange(u_now, "b t p d -> b (t p) d")        

        # append u_hist to z
        self.print(f"z.shape: {z.shape}, u_hist.shape: {u_hist.shape}, u_now.shape: {u_now.shape}")
        z = torch.cat([z, u_hist], dim=-1)
        self.print(f"z.shape (after concat u_hist): {z.shape}")
        # (b, num_hist * num_patches per img, emb_dim)
        dz = self.predictor(z, u_now)
        zp1 = z + self.Ts*dz

        # reshape back to (b, num_hist, num_patches, emb_dim)
        dz = rearrange(dz, "b (t p) d -> b t p d", t=self.num_hist-1)
        zp1 = rearrange(zp1, "b (t p) d -> b t p d", t=self.num_hist-1)
        self.print(f"dz.shape: {dz.shape}, zp1.shape: {zp1.shape}\n")
        return zp1, dz

    def decode(self, z, u):
        """
        input :   z: (b, num_frames, num_patches, emb_dim)
        output: obs: (b, num_frames, 3, img_size, img_size)
        """
        z_obs, z_act_hist = self.separate_emb(z)
        obs, diff = self.decode_obs(z_obs)
        act = self.decode_act(u[:, -1, -1, None, :]) # (b, num_frames, action_dim)
        return obs, diff, act

    def decode_obs(self, z_obs):
        """
        input :   z: (b, num_frames, num_patches, emb_dim)
        output: obs: (b, num_frames, 3, img_size, img_size)
        """
        b, num_frames, num_patches, emb_dim = z_obs["visual"].shape
        self.print(f"z_obs[visual].shape: {z_obs['visual'].shape}")
        visual, diff = self.decoder(z_obs["visual"])  # (b*num_frames, 3, 224, 224)
        self.print(f"visual.shape (before rearrange): {visual.shape}")
        # visual = rearrange(visual, "(b t) c h w -> b t c h w", t=3)
        visual = rearrange(visual, "(b t) c h w -> b t c h w", t=z_obs["visual"].shape[1])
        self.print(f"visual.shape (after rearrange): {visual.shape}")
        obs = {
            "visual": visual,
            "proprio": z_obs["proprio"], # Note: no decoder for proprio for now!
        }
        return obs, diff
    
    def separate_emb(self, z):
        """
        input: z (tensor)
        output: z_obs (dict), z_act (tensor)
        """
        if self.concat_dim == 0:
            z_visual, z_proprio, z_act = z[:, :, :-2, :], z[:, :, -2, :], z[:, :, -1, :]
        elif self.concat_dim == 1:
            self.print(f"\n\n Separate embedding")
            self.print(f"z.shape: {z.shape}")
            z_visual, z_proprio, z_act = z[..., :-(self.proprio_dim*self.num_proprio_repeat + self.action_dim*self.num_action_repeat)], \
                                         z[..., -(self.proprio_dim*self.num_proprio_repeat + self.action_dim*self.num_action_repeat) :-self.action_dim*self.num_action_repeat],  \
                                         z[..., -self.action_dim*self.num_action_repeat:]
            self.print(f"z_visual.shape: {z_visual.shape}")
            self.print(f"z_proprio.shape: {z_proprio.shape}")
            self.print(f"z_act.shape: {z_act.shape}")
            # # remove tiled dimensions
            # z_visual = rearrange(z_visual, 'b (t p) d -> b t p d', t=self.num_hist-1)
            # z_proprio = rearrange(z_proprio, 'b (t p) d -> b t p d', t=self.num_hist-1)
            # z_act = rearrange(z_act, 'b (t p) d -> b t p d', t=self.num_hist-1)
            # self.print(f"After rearrange:")
            # self.print(f"z_visual.shape: {z_visual.shape}")
            # self.print(f"z_proprio.shape: {z_proprio.shape}")
            # self.print(f"z_act.shape: {z_act.shape}\n\n")

        z_obs = {"visual": z_visual, "proprio": z_proprio}
        return z_obs, z_act

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
        loss = 0
        loss_components = {}
        z, u = self.encode(obs, act)
        z_src = z[:, : self.num_hist, :, :]  # (b, num_hist, num_patches, dim)
        z_tgt = z[:, self.num_pred:self.num_pred + self.num_hist-1, :, :]  # (b, num_hist, num_patches, dim)
        u_src = u[:, : self.num_hist, :, :]  # (b, num_hist, action_dim)
        u_tgt = u[:, self.num_pred :, :, :]  # (b, num_hist, action_dim)
        visual_src = obs['visual'][:, : self.num_hist, ...]  # (b, num_hist, 3, img_size, img_size)
        #TODO: why this size? should just predict 1 frame
        visual_tgt = obs['visual'][:, self.num_pred:self.num_pred + self.num_hist-1, ...]  # (b, num_hist, 3, img_size, img_size)

        if self.predictor is not None:
            z_pred, dz_pred = self.predict(z_src, u_src)
            if self.decoder is not None:
                self.print(f"GOING DECODING")
                self.print(f"z_pred.shape: {z_pred.shape}, u_src.shape: {u_src.shape}")
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

            # Compute loss for visual, proprio dims (i.e. exclude action dims)
            if self.concat_dim == 0:
                z_visual_loss = self.emb_criterion(z_pred[:, :, :-2, :], z_tgt[:, :, :-2, :].detach())
                z_proprio_loss = self.emb_criterion(z_pred[:, :, -2, :], z_tgt[:, :, -2, :].detach())
                z_loss = self.emb_criterion(z_pred[:, :, :-1, :], z_tgt[:, :, :-1, :].detach())
            elif self.concat_dim == 1:
                # reshape z_pred and dz_pred to remove the actions
                z_pred = z_pred[..., : - self.action_dim]
                dz_pred = dz_pred[..., : - self.action_dim]

                self.print(f"z_pred.shape: {z_pred.shape}, z_tgt.shape: {z_tgt.shape}")
                z_visual_loss = self.emb_criterion(
                    z_pred[:, :, :, :-(self.proprio_dim)], \
                    z_tgt[:, :, :, :-(self.proprio_dim)].detach()
                )
                z_proprio_loss = self.emb_criterion(
                    z_pred[:, :, :, -(self.proprio_dim):], 
                    z_tgt[:, :, :, -(self.proprio_dim):].detach()
                )
                z_loss = self.emb_criterion(
                    z_pred[:, :, :, :], 
                    z_tgt[:, :, :, :].detach()
                )

            loss = loss + z_loss
            loss_components["z_loss"] = z_loss
            loss_components["z_visual_loss"] = z_visual_loss
            loss_components["z_proprio_loss"] = z_proprio_loss
        else:
            visual_pred = None
            z_pred = None

        if self.decoder is not None:
            self.print(f"GOING DECODING FULL")
            # attach u to z as the decoder also needs the action history
            z = torch.cat([z, u], dim=-1)
            z = z[:, :3, :, :]
            u = u[:, :4, :, :]
            self.print(f"z.shape: {z.shape}, u.shape: {u.shape}")
            obs_reconstructed, diff_reconstructed, act_reconstructed = self.decode(
                z.detach(),
                u.detach()
            )  # recon loss should only affect decoder
            # Latent reconstruction losses
            visual_reconstructed = obs_reconstructed["visual"]
            self.print(f"visual_reconstructed.shape: {visual_reconstructed.shape}, obs['visual'].shape: {obs['visual'].shape}")
            recon_loss_reconstructed = self.decoder_criterion(visual_reconstructed, obs['visual'][:, :self.num_hist-1, ...])
            decoder_loss_reconstructed = (
                recon_loss_reconstructed
                + self.decoder_latent_loss_weight * diff_reconstructed
            )

            loss_components["decoder_recon_loss_reconstructed"] = (
                recon_loss_reconstructed
            )
            loss_components["decoder_vq_loss_reconstructed"] = diff_reconstructed
            loss_components["decoder_loss_reconstructed"] = (
                decoder_loss_reconstructed
            )
            loss = loss + decoder_loss_reconstructed

            # Control reconstruction loss
            self.print(f"act_reconstructed.shape: {act_reconstructed.shape}, act.shape: {act.shape}")
            act_loss = self.emb_criterion(act_reconstructed, act)
            loss_components["act_loss"] = act_loss
            loss = loss + act_loss
        else:
            visual_reconstructed = None

        if self.action_decoder is not None:
            _, z_act = self.separate_emb(z)
            self.print(f"DECODING ACTION FROM EMB SHAPE: {z_act.shape}")
            act_reconstructed = self.decode_act(z_act[:, -1, -1, None, :]) # (b, num_frames, action_dim)
            self.print(f"act_reconstructed.shape: {act_reconstructed.shape}, act.shape: {act.shape}")
            act_loss = self.emb_criterion(act_reconstructed, act[:, -1, None, :])
            loss = loss + act_loss
            loss_components["act_loss"] = act_loss

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
        self.print(f"obs_0['visual'].shape: {obs_0['visual'].shape}, obs_0['proprio'].shape: {obs_0['proprio'].shape}, act.shape: {act.shape}")
        num_obs_init = obs_0['visual'].shape[1]
        act_0 = act[:, :num_obs_init]
        action = act[:, num_obs_init:] 
        z, u = self.encode(obs_0, act_0)
        self.print(f"Initial z.shape: {z.shape}, u.shape: {u.shape}, action.shape: {action.shape}")
        t = 0
        inc = 1
        while t < action.shape[1]:
            z_pred, _ = self.predict(z[:, -self.num_hist :], u[:, -self.num_hist :])
            z_new = z_pred[:, -inc:, ...]
            u_new = u[:, t : t + inc, ...]
            # remove u from z_new
            z_new = z_new[..., :-self.action_dim]
            z = torch.cat([z, z_new], dim=1)
            u = torch.cat([u, u_new], dim=1)
            print(f"Rollout step {t}: z.shape: {z.shape}, u.shape: {u.shape}")
            t += inc

        z_pred, _ = self.predict(z[:, -self.num_hist :], u[:, -self.num_hist :])
        z_new = z_pred[:, -1 :, ...] # take only the next pred
        u_new = u[:, -1 :, ...]
        # remove u from z_new
        z_new = z_new[..., :-self.action_dim]
        z = torch.cat([z, z_new], dim=1)
        u = torch.cat([u, u_new], dim=1)

        self.print(f"Final z.shape: {z.shape}, u.shape: {u.shape}")
        # z requires u appended in separate_emb because of the concatenated u_hist
        z_obses, z_act = self.separate_emb(torch.cat([z, u], dim=-1))
        return z_obses, z # TODO: add u to z_obses to analyze it later..

