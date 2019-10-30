# Copyright (c) 2018 Rui Shu
import torch
from torch.distributions import Normal

from codebase import utils as ut
from codebase.models import nns
from torch import nn
from torch.nn import functional as F
import numpy as np

class VAE(nn.Module):
    def __init__(self, nn='v1', name='vae', z_dim=2):
        super().__init__()
        self.name = name
        self.z_dim = z_dim
        # Small note: unfortunate name clash with torch.nn
        # nn here refers to the specific architecture file found in
        # codebase/models/nns/*.py
        nn = getattr(nns, nn)
        self.enc = nn.Encoder(self.z_dim)
        self.dec = nn.Decoder(self.z_dim)

        # Set prior as fixed parameter attached to Module
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)

    def negative_elbo_bound(self, x):
        """
        Computes the Evidence Lower Bound, KL and, Reconstruction costs

        Args:
            x: tensor: (batch, dim): Observations

        Returns:
            nelbo: tensor: (): Negative evidence lower bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute negative Evidence Lower Bound and its KL and Rec decomposition
        #
        # Note that nelbo = kl + rec
        #
        # Outputs should all be scalar
        ################################################################################

        # 1. get latent distribution and one sample.
        m, v = self.enc.encode(x)
        z = ut.sample_gaussian(m, v)

        x_logits = self.dec.decode(z)

        # 2. get KL divergent of q(z|x) and p(z). (assume z belongs to standard guassian distribution)
        pz_m, pz_v = self.z_prior[0], self.z_prior[1]
        kl_loss = ut.kl_normal(m, v, pz_m, pz_v)

        # 3. reconstruct loss, encourage x = x_hat
        r_loss = ut.log_bernoulli_with_logits(x, x_logits)
        nelbo = -1 * (r_loss - kl_loss)
        nelbo, kl, r = nelbo.mean(), kl_loss.mean(), -1 * r_loss.mean()
        return nelbo, kl, r

        ################################################################################
        # End of code modification
        ################################################################################
        return nelbo, kl, rec

    def negative_iwae_bound(self, x, iw):
        """
        Computes the Importance Weighted Autoencoder Bound
        Additionally, we also compute the ELBO KL and reconstruction terms

        Args:
            x: tensor: (batch, dim): Observations
            iw: int: (): Number of importance weighted samples

        Returns:
            niwae: tensor: (): Negative IWAE bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute niwae (negative IWAE) with iw importance samples, and the KL
        # and Rec decomposition of the Evidence Lower Bound
        #
        # Outputs should all be scalar
        ################################################################################
        # m, v = self.enc.encode(x)
        #
        # # expand m to iw samples
        # m_iw = ut.duplicate(m, iw)
        # v_iw = ut.duplicate(v, iw)
        # x_iw = ut.duplicate(x, iw)
        #
        # # sample z [iw]
        # z = ut.sample_gaussian(m_iw, v_iw)
        # x_logits = self.dec.decode(z)
        #
        # # reconstruct loss
        # rec_loss = -ut.log_bernoulli_with_logits(x_iw, x_logits)
        #
        # # kl
        # kl = ut.log_normal(z, m, v) - ut.log_normal(z, self.z_prior[0], self.z_prior[1])
        #
        # # iw nelbo
        # nelbo = kl + rec_loss
        #
        # niwae = -ut.log_mean_exp(-nelbo.reshape(iw, -1), dim=0)
        # niwae, kl, rec = niwae.mean(), kl.mean(), rec_loss.mean()

        m, v = self.enc.encode(x)

        dist = Normal(loc=m, scale=torch.sqrt(v))
        z_iw = dist.rsample(sample_shape=torch.Size([iw]))

        log_z_batch, kl_z_batch = [], []

        # for each z sample
        for i in range(iw):
            recon_logits = self.dec.decode(z_iw[i])
            log_z_batch.append(ut.log_bernoulli_with_logits(x, recon_logits))  # [batch, z_sample]
            kl_z_batch.append(ut.kl_normal(m, v, torch.zeros_like(m), torch.ones_like(v)))

        # aggregate result together
        log_z = torch.stack(log_z_batch, dim=1)
        kl_z = torch.stack(kl_z_batch, dim=1)

        niwae = -ut.log_mean_exp(log_z - kl_z, dim=1).mean(dim=0)

        rec_loss = -torch.mean(log_z, dim=0)  # over batch
        kl = torch.mean(kl_z, dim=0)

        ################################################################################
        # End of code modification
        ################################################################################
        return niwae, kl, rec_loss

    def loss(self, x):
        nelbo, kl, rec = self.negative_elbo_bound(x)
        loss = nelbo

        summaries = dict((
            ('train/loss', nelbo),
            ('gen/elbo', -nelbo),
            ('gen/kl_z', kl),
            ('gen/rec', rec),
        ))

        return loss, summaries

    def sample_sigmoid(self, batch):
        z = self.sample_z(batch)
        return self.compute_sigmoid_given(z)

    def compute_sigmoid_given(self, z):
        logits = self.dec.decode(z)
        return torch.sigmoid(logits)

    def sample_z(self, batch):
        return ut.sample_gaussian(
            self.z_prior[0].expand(batch, self.z_dim),
            self.z_prior[1].expand(batch, self.z_dim))

    def sample_x(self, batch):
        z = self.sample_z(batch)
        return self.sample_x_given(z)

    def sample_x_given(self, z):
        return torch.bernoulli(self.compute_sigmoid_given(z))
