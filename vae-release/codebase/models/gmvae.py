# Copyright (c) 2018 Rui Shu
import numpy as np
import torch
from torch.distributions import Normal

from codebase import utils as ut
from codebase.models import nns
from torch import nn
from torch.nn import functional as F

class GMVAE(nn.Module):
    def __init__(self, nn='v1', z_dim=2, k=500, name='gmvae'):
        super().__init__()
        self.name = name
        self.k = k
        self.z_dim = z_dim
        nn = getattr(nns, nn)
        self.enc = nn.Encoder(self.z_dim)
        self.dec = nn.Decoder(self.z_dim)

        # Mixture of Gaussians prior
        self.z_pre = torch.nn.Parameter(torch.randn(1, 2 * self.k, self.z_dim)
                                        / np.sqrt(self.k * self.z_dim))
        # Uniform weighting
        self.pi = torch.nn.Parameter(torch.ones(k) / k, requires_grad=False)

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
        # To help you start, we have computed the mixture of Gaussians prior
        # prior = (m_mixture, v_mixture) for you, where
        # m_mixture and v_mixture each have shape (1, self.k, self.z_dim)
        #
        # Note that nelbo = kl + rec
        #
        # Outputs should all be scalar
        ################################################################################
        # Compute the mixture of Gaussian prior
        prior = ut.gaussian_parameters(self.z_pre, dim=1)

        m, v = self.enc.encode(x)
        z = ut.sample_gaussian(m, v)
        logits = self.dec.decode(z)
        kl = ut.log_normal(z, m, v) - ut.log_normal_mixture(z, *prior)
        rec = -ut.log_bernoulli_with_logits(x, logits)
        nelbo = kl + rec
        nelbo, kl, rec = nelbo.mean(), kl.mean(), rec.mean()
        return nelbo, kl, rec

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
        # Compute the mixture of Gaussian prior

        prior = ut.gaussian_parameters(self.z_pre, dim=1)
        m, v = self.enc.encode(x)

        dist = Normal(loc=m, scale=torch.sqrt(v))
        z_iw = dist.rsample(sample_shape=torch.Size([iw]))

        log_z_batch, kl_z_batch = [], []

        # for each z sample
        for i in range(iw):
            recon_logits = self.dec.decode(z_iw[i])
            log_z_batch.append(ut.log_bernoulli_with_logits(x, recon_logits))  # [batch, z_sample]
            kl_z_batch.append(ut.log_normal(z_iw[i], m, v) - ut.log_normal_mixture(z_iw[i], prior[0], prior[1]))

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
        m, v = ut.gaussian_parameters(self.z_pre.squeeze(0), dim=0)
        idx = torch.distributions.categorical.Categorical(self.pi).sample((batch,))
        m, v = m[idx], v[idx]
        return ut.sample_gaussian(m, v)

    def sample_x(self, batch):
        z = self.sample_z(batch)
        return self.sample_x_given(z)

    def sample_x_given(self, z):
        return torch.bernoulli(self.compute_sigmoid_given(z))
