import math

import torch
import torch.nn.functional as F

import midi.utils as utils
from midi.diffusion import diffusion_utils


class NoiseModel:
    def __init__(self, cfg):
        self.mapping = ['p', 'x', 'c', 'e', 'y']
        self.inverse_mapping = {m: i for i, m in enumerate(self.mapping)}
        nu = cfg.model.nu
        self.nu_arr = []
        for m in self.mapping:
            self.nu_arr.append(nu[m])

        # Define the transition matrices for the discrete features
        self.Px = None
        self.Pe = None
        self.Py = None
        self.Pcharges = None
        self.X_classes = None
        self.charges_classes = None
        self.E_classes = None
        self.y_classes = None
        self.X_marginals = None
        self.charges_marginals = None
        self.E_marginals = None
        self.y_marginals = None

        self.noise_schedule = cfg.model.diffusion_noise_schedule
        self.timesteps = cfg.model.diffusion_steps
        self.T = cfg.model.diffusion_steps

        if self.noise_schedule == 'cosine':
            betas = diffusion_utils.cosine_beta_schedule_discrete(self.timesteps, self.nu_arr)#(501, 5)
        else:
            raise NotImplementedError(self.noise_schedule)

        self._betas = torch.from_numpy(betas)
        self._alphas = 1 - torch.clamp(self._betas, min=0, max=0.9999)
        # self._alphas = 1 - self._betas
        log_alpha = torch.log(self._alphas)
        log_alpha_bar = torch.cumsum(log_alpha, dim=0)
        self._log_alpha_bar = log_alpha_bar
        self._alphas_bar = torch.exp(log_alpha_bar)
        self._sigma2_bar = -torch.expm1(2 * log_alpha_bar)
        self._sigma_bar = torch.sqrt(self._sigma2_bar)
        self._gamma = torch.log(-torch.special.expm1(2 * log_alpha_bar)) - 2 * log_alpha_bar
        # print(f"[Noise schedule: {noise_schedule}] alpha_bar:", self.alphas_bar)

    def move_P_device(self, tensor):
        """ Move the transition matrices to the device specified by tensor."""
        return diffusion_utils.PlaceHolder(X=self.Px.float().to(tensor.device),
                                           charges=self.Pcharges.float().to(tensor.device),
                                           E=self.Pe.float().to(tensor.device).float(),
                                           y=self.Py.float().to(tensor.device), pos=None)

    def get_Qt(self, t_int):
        """ Returns one-step transition matrices for X and E, from step t - 1 to step t.
        Qt = (1 - beta_t) * I + beta_t / K

        beta_t: (bs)                         noise level between 0 and 1
        returns: qx (bs, dx, dx), qe (bs, de, de), qy (bs, dy, dy).
        """
        P = self.move_P_device(t_int)
        kwargs = {'device': t_int.device, 'dtype': torch.float32}

        bx = self.get_beta(t_int=t_int, key='x').unsqueeze(1)
        q_x = bx * P.X + (1 - bx) * torch.eye(self.X_classes, **kwargs).unsqueeze(0)

        bc = self.get_beta(t_int=t_int, key='c').unsqueeze(1)
        q_c = bc * P.charges + (1 - bc) * torch.eye(self.charges_classes, **kwargs).unsqueeze(0)

        be = self.get_beta(t_int=t_int, key='e').unsqueeze(1)
        q_e = be * P.E + (1 - be) * torch.eye(self.E_classes, **kwargs).unsqueeze(0)

        by = self.get_beta(t_int=t_int, key='y').unsqueeze(1)
        q_y = by * P.y + (1 - by) * torch.eye(self.y_classes, **kwargs).unsqueeze(0)

        return utils.PlaceHolder(X=q_x, charges=q_c, E=q_e, y=q_y, pos=None)

    def get_Qt_bar(self, t_int):
        """ Returns t-step transition matrices for X and E, from step 0 to step t.
            Qt = prod(1 - beta_t) * I + (1 - prod(1 - beta_t)) / K

            alpha_bar_t: (bs)         Product of the (1 - beta_t) for each time step from 0 to t.
            returns: qx (bs, dx, dx), qe (bs, de, de), qy (bs, dy, dy).
        """
        a_x = self.get_alpha_bar(t_int=t_int, key='x').unsqueeze(1)
        a_c = self.get_alpha_bar(t_int=t_int, key='c').unsqueeze(1)
        a_e = self.get_alpha_bar(t_int=t_int, key='e').unsqueeze(1)
        a_y = self.get_alpha_bar(t_int=t_int, key='y').unsqueeze(1)

        P = self.move_P_device(t_int)
        # [X, charges, E, y, pos]
        dev = t_int.device
        q_x = a_x * torch.eye(self.X_classes, device=dev).unsqueeze(0) + (1 - a_x) * P.X
        q_c = a_c * torch.eye(self.charges_classes, device=dev).unsqueeze(0) + (1 - a_c) * P.charges
        q_e = a_e * torch.eye(self.E_classes, device=dev).unsqueeze(0) + (1 - a_e) * P.E
        q_y = a_y * torch.eye(self.y_classes, device=dev).unsqueeze(0) + (1 - a_y) * P.y

        assert ((q_x.sum(dim=2) - 1.).abs() < 1e-4).all(), q_x.sum(dim=2) - 1
        assert ((q_e.sum(dim=2) - 1.).abs() < 1e-4).all()

        return utils.PlaceHolder(X=q_x, charges=q_c, E=q_e, y=q_y, pos=None)

    def get_beta(self, t_normalized=None, t_int=None, key=None):
        assert int(t_normalized is None) + int(t_int is None) == 1
        if t_int is None:
            t_int = torch.round(t_normalized * self.T)
        b = self._betas.to(t_int.device)[t_int.long()]
        if key is None:
            return b.float()
        else:
            return b[..., self.inverse_mapping[key]].float()

    def get_alpha_bar(self, t_normalized=None, t_int=None, key=None):
        assert int(t_normalized is None) + int(t_int is None) == 1
        if t_int is None:
            t_int = torch.round(t_normalized * self.T)
        a = self._alphas_bar.to(t_int.device)[t_int.long()]
        if key is None:
            return a.float()
        else:
            return a[..., self.inverse_mapping[key]].float()

    def get_sigma_bar(self, t_normalized=None, t_int=None, key=None):
        assert int(t_normalized is None) + int(t_int is None) == 1
        if t_int is None:
            t_int = torch.round(t_normalized * self.T)
        s = self._sigma_bar.to(t_int.device)[t_int]
        if key is None:
            return s.float()
        else:
            return s[..., self.inverse_mapping[key]].float()

    def get_sigma2_bar(self, t_normalized=None, t_int=None, key=None):
        assert int(t_normalized is None) + int(t_int is None) == 1
        if t_int is None:
            t_int = torch.round(t_normalized * self.T)
        s = self._sigma2_bar.to(t_int.device)[t_int]
        if key is None:
            return s.float()
        else:
            return s[..., self.inverse_mapping[key]].float()

    def get_gamma(self, t_normalized=None, t_int=None, key=None):
        assert int(t_normalized is None) + int(t_int is None) == 1
        if t_int is None:
            t_int = torch.round(t_normalized * self.T)
        g = self._gamma.to(t_int.device)[t_int]
        if key is None:
            return g.float()
        else:
            return g[..., self.inverse_mapping[key]].float()

    def sigma_pos_ts_sq(self, t_int, s_int):
        gamma_s = self.get_gamma(t_int=s_int, key='p')
        gamma_t = self.get_gamma(t_int=t_int, key='p')
        delta_soft = F.softplus(gamma_s) - F.softplus(gamma_t)
        sigma_squared = - torch.expm1(delta_soft)
        return sigma_squared

    # def get_ratio_sigma_ts(self, t_int, s_int):
    #     """ Compute sigma_t_over_s^2 / (1 - sigma_t_over_s^2)"""
    #     delta_soft = F.softplus(self._gamma[t_int]) - F.softplus(self._gamma[s_int])
    #     return torch.expm1(delta_soft)

    def get_alpha_pos_ts(self, t_int, s_int):
        log_a_bar = self._log_alpha_bar[..., self.inverse_mapping['p']].to(t_int.device)
        ratio = torch.exp(log_a_bar[t_int] - log_a_bar[s_int])
        return ratio.float()

    def get_alpha_pos_ts_sq(self, t_int, s_int):
        log_a_bar = self._log_alpha_bar[..., self.inverse_mapping['p']].to(t_int.device)
        ratio = torch.exp(2 * log_a_bar[t_int] - 2 * log_a_bar[s_int])
        return ratio.float()


    def get_sigma_pos_sq_ratio(self, s_int, t_int):
        log_a_bar = self._log_alpha_bar[..., self.inverse_mapping['p']].to(t_int.device)
        s2_s = - torch.expm1(2 * log_a_bar[s_int])
        s2_t = - torch.expm1(2 * log_a_bar[t_int])
        ratio = torch.exp(torch.log(s2_s) - torch.log(s2_t))
        return ratio.float()

    def get_x_pos_prefactor(self, s_int, t_int):
        """ a_s (s_t^2 - a_t_s^2 s_s^2) / s_t^2"""
        a_s = self.get_alpha_bar(t_int=s_int, key='p')
        alpha_ratio_sq = self.get_alpha_pos_ts_sq(t_int=t_int, s_int=s_int)
        sigma_ratio_sq = self.get_sigma_pos_sq_ratio(s_int=s_int, t_int=t_int)
        prefactor = a_s * (1 - alpha_ratio_sq * sigma_ratio_sq)
        return prefactor.float()

    def control_data_add_noise(self, dense_data, noise_std):
        node_mask = dense_data.node_mask
        bs, n = node_mask.shape
        inverse_edge_mask = ~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))
        diag_mask = torch.eye(n).unsqueeze(0).expand(bs, -1, -1)

        if self.control_add_noise_dict['cX']==True:
            cX = F.softmax(dense_data.cX + torch.normal(mean=0, std=noise_std, size=dense_data.cX.shape,
                                                        device=dense_data.cX.device), dim=-1)
            cX[~node_mask] = 1 / cX.shape[-1]
            ccharges = F.softmax(dense_data.ccharges + torch.normal(mean=0, std=noise_std, size=dense_data.ccharges.shape,
                                                                    device=dense_data.ccharges.device), dim=-1)
            ccharges[~node_mask] = 1 / ccharges.shape[-1]
        else:
            cX = dense_data.cX
            ccharges = dense_data.ccharges

        if self.control_add_noise_dict['cE']==True:
            probcE = torch.normal(mean=0, std=noise_std, size=dense_data.cE.shape, device=dense_data.cE.device)
            cE = F.softmax(dense_data.cE + probcE, dim=-1)
            cE[inverse_edge_mask] = 1 / cE.shape[-1]
        elif self.control_add_noise_dict['cE']=='mean':
            cE = torch.full(dense_data.cE.shape, 1 / dense_data.cE.shape[-1], device=dense_data.cE.device)
        else:
            cE = dense_data.cE

        if self.control_add_noise_dict['cpos']==True:
            cpos = dense_data.cpos + torch.normal(mean=0, std=noise_std, size=dense_data.cpos.shape,
                                                  device=dense_data.cpos.device)
            cpos_masked = cpos*dense_data.node_mask.unsqueeze(-1)
            cpos = utils.remove_mean_with_mask(cpos_masked, dense_data.node_mask)
        else:
            cpos = dense_data.cpos

        # # *************************************#
        # ######cX, ccharge add noise#####
        # # cX = F.softmax(dense_data.cX + torch.normal(mean=0, std=0.3, size=dense_data.cX.shape, device=dense_data.cX.device), dim=-1)
        # # cX[~node_mask] = 1 / cX.shape[-1]
        # # ccharges = F.softmax(dense_data.ccharges + torch.normal(mean=0, std=0.3, size=dense_data.ccharges.shape, device=dense_data.ccharges.device), dim=-1)
        # # ccharges[~node_mask] = 1 / ccharges.shape[-1]
        # ######cX, ccharge not noise#####
        # cX = dense_data.cX
        # ccharges = dense_data.ccharges
        # #*************************************#
        #
        # ######cE add noise#####
        # # probcE = torch.normal(mean=0, std=0.3, size=dense_data.cE.shape, device=dense_data.cE.device)
        # # cE = F.softmax(dense_data.cE + probcE, dim=-1)
        # # cE[inverse_edge_mask] = 1 / cE.shape[-1]
        # ######cE not add noise#####
        # cE = dense_data.cE
        # ######all cE=None,  probcE=1 / cE.shape[-1]#####
        # # cE = torch.full(dense_data.cE.shape, 1 / dense_data.cE.shape[-1], device=dense_data.cE.device)
        # # *************************************#
        #
        # ######cpos add noise#####
        # # cpos = dense_data.cpos + torch.normal(mean=0, std=0.3, size=dense_data.cpos.shape, device=dense_data.cpos.device)
        # # cpos_masked = cpos*dense_data.node_mask.unsqueeze(-1)
        # # cpos = utils.remove_mean_with_mask(cpos_masked, dense_data.node_mask)
        # ######cE not add noise#####
        # cpos = dense_data.cpos
        # # *************************************#

        control_data = utils.PlaceHolder(X=dense_data.X, charges=dense_data.charges, E=dense_data.E, y=dense_data.y, pos=dense_data.pos,
                                         cX=cX, ccharges=ccharges, cE=cE, cy=dense_data.cy,
                                         cpos=cpos, node_mask=dense_data.node_mask).mask()
        return control_data


    def apply_noise(self, dense_data, noise_std, fixed_time = None):
        """ Sample noise and apply it to the data. """
        device = dense_data.X.device
        if fixed_time is None:
            t_int = torch.randint(1, self.T + 1, size=(dense_data.X.size(0), 1), device=device)
        else:
            t_int = torch.full((dense_data.X.size(0), 1), fixed_time, device=device)
        t_float = t_int.float() / self.T

        # Qtb returns two matrices of shape (bs, dx_in, dx_out) and (bs, de_in, de_out)
        Qtb = self.get_Qt_bar(t_int=t_int)

        # Compute transition probabilities
        probX = dense_data.X @ Qtb.X  # (bs, n, dx_out)
        prob_charges = dense_data.charges @ Qtb.charges
        probE = dense_data.E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out)

        sampled_t = diffusion_utils.sample_discrete_features(probX=probX, probE=probE, prob_charges=prob_charges,
                                                             node_mask=dense_data.node_mask)

        X_t = F.one_hot(sampled_t.X, num_classes=self.X_classes) #torch.Size([64, 17, 5])
        E_t = F.one_hot(sampled_t.E, num_classes=self.E_classes) #torch.Size([64, 17, 5])
        charges_t = F.one_hot(sampled_t.charges, num_classes=self.charges_classes) #torch.Size([64, 17, 3])
        assert (dense_data.X.shape == X_t.shape) and (dense_data.E.shape == E_t.shape)

        noise_pos = torch.randn(dense_data.pos.shape, device=dense_data.pos.device) #torch.Size([64, 17, 3])
        noise_pos_masked = noise_pos * dense_data.node_mask.unsqueeze(-1)
        noise_pos_masked = utils.remove_mean_with_mask(noise_pos_masked, dense_data.node_mask)

        # beta_t and alpha_s_bar are used for denoising/loss computation
        a = self.get_alpha_bar(t_int=t_int, key='p').unsqueeze(-1)
        s = self.get_sigma_bar(t_int=t_int, key='p').unsqueeze(-1)
        pos_t = a * dense_data.pos + s * noise_pos_masked

        # control_data = dense_data
        control_data = self.control_data_add_noise(dense_data, noise_std=noise_std)
        z_t = utils.PlaceHolder(X=X_t, charges=charges_t, E=E_t, y=dense_data.y, pos=pos_t, t_int=t_int,
                                t=t_float, node_mask=dense_data.node_mask,
                                cX=control_data.cX, ccharges=control_data.ccharges, cE=control_data.cE, cy=control_data.cy, cpos=control_data.cpos).mask()
        # z_t = utils.PlaceHolder(X=X_t, charges=charges_t, E=E_t, y=dense_data.y, pos=pos_t, t_int=t_int,
        #                         t=t_float, node_mask=dense_data.node_mask,
        #                         cX=dense_data.cX, ccharges=dense_data.ccharges, cE=dense_data.cE, cy=dense_data.cy, cpos=dense_data.cpos).mask()
        return z_t

    def get_limit_dist(self):
        X_marginals = self.X_marginals + 1e-7
        X_marginals = X_marginals / torch.sum(X_marginals)
        E_marginals = self.E_marginals + 1e-7
        E_marginals = E_marginals / torch.sum(E_marginals)
        charges_marginals = self.charges_marginals + 1e-7
        charges_marginals = charges_marginals / torch.sum(charges_marginals)
        limit_dist = utils.PlaceHolder(X=X_marginals, E=E_marginals, charges=charges_marginals,
                                       y=None, pos=None)
        return limit_dist

    def sample_limit_dist(self, node_mask, template, noise_std):
        """ Sample from the limit distribution of the diffusion process"""

        bs, n_max = node_mask.shape
        x_limit = self.X_marginals.expand(bs, n_max, -1)#torch.Size([20, 16, 5])
        e_limit = self.E_marginals[None, None, None, :].expand(bs, n_max, n_max, -1)#torch.Size([20, 16, 16, 5])
        charges_limit = self.charges_marginals.expand(bs, n_max, -1)#torch.Size([20, 16, 3])

        U_X = x_limit.flatten(end_dim=-2).multinomial(1).reshape(bs, n_max).to(node_mask.device)#torch.Size([20, 16])
        U_c = charges_limit.flatten(end_dim=-2).multinomial(1).reshape(bs, n_max).to(node_mask.device)#torch.Size([20, 16])
        U_E = e_limit.flatten(end_dim=-2).multinomial(1).reshape(bs, n_max, n_max).to(node_mask.device)#torch.Size([20, 16, 16])
        U_y = torch.zeros((bs, 0), device=node_mask.device)#torch.Size([20, 0])

        # #control data
        # cX = (U_X != 0).long()
        # ccharges = torch.zeros_like(U_c)

        U_X = F.one_hot(U_X, num_classes=x_limit.shape[-1]).float()#torch.Size([20, 16, 5])
        U_E = F.one_hot(U_E, num_classes=e_limit.shape[-1]).float()#torch.Size([20, 16, 16, 5])
        U_c = F.one_hot(U_c, num_classes=charges_limit.shape[-1]).float()#torch.Size([20, 16, 3])
        # #control data
        # cX = F.one_hot(cX, num_classes=x_limit.shape[-1]).float()#torch.Size([20, 16, 5])
        # ccharges = F.one_hot(ccharges, num_classes=charges_limit.shape[-1]).float()  # torch.Size([20, 16, 3])
        # cy=U_y.clone()

        # Get upper triangular part of edge noise, without main diagonal
        upper_triangular_mask = torch.zeros_like(U_E)
        indices = torch.triu_indices(row=U_E.size(1), col=U_E.size(2), offset=1)
        upper_triangular_mask[:, indices[0], indices[1], :] = 1

        U_E = U_E * upper_triangular_mask
        U_E = (U_E + torch.transpose(U_E, 1, 2))
        assert (U_E == torch.transpose(U_E, 1, 2)).all()
        # cE = U_E.clone()

        pos = torch.randn(node_mask.shape[0], node_mask.shape[1], 3, device=node_mask.device)
        pos = pos * node_mask.unsqueeze(-1)
        pos = utils.remove_mean_with_mask(pos, node_mask)
        # cpos = pos.clone()

        t_array = pos.new_ones((pos.shape[0], 1))
        t_int_array = self.T * t_array.long()

        # control_data = template
        control_data = self.control_data_add_noise(template, noise_std)
        return utils.PlaceHolder(X=U_X, charges=U_c, E=U_E, y=U_y, pos=pos, t_int=t_int_array, t=t_array,
                                 node_mask=node_mask, cX=control_data.cX, ccharges=control_data.ccharges,
                                 cE=control_data.cE, cy=control_data.cy, cpos=control_data.cpos, idx=template.idx).mask(node_mask)


        # return utils.PlaceHolder(X=U_X, charges=U_c, E=U_E, y=U_y, pos=pos, t_int=t_int_array, t=t_array,
        #                          node_mask=node_mask, cX=template.cX, ccharges=template.ccharges,
        #                          cE=template.cE, cy=template.cy, cpos=template.cpos, idx=template.idx).mask(node_mask)

    def sample_zt_from_zs(self, z_s, t_int):
        """Samples from zt ~ p(zt | zs). Only used during inpainting sampling. """
        bs, n, dxs = z_s.X.shape
        node_mask = z_s.node_mask
        # Retrieve transitions matrix
        Qt = self.get_Qt(t_int)

        # # Sample the positions
        
        z_s_prefactor = torch.square(self._alphas[..., self.inverse_mapping['p']].float().to(z_s.pos.device)[t_int])
        mu = z_s_prefactor.unsqueeze(-1) * z_s.pos             # bs, n, 3

        sampled_pos = torch.randn(z_s.pos.shape, device=z_s.pos.device) * node_mask.unsqueeze(-1)
        noise = utils.remove_mean_with_mask(sampled_pos, node_mask=node_mask)

        noise_prefactor_sq = 1 - z_s_prefactor ** 2
        # noise_prefactor_sq = self.get_sigma2_bar(t_int=t_int, key='p') + torch.square(self._alphas[t_int]).to(t_int.device) ** 2
        noise_prefactor = torch.sqrt(noise_prefactor_sq)
        pos = mu + noise_prefactor.unsqueeze(-1) * noise   # bs, n, 3

        p_t_X = diffusion_utils.compute_batched_overzs_posterior_distribution(X_t=z_s.X, Qt=Qt.X) # bs, n, d_t
        p_t_E = diffusion_utils.compute_batched_overzs_posterior_distribution(X_t=z_s.E, Qt=Qt.E)
        p_t_c = diffusion_utils.compute_batched_overzs_posterior_distribution(X_t=z_s.charges, Qt=Qt.charges)

        p_t_X[torch.sum(p_t_X, dim=-1) == 0] = 1e-5
        prob_X = p_t_X / torch.sum(p_t_X, dim=-1, keepdim=True)  # bs, n, d_t

        p_t_c[torch.sum(p_t_c, dim=-1) == 0] = 1e-5
        prob_c = p_t_c / torch.sum(p_t_c, dim=-1, keepdim=True)  # bs, n, d_t

        p_t_E[torch.sum(p_t_E, dim=-1) == 0] = 1e-5
        prob_E = p_t_E / torch.sum(p_t_E, dim=-1, keepdim=True)  
        prob_E = prob_E.reshape(bs, n, n, z_s.E.shape[-1])

        sampled_t = diffusion_utils.sample_discrete_features(prob_X, prob_E, prob_c, node_mask=z_s.node_mask)

        X_t = F.one_hot(sampled_t.X, num_classes=self.X_classes).float()
        charges_t = F.one_hot(sampled_t.charges, num_classes=self.charges_classes).float()
        E_t = F.one_hot(sampled_t.E, num_classes=self.E_classes).float()

        assert (E_t == torch.transpose(E_t, 1, 2)).all()
        assert (z_s.X.shape == X_t.shape) and (z_s.E.shape == E_t.shape)

        z_t = utils.PlaceHolder(X=X_t, charges=charges_t,
                                E=E_t, y=torch.zeros(z_s.y.shape[0], 0, device=X_t.device), pos=pos,
                                t_int=t_int, t=t_int / self.T, node_mask=node_mask,
                                cX=z_s.cX, ccharges=z_s.ccharges, cE=z_s.cE, cy=z_s.cy, cpos=z_s.cpos
                                ).mask(node_mask)
        return z_t
    
    def sample_zs_from_zt_and_pred(self, z_t, pred, s_int):
        """Samples from zs ~ p(zs | zt). Only used during sampling. """
        bs, n, dxs = z_t.X.shape
        node_mask = z_t.node_mask
        t_int = z_t.t_int

        # Retrieve transitions matrix
        Qtb = self.get_Qt_bar(t_int=t_int)
        Qsb = self.get_Qt_bar(t_int=s_int)
        Qt = self.get_Qt(t_int)

        # # Sample the positions
        sigma_sq_ratio = self.get_sigma_pos_sq_ratio(s_int=s_int, t_int=t_int)
        z_t_prefactor = (self.get_alpha_pos_ts(t_int=t_int, s_int=s_int) * sigma_sq_ratio).unsqueeze(-1)
        x_prefactor = self.get_x_pos_prefactor(s_int=s_int, t_int=t_int).unsqueeze(-1)

        mu = z_t_prefactor * z_t.pos + x_prefactor * pred.pos             # bs, n, 3

        sampled_pos = torch.randn(z_t.pos.shape, device=z_t.pos.device) * node_mask.unsqueeze(-1)
        noise = utils.remove_mean_with_mask(sampled_pos, node_mask=node_mask)

        prefactor1 = self.get_sigma2_bar(t_int=t_int, key='p')
        prefactor2 = self.get_sigma2_bar(t_int=s_int, key='p') * self.get_alpha_pos_ts_sq(t_int=t_int, s_int=s_int)
        sigma2_t_s = prefactor1 - prefactor2
        noise_prefactor_sq = sigma2_t_s * sigma_sq_ratio
        noise_prefactor = torch.sqrt(noise_prefactor_sq).unsqueeze(-1)

        pos = mu + noise_prefactor * noise                                # bs, n, 3

        # Normalize predictions for the categorical features
        pred_X = F.softmax(pred.X, dim=-1)               # bs, n, d0
        pred_E = F.softmax(pred.E, dim=-1)               # bs, n, n, d0
        pred_charges = F.softmax(pred.charges, dim=-1)

        p_s_and_t_given_0_X = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=z_t.X,
                                                                                           Qt=Qt.X,
                                                                                           Qsb=Qsb.X,
                                                                                           Qtb=Qtb.X)

        p_s_and_t_given_0_E = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=z_t.E,
                                                                                           Qt=Qt.E,
                                                                                           Qsb=Qsb.E,
                                                                                           Qtb=Qtb.E)
        p_s_and_t_given_0_c = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=z_t.charges,
                                                                                           Qt=Qt.charges,
                                                                                           Qsb=Qsb.charges,
                                                                                           Qtb=Qtb.charges)

        # Dim of these two tensors: bs, N, d0, d_t-1
        weighted_X = pred_X.unsqueeze(-1) * p_s_and_t_given_0_X         # bs, n, d0, d_t-1
        unnormalized_prob_X = weighted_X.sum(dim=2)                     # bs, n, d_t-1
        unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
        prob_X = unnormalized_prob_X / torch.sum(unnormalized_prob_X, dim=-1, keepdim=True)  # bs, n, d_t-1

        weighted_c = pred_charges.unsqueeze(-1) * p_s_and_t_given_0_c         # bs, n, d0, d_t-1
        unnormalized_prob_c = weighted_c.sum(dim=2)                     # bs, n, d_t-1
        unnormalized_prob_c[torch.sum(unnormalized_prob_c, dim=-1) == 0] = 1e-5
        prob_c = unnormalized_prob_c / torch.sum(unnormalized_prob_c, dim=-1, keepdim=True)  # bs, n, d_t-1

        pred_E = pred_E.reshape((bs, -1, pred_E.shape[-1]))
        weighted_E = pred_E.unsqueeze(-1) * p_s_and_t_given_0_E        # bs, N, d0, d_t-1
        unnormalized_prob_E = weighted_E.sum(dim=-2)
        unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
        prob_E = unnormalized_prob_E / torch.sum(unnormalized_prob_E, dim=-1, keepdim=True)
        prob_E = prob_E.reshape(bs, n, n, pred_E.shape[-1])

        assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all()
        assert ((prob_c.sum(dim=-1) - 1).abs() < 1e-4).all()
        assert ((prob_E.sum(dim=-1) - 1).abs() < 1e-4).all()

        sampled_s = diffusion_utils.sample_discrete_features(prob_X, prob_E, prob_c, node_mask=z_t.node_mask)

        X_s = F.one_hot(sampled_s.X, num_classes=self.X_classes).float()
        charges_s = F.one_hot(sampled_s.charges, num_classes=self.charges_classes).float()
        E_s = F.one_hot(sampled_s.E, num_classes=self.E_classes).float()

        assert (E_s == torch.transpose(E_s, 1, 2)).all()
        assert (z_t.X.shape == X_s.shape) and (z_t.E.shape == E_s.shape)

        z_s = utils.PlaceHolder(X=X_s, charges=charges_s,
                                E=E_s, y=torch.zeros(z_t.y.shape[0], 0, device=X_s.device), pos=pos,
                                t_int=s_int, t=s_int / self.T, node_mask=node_mask,
                                cX=z_t.cX, ccharges=z_t.ccharges, cE=z_t.cE, cy=z_t.cy, cpos=z_t.cpos,
                                idx=z_t.idx
                                ).mask(node_mask)
        return z_s

class DiscreteUniformTransition(NoiseModel):
    def __init__(self, cfg, output_dims):
        super().__init__(cfg=cfg)
        self.X_classes = output_dims.X
        self.charges_classes = output_dims.charges
        self.E_classes = output_dims.E
        self.y_classes = output_dims.y
        self.X_marginals = torch.ones(self.X_classes) / self.X_classes
        self.charges_marginals = torch.ones(self.charges_classes) / self.charges_classes
        self.E_marginals = torch.ones(self.E_classes) / self.E_classes
        self.y_marginals = torch.ones(self.y_classes) / self.y_classes
        self.Px = torch.ones(1, self.X_classes, self.X_classes) / self.X_classes
        self.Pcharges = torch.ones(1, self.charges_classes, self.charges_classes) / self.charges_classes
        self.Pe = torch.ones(1, self.E_classes, self.E_classes) / self.E_classes
        self.Pe = torch.ones(1, self.y_classes, self.y_classes) / self.y_classes
        self.control_add_noise_dict = cfg.dataset.control_add_noise_dict if hasattr(cfg.dataset, "control_add_noise_dict") else {'cX': False, 'cE': False, 'cpos': False}


class MarginalUniformTransition(NoiseModel):
    def __init__(self, cfg, x_marginals, e_marginals, charges_marginals, y_classes):
        super().__init__(cfg=cfg)
        self.X_classes = len(x_marginals)
        self.E_classes = len(e_marginals)
        self.charges_classes = len(charges_marginals)
        self.y_classes = y_classes
        self.X_marginals = x_marginals
        self.E_marginals = e_marginals
        self.charges_marginals = charges_marginals
        self.y_marginals = torch.ones(self.y_classes) / self.y_classes

        self.Px = x_marginals.unsqueeze(0).expand(self.X_classes, -1).unsqueeze(0)
        self.Pe = e_marginals.unsqueeze(0).expand(self.E_classes, -1).unsqueeze(0)
        self.Pcharges = charges_marginals.unsqueeze(0).expand(self.charges_classes, -1).unsqueeze(0)
        self.Py = torch.ones(1, self.y_classes, self.y_classes) / self.y_classes
        self.control_add_noise_dict = cfg.dataset.control_add_noise_dict if hasattr(cfg.dataset, "control_add_noise_dict") else {'cX': False, 'cE': False, 'cpos': False}
