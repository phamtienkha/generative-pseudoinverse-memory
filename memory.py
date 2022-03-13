import torch
from torch import nn
import numpy as np

EPSILON = 1e-6

class GPM(nn.Module):
    def __init__(self,
                 code_size,
                 memory_size,
                 direct_writing,
                 ordering,
                 pseudoinverse_approx_step=3,
                 observation_noise_std=0.5):

        super(GPM, self).__init__()
        self._input_encoded_size = code_size
        self._memory_size = memory_size # K
        self._code_size = code_size # C
        self._observation_noise_std = observation_noise_std
        self._direct_writing = direct_writing
        self._ordering = ordering
        self._pseudoinverse_approx_step = pseudoinverse_approx_step

        # Related to w
        self.w_logvar = nn.Parameter(torch.from_numpy(np.array([0.])).float(), requires_grad=True)

        # Prior params for memory
        self.memory_logvar = nn.Parameter(torch.from_numpy(np.array([0.])).float(), requires_grad=True)
        self.memory_mean = nn.Parameter(torch.randn(self._memory_size, self._code_size), requires_grad=True)

        # Related to approximate writing
        self.ben_cohen_init = nn.Parameter(torch.from_numpy(np.array([-5.])).float(), requires_grad=True)
        self.ben_cohen_memory_init = nn.Parameter(torch.from_numpy(np.array([-5.])).float(), requires_grad=True)

        # Related to ordering
        self.lstm_z = nn.LSTM(input_size=self._code_size, hidden_size=self._code_size//2, num_layers=2, bidirectional=True)

    def _get_prior_params(self):
        _prior_var = torch.ones(self._memory_size).cuda() * torch.exp(self.memory_logvar) + EPSILON # of size (memory_size)
        prior_cov = torch.diag(_prior_var) # of size (memory_size, memory_size)
        return self.memory_mean, prior_cov

    def _get_prior_state(self, batch_size):
        """Return the prior distribution of memory as a tuple."""

        prior_mean, prior_cov = self._get_prior_params()  # prior_mean: (memory_size, code_size), prior_cov: (memory_size, memory_size)
        batch_prior_mean = torch.cat([prior_mean.unsqueeze(0)] * batch_size, dim=0)  # of size (batch_size, memory_size, code_size)
        batch_prior_cov = torch.cat([prior_cov.unsqueeze(0)] * batch_size, dim=0)  # of size (batch_size, memory_size, memory_size)

        return (batch_prior_mean, batch_prior_cov)

    def _sample_M(self, memory_state):
        """

        :param memory_state: tuple of (memory_mean, memory_covariance_matrix)
        :return: memory_mean
        """
        return memory_state[0]

    def _update_memory(self, old_memory, w, z):
        """
        Setting new_z_var=0 for sample based update.

        Args:
          old_mean: of size (batch_size, memory_size, code_size)
          old_cov: of size (batch_size, memory_size, memory_size)
          w: of size (1, batch_size, memory_size)
          z: of size (1, batch_size, code_size)
        """

        old_mean, old_cov = old_memory

        Delta = z - torch.bmm(w.transpose(0, 1), old_mean).transpose(0, 1) # of size (1, batch_size, code_size)
        wU = torch.bmm(w.transpose(0, 1), old_cov).transpose(0, 1) # of size (1, batch_size, memory_size)
        wUw = torch.bmm(wU.transpose(0, 1), w.transpose(0, 1).transpose(1, 2)).transpose(0, 1)  # of size (1, batch_size, 1)

        sigma_z = wUw + self._observation_noise_std**2 * \
                  torch.cat([torch.eye(w.shape[0]).unsqueeze(0)] * w.shape[1], dim=0).transpose(0, 1).cuda() # of size (1, batch_size, 1)
        c_z = wU / sigma_z  # of size (1, batch_size, memory_size)

        posterior_mean = old_mean + torch.bmm(c_z.transpose(0, 1).transpose(1, 2),
                                              Delta.transpose(0, 1))  # of size (batch_size, memory_size, code_size)
        posterior_cov = old_cov - torch.bmm(c_z.transpose(0, 1).transpose(1, 2),
                                            wU.transpose(0, 1))  # of size (batch_size, memory_size, memory_size)

        new_memory = (posterior_mean, posterior_cov)
        return new_memory

    def _update_state(self, z, memory_state):
        """

        :param z: (episode_size, batch_size, code_size)
        :param w: (episode_size, batch_size, memory_size)
        :param memory_state: tuple of (memory_mean, memory_cov)

        :return:
            final_memory: A tuple containing the new memory state after the update.

        """

        episode_size, batch_size = list(z.shape)[:2]

        if not self._direct_writing:
            new_memory = memory_state
            for i in range(episode_size):
              z_step = z[i].unsqueeze(0) # of size (1, batch_size, code_size)
              w_step = self._solve_w_mean(z=z_step, M=new_memory[0])
              new_memory = self._update_memory(old_memory=new_memory, w=w_step, z=z_step)
        else:
            noise = torch.randn_like(z) * self._observation_noise_std
            z_noise = z + noise

            w = self._solve_w_mean(z=z_noise, M=memory_state[0], pseudoinverse=True) # of size (episode_size, batch_size, memory_size)
            # w = torch.randn(episode_size, batch_size, self._memory_size).cuda()
            w_pseudo_inverse = self._approx_pseudo_inverse(w.transpose(0, 1), iterative_step=self._pseudoinverse_approx_step)

            new_M_mean = torch.bmm(w_pseudo_inverse, z_noise.transpose(0, 1))  # of size (batch_size, memory_size, code_size)
            new_memory = (new_M_mean, memory_state[1])

        final_memory = new_memory
        return final_memory

    def write_to_memory(self, input_encoded):
        """

        :param input_encoded: of size (episode_size, batch_size, input_encoded_size)
        :return: updated memory
        """
        batch_size = input_encoded.shape[1]
        prior_memory = self._get_prior_state(batch_size)

        posterior_memory = self._update_state(z=input_encoded, memory_state=prior_memory)

        dkl_M = self._dkl_M(prior_memory=prior_memory, posterior_memory=posterior_memory)
        # dkl_M = torch.zeros(1).cuda()
        return posterior_memory, dkl_M

    def read_with_encoded_input(self, input_encoded, memory_state):
        """
        :param input_encoded: of size (episode_size, batch_size, input_encoded_size)
        :param memory_state: tuple of (memory_mean, memory_cov)
        :return:
            z: of size (episode_size, batch_size, code_size)
            dkl_y: KL div. of y w.r.t the isotropic Gaussian
            dkl_z: KL div of z w.r.t the Gaussian (z_prior_mean, 1)
        """
        episode_size, batch_size = list(input_encoded.shape)[:2]

        M = self._sample_M(memory_state) # of size (batch_size, memory_size, code_size)

        w_mean = self._solve_w_mean(z=input_encoded, M=M, pseudoinverse=True)
        w = self._sample_w(w_mean=w_mean) # of size (episode_size, batch_size, memory_size)
        dkl_w = self._dkl_w(w_mean=w_mean)

        z_mean = torch.bmm(w.transpose(0, 1), M).transpose(0, 1) # of size (episode_size, batch_size, code_size)
        z = z_mean + self._observation_noise_std * torch.randn_like(z_mean)
        return z, dkl_w

    def _z_attention(self, input_encoded):
        z_lstm, _ = self.lstm_z(input_encoded) # of size (episode_size, batch_size, code_size)
        return z_lstm

    def _solve_w_mean(self, z, M, pseudoinverse=False):
        """
        :param z: of size (_, batch_size, code_size)
        :param M: of size (batch_size, memory_size, code_size)
        :return: w_mean of size (_, batch_size, memory_size)
        """
        batch_size = M.shape[0]

        if not pseudoinverse:
            z = z.transpose(0, 1).transpose(1, 2)  # of size (batch_size, code_size, _)
            batch_identity = torch.cat([torch.eye(self._memory_size).unsqueeze(0)] * batch_size, dim=0).cuda() # of size (batch_size, memory_size, memory_size)
            temp = torch.bmm(torch.inverse(torch.bmm(M, M.transpose(1, 2)) + self._observation_noise_std**2 * batch_identity), M) # of size (batch_size, memory_size, code_size)
            w_mean = torch.bmm(temp, z) # of size (batch_size, memory_size, _)
            w_mean = w_mean.transpose(0, 2).transpose(1, 2) # of size (_, batch_size, memory_size)
        else:
            z = z.transpose(0, 1)  # of size (batch_size, _, code_size)
            M_pseudoinverse = self._approx_pseudo_inverse(M, iterative_step=self._pseudoinverse_approx_step, memory=True) # of size (batch_size, code_size, memory_size)
            z_noise = z + torch.randn_like(z) * self._observation_noise_std
            w_mean = torch.bmm(z_noise, M_pseudoinverse) # of size (batch_size, _, memory_size)
            w_mean = w_mean.transpose(0, 1)
        return w_mean

    def _sample_w(self, w_mean):
        std = torch.exp(0.5 * self.w_logvar)
        w_sample = w_mean + std * torch.randn_like(std)
        return w_sample

    def _dkl_w(self, w_mean):
        w_mean_prior = 0
        dkl_w = torch.mean(0.5 * torch.sum(torch.exp(self.w_logvar) + (w_mean - w_mean_prior) ** 2
                                           - 1 - self.w_logvar, -1), dim=[0, 1])
        return dkl_w

    def _dkl_M(self, prior_memory, posterior_memory):
        R_prior, U_prior = prior_memory
        R, U = posterior_memory

        p_diag = torch.diagonal(U_prior, dim1=-2, dim2=-1)
        q_diag = torch.diagonal(U, dim1=-2, dim2=-1)  # B, K

        t1 = self._code_size * torch.sum(q_diag / p_diag, dim=-1)
        t2 = torch.sum((R - R_prior) ** 2 / torch.unsqueeze(p_diag, -1), [-2, -1])
        t3 = -self._code_size * self._memory_size
        t4 = self._code_size * torch.sum(torch.log(p_diag) - torch.log(q_diag), -1)

        dkl_M_batch = t1 + t2 + t3 + t4
        dkl_M = torch.mean(dkl_M_batch)
        return dkl_M

    def _approx_pseudo_inverse(self, A, iterative_step=3, memory=False):
        if not memory:
            A_init = min(torch.exp(self.ben_cohen_memory_init), 5e-4) * A
            A_pseudoinverse = A_init.transpose(1, 2)  # of size (batch_size, B, A)
            for i in range(iterative_step):
                A_pseudoinverse = 2 * A_pseudoinverse - torch.bmm(torch.bmm(A_pseudoinverse, A), A_pseudoinverse)
        else:
            A_init = min(torch.exp(self.ben_cohen_memory_init), 5e-4) * A
            A_pseudoinverse = A_init.transpose(1, 2)  # of size (batch_size, B, A)
            for i in range(iterative_step):
                A_pseudoinverse = 2 * A_pseudoinverse - torch.bmm(torch.bmm(A_pseudoinverse, A), A_pseudoinverse)
        return A_pseudoinverse
