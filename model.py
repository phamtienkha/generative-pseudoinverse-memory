import torch
from torch import nn
from cnn_resnet import Encoder, Decoder
from memory import GPM

class Model(nn.Module):
    def __init__(self, input_size, memory_size, code_size, direct_writing, ordering, pseudoinverse_approx_step):
        super(Model, self).__init__()
        self._input_size = input_size
        self._memory_size = memory_size
        self._code_size = code_size
        self._direct_writing = direct_writing
        self._ordering = ordering
        self._pseudoinverse_approx_step = pseudoinverse_approx_step

        self.encoder_cnn = Encoder(base_channel=16, original_channel=1, input_encoded_size=self._code_size)
        self.decoder_cnn = Decoder(base_channel=16, original_channel=1, latent_size=self._code_size)
        self.memory = GPM(code_size=self._code_size,
                          memory_size=self._memory_size,
                          direct_writing=self._direct_writing,
                          ordering=self._ordering,
                          pseudoinverse_approx_step=self._pseudoinverse_approx_step)

    def encode(self, input):
        input = input.reshape(input.shape[0], 1, 28, 28)
        z = self.encoder_cnn(input)
        return z

    def decode(self, input):
        input_recon = self.decoder_cnn(input)
        input_recon = input_recon.reshape(input.shape[0], -1)
        input_recon = torch.sigmoid(input_recon)
        return input_recon

    def write(self, input_encoded):
        posterior_memory, dkl_M = self.memory.write_to_memory(input_encoded)
        return posterior_memory, dkl_M

    def read(self, input_encoded, posterior_memory):
        z, dkl_w = self.memory.read_with_encoded_input(input_encoded=input_encoded,
                                                       memory_state=posterior_memory)
        return z, dkl_w

    def forward(self, input, episode_size):
        batch_size = input.shape[0] // episode_size
        input_encoded = self.encode(input)
        input_encoded = input_encoded.reshape(episode_size, batch_size, self._code_size)

        if self._ordering:
            input_encoded = self.memory._z_attention(input_encoded)

        posterior_memory, dkl_M = self.write(input_encoded)
        z, dkl_w = self.read(input_encoded, posterior_memory)
        z = z.reshape(episode_size * batch_size, self._code_size)
        input_recon = self.decode(z.reshape(episode_size * batch_size, self._code_size))

        recon_loss = -torch.mean(torch.sum(input * torch.log(input_recon + 1e-12), dim=1) +
                                 torch.sum((1 - input) * torch.log(1 - input_recon + 1e-12), dim=1))
        bijective_loss = self._bijective_loss(input=input,
                                              input_encoded=input_encoded.reshape(episode_size * batch_size, self._code_size))
        return input_recon, (recon_loss, dkl_w, dkl_M, bijective_loss)

    def _bijective_loss(self, input, input_encoded):
        input_recon = self.decode(input_encoded)
        bijective_loss = -torch.mean(torch.sum(input * torch.log(input_recon + 1e-12), dim=1) +
                                     torch.sum((1 - input) * torch.log(1 - input_recon + 1e-12), dim=1))
        return bijective_loss
    
        def denoise(self, input, input_noise, episode_size, iterative_step):
        batch_size = input.shape[0] // episode_size
        input_encoded = self.encode(input)
        input_encoded = input_encoded.reshape(episode_size, batch_size, self._code_size)

        input_encoded_noise = self.encode(input_noise)
        input_encoded_noise = input_encoded_noise.reshape(episode_size, batch_size, self._code_size)

        posterior_memory, dkl_M = self.write(input_encoded=input_encoded)

        input_recon_list = []
        z, dkl_w = self.read(input_encoded_noise, posterior_memory)
        z = z.reshape(episode_size * batch_size, self._code_size)
        input_recon = self.decode(z.reshape(episode_size * batch_size, self._code_size))
        input_recon = self._binarize(input=input_recon, threshold=0.1)
        input_recon_list.append(input_recon)

        for _ in range(iterative_step - 1):
            input_recon_encoded = self.encode(input_recon)
            input_recon_encoded = input_recon_encoded.reshape(episode_size, batch_size, self._code_size)

            z, dkl_w = self.read(input_recon_encoded, posterior_memory)
            z = z.reshape(episode_size * batch_size, self._code_size)
            input_recon = self.decode(z.reshape(episode_size * batch_size, self._code_size))
            input_recon = self._binarize(input=input_recon, threshold=0.1)
            input_recon_list.append(input_recon)

        input_recon = input_recon_list[-1]
        recon_loss = -torch.mean(torch.sum(input * torch.log(input_recon + 1e-12), dim=1) +
                                 torch.sum((1 - input) * torch.log(1 - input_recon + 1e-12), dim=1))
        
        return input_recon_list

    def generate(self, input, episode_size, sample_size, iterative_step, threshold=0.3):
        batch_size = input.shape[0] // episode_size
        input_encoded = self.encode(input)
        input_encoded = input_encoded.reshape(episode_size, batch_size, self._code_size)

        posterior_memory, dkl_M = self.write(input_encoded=input_encoded)
        M = posterior_memory[0]
        w = torch.randn(sample_size, batch_size, self._memory_size)
        z = torch.bmm(w.transpose(0, 1), M).transpose(0, 1)
        z = z.reshape(sample_size * batch_size, self._code_size)

        sample_list = []
        sample = self.decode(z.reshape(sample_size * batch_size, self._code_size))
        sample = self._binarize(input=sample, threshold=0.05)
        sample_list.append(sample)

        for _ in range(iterative_step - 1):
            sample_encoded = self.encode(sample)
            sample_encoded = sample_encoded.reshape(sample_size, batch_size, self._code_size)

            z, dkl_w = self.read(sample_encoded, posterior_memory)
            z = z.reshape(sample_size * batch_size, self._code_size)
            sample = self.decode(z.reshape(sample_size * batch_size, self._code_size))
            sample = self._binarize(input=sample, threshold=threshold)
            sample_list.append(sample)

        return sample_list

    def _binarize(self, input, threshold=0.3):
        input_binarized = torch.zeros_like(input)
        for i in range(input.shape[0]):
            for j in range(input.shape[1]):
                if input[i, j] >= threshold:
                    input_binarized[i, j] = 1
        return input_binarized



