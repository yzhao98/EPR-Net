import math

import torch
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

from plot import plot_samples


class SimDataset(Dataset):
    def __init__(self, problem):
        self.problem = problem
        self.writer = self.problem.writer
        self.z_dist = torch.distributions.MultivariateNormal(
            torch.zeros(self.problem.dimension).to(self.problem.device),
            torch.eye(self.problem.dimension).to(self.problem.device)).expand((self.problem.sample_size,))
        self.enh_dist = torch.distributions.MultivariateNormal(
            torch.zeros(self.problem.dimension).to(self.problem.device),
            torch.eye(self.problem.dimension).to(self.problem.device)).expand((self.problem.sample_size,))

        if self.problem.resimulate:
            self.x_sde = self.simulate_sde(self.problem.D)
            if self.problem.problem_id == 2010:
                self.x_enh = self.simulate_sde(self.problem.enh_scale * self.problem.D)
                self.problem.sample_size = max(self.x_sde.shape[0], self.x_enh.shape[0])
                print("left data", self.problem.sample_size)
                self.z_dist = torch.distributions.MultivariateNormal(
                    torch.zeros(self.problem.dimension).to(self.problem.device),
                    torch.eye(self.problem.dimension).to(self.problem.device)).expand((self.problem.sample_size,))
            elif self.problem.problem_id == 2008:
                x = self.x_sde.detach() + self.problem.enh_scale * self.enh_dist.sample().detach()
                x[x < self.problem.x_min] = -x[x < self.problem.x_min] + 2 * self.problem.x_min
                x[x > self.problem.x_max] = -x[x > self.problem.x_max] + 2 * self.problem.x_max
                self.x_enh = x
            elif self.problem.problem_id == 202203:
                self.x_enh = self.x_sde.detach() + self.problem.enh_scale * self.enh_dist.sample().detach()
            else:
                self.x_enh = self.simulate_sde(self.problem.enh_scale * self.problem.D)
            torch.save(self.x_sde,
                       f"samples/PID={self.problem.problem_id}-dim={self.problem.dimension}-D={self.problem.D}.pth")
            torch.save(self.x_enh,
                       f"samples/PID={self.problem.problem_id}-dim={self.problem.dimension}-D={self.problem.D}-enh.pth")
        else:
            self.x_sde = torch.load(
                f"samples/PID={self.problem.problem_id}-dim={self.problem.dimension}-D={self.problem.D}.pth",
                map_location="cpu").to(
                self.problem.device)
            self.x_enh = torch.load(
                f"samples/PID={self.problem.problem_id}-dim={self.problem.dimension}-D={self.problem.D}-enh.pth",
                map_location="cpu").to(
                self.problem.device)

        plot_samples(self.problem, self.x_sde, self.x_enh)

    def __getitem__(self, index):
        data_sde = self.x_sde[index, :].detach().cpu()
        data_enh = self.x_enh[index, :].detach().cpu()
        return data_sde, data_enh

    def __len__(self):
        return self.x_sde.shape[0]

    def sde_forward(self, x, D, scale=1.0):
        step = self.problem.delta_t / scale
        z = self.z_dist.sample().detach()
        x = x.detach() + step * self.problem.force(x).detach() + math.sqrt(2 * D * step) * z
        if self.problem.boundary_type == "reflect":
            x[x < self.problem.x_min] = -x[x < self.problem.x_min] + 2 * self.problem.x_min
            x[x > self.problem.x_max] = -x[x > self.problem.x_max] + 2 * self.problem.x_max
        return x.detach()

    def update_sde_data(self):
        self.x_sde = self.sde_forward(self.x_sde, self.problem.D)

    def update_enh_data(self):
        if (self.problem.problem_id == 202203) or (self.problem.problem_id == 2008):
            x = self.x_sde.detach() + self.problem.enh_scale * self.enh_dist.sample().detach()
        else:
            x = self.sde_forward(self.x_enh, self.problem.enh_scale * self.problem.D, self.problem.enh_scale)

        if self.problem.boundary_type == "reflect":
            x[x < self.problem.x_min] = -x[x < self.problem.x_min] + 2 * self.problem.x_min
            x[x > self.problem.x_max] = -x[x > self.problem.x_max] + 2 * self.problem.x_max
        self.x_enh = x

    def simulate_sde(self, D):
        simulate_sample_size = self.problem.sample_size
        x = sample((simulate_sample_size, self.problem.dimension), self.problem.x_min, self.problem.x_max,
                   False).float().to(self.problem.device)
        for _ in tqdm(range(self.problem.max_iter)):
            x = self.sde_forward(x, D)

        if self.problem.problem_id == 2010:
            x = x[~(x > 1.5 * self.problem.x_max).sum(1).bool()]
            x = x[~torch.isnan(x).sum(1).bool()]
            self.problem.sample_size = x.shape[0]
            self.z_dist = torch.distributions.MultivariateNormal(
                torch.zeros(self.problem.dimension).to(self.problem.device),
                torch.eye(self.problem.dimension).to(self.problem.device)).expand((self.problem.sample_size,))

        return x.detach()


def sample(size, low, high, requires_grad=False):
    return Variable(torch.rand(size) * (high - low) + low, requires_grad=requires_grad)
