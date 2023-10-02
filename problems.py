import numpy as np
import scipy.io as scio
import torch
import torch.distributions as dist
from tensorboardX import SummaryWriter

from plot import get_grid_data


class Problem:
    def __init__(self, model_path, args):
        self.device = args.device
        self.x_max = args.x_max
        self.x_min = args.x_min
        self.sample_size = args.sample_size
        self.batch_size = args.batch_size
        self.dimension = args.dimension
        self.resimulate = args.resimulate
        self.boundary_type = args.boundary_type
        self.num_potential_epochs = args.num_potential_epochs
        self.num_force_epochs = args.num_force_epochs

        self.D = args.D
        self.enh_scale = args.enh_scale

        # misc
        self.args = args
        self.save_ckpt = args.save_ckpt
        self.model_path = model_path
        self.writer = SummaryWriter(f'{self.model_path}')

        # problem
        self.problem_id = args.problem_id
        self.index_1 = 0
        self.index_2 = 1
        self.delta_t = args.delta_t
        self.max_iter = 10000
        self.set_problem()

    def set_problem(self):
        if self.problem_id == 2008:  # bio2d limit cycle
            return self.set_problem_2008()
        elif self.problem_id == 2010:  # bio8d limit cycle
            return self.set_problem_2010()
        elif self.problem_id == 2011:  # bio3d limit cycle
            return self.set_problem_2011()
        elif self.problem_id == 2013:  # bio2d & 52d multi-stable
            return self.set_problem_2013()
        elif self.problem_id == 202201:  # dw2d double well
            return self.set_problem_double_well()
        elif self.problem_id == 202202:  # dw12d gmm
            return self.set_problem_12d_dw()
        elif self.problem_id == 202203:  # 3d Lorenz
            return self.set_problem_lorenz()
        else:
            raise "Not Implemented Problem."

    def set_problem_2008(self):
        self.a = 0.1
        self.b = 0.1
        self.c = 100
        self.epsilon = 0.1
        self.tau_0 = 5.0

        U_Raw = torch.from_numpy(scio.loadmat(f"data/2d_2008_D={self.D}_use.mat")["U"]).to(self.device)
        col_index = 401
        row_index = 301
        U_Raw = U_Raw[:row_index, :col_index].float()  # 601*801
        U_Raw = U_Raw - U_Raw.min()
        self.U_True = U_Raw

        self.max_iter = 100000
        self.delta_t = 0.1

    def set_problem_2010(self):
        # Cycb
        self.k1 = 0.04
        self.k21 = 0.04
        self.k22 = 1
        self.k23 = 1
        # Cdh1
        self.k31 = 1
        self.k32 = 10
        self.k41 = 2
        self.k4 = 35
        # Cdc20T
        self.k51 = 0.005
        self.k52 = 0.2
        self.k6 = 0.1
        # Cdc20A
        self.k7 = 1
        self.k8 = 0.5
        # IE
        self.k9 = 0.1
        self.k10 = 0.02
        # CKI
        self.k11 = 1
        self.k121 = 0.2
        self.k122 = 50
        self.k123 = 100
        # SK
        self.k131 = 0
        self.k132 = 1
        self.k14 = 1
        self.k151 = 1.5
        self.k152 = 0.05
        self.k161 = 1
        self.k162 = 3
        # M
        self.mu_p = 0.01

        # Dimensionless constant
        self.Cycb = 0.1
        # Cdh1
        self.J3 = 0.04
        self.J4 = 0.04
        # Cdc20T
        self.J5 = 0.3
        self.n = 4
        # Cdc20A
        self.J7 = 0.001
        self.J8 = 0.001
        self.Mad = 1
        # CKI
        self.Keq = 1000
        # SK
        self.J15 = 0.01
        self.J16 = 0.01
        # M
        self.mx = 10
        self.m = 0.8

        self.index_1 = 0
        self.index_2 = 2
        self.max_iter = 50000
        self.delta_t = 0.0005

    def set_problem_2011(self):
        self.alpha1 = 0.1
        self.alpha2 = 3
        self.alpha3 = 3
        self.beta1 = 3
        self.beta2 = 1
        self.beta3 = 1
        self.K1 = 0.5
        self.K2 = 0.5
        self.K3 = 0.5
        self.n1 = 8
        self.n2 = 8
        self.n3 = 8
        self.delta_t = 0.01
        return

    def set_problem_2013(self):
        file = f'data/{self.dimension}d_matrix.csv'
        if self.dimension == 2:
            self.S = 0.5
            self.n = 4
            self.k = 1.0
            self.a = 1.0
            self.b = 1.0
            U_Raw = torch.from_numpy(scio.loadmat(f"data/2d_2011_D={self.D}_use.mat")["U"]).to(self.device)  # 121*121
            U_True = U_Raw.float().reshape(-1, 1)
            self.U_True = U_True - U_True.min()
        elif self.dimension == 52:
            self.S = 0.5
            self.n = 3
            self.k = 1.0
            self.a = 0.37
            self.b = 0.5
            self.delta_t = 0.1
            self.max_iter = 5000
            # GATA6 & NANOG
            self.index_1 = 16
            self.index_2 = 31
        else:
            raise NotImplementedError(
                f"Only support 2d and 52d for problem 2013.")  # though other dimensions may also be possible.

        self.S_n = self.S ** self.n
        with open(file, encoding='utf-8-sig') as f:
            self.matrix = np.loadtxt(f, delimiter=",")

        data = torch.tensor(self.matrix).float()
        self.matrix_act = (data == 1).float().T.to(self.device)
        self.matrix_res = (data == -1).float().T.to(self.device)

    def set_problem_double_well(self):
        self.c = 0.5
        self.delta_t = 0.001

    def set_problem_12d_dw(self):
        mean_1 = torch.tensor([[1.2, 2.0, 0.6, 1.5, 0.9, 1.5, 1.5, 0.9, 1.2, 1.2, 0.5, 1.8]])
        mean_2 = torch.tensor([[1.8, 1.4, 0.8, 0.9, 0.9, 1.5, 2.0, 1.0, 1.6, 1.0, 0.7, 1.4]])
        mean = torch.cat([mean_2, mean_1], dim=0).to(self.device)
        # mean = mean[:, :self.dimension]
        print("Problem Dimension:", self.dimension, mean.shape)
        cov_1 = 0.04 * torch.eye(self.dimension).unsqueeze(0)
        cov_2 = 0.02 * torch.eye(self.dimension).unsqueeze(0)
        cov = torch.cat([cov_1, cov_2], dim=0).to(self.device)
        mix_weight = torch.tensor([0.6, 0.4]).to(self.device)
        mix = dist.Categorical(mix_weight)  # weight会被归一化
        # comp = dist.Independent(dist.Normal(mean, cov), 1)
        comp = dist.Independent(dist.MultivariateNormal(mean, cov), 0)
        self.distribution = dist.mixture_same_family.MixtureSameFamily(mix, comp)

        mat = torch.rand(self.dimension, self.dimension).to(self.device)
        self.mat_J_I = mat / 2 - mat.T / 2 - torch.eye(self.dimension).to(self.device)

        # marginal
        comp_margin = dist.Independent(dist.MultivariateNormal(mean[:, [self.index_1, self.index_2]], cov), 0)
        self.marginal_distribution = dist.mixture_same_family.MixtureSameFamily(mix, comp_margin)

        # high line
        direction = (mean_2 - mean_1).reshape(-1)
        self.t_list = torch.linspace(-1, 3, steps=200)
        self.draw_x = torch.ger(self.t_list, direction) + mean_1
        line_U = - self.D * self.distribution.log_prob(self.draw_x.to(self.device)).detach().cpu()
        line_U = line_U - line_U.min()
        line_U[line_U > 20 * self.D] = 20 * self.D
        self.line_U = line_U

    def set_problem_lorenz(self):
        self.rho = 28.0
        self.sigma = 10.0
        self.beta = 8.0 / 3.0
        self.delta_t = 0.01
        self.max_iter = 50000
        self.index_1 = 0
        self.index_2 = 2

    def activate(self, x):
        return self.a * torch.pow(x, self.n) / (self.S_n + torch.pow(x, self.n))

    def restrict(self, x):
        return self.b * self.S_n / (self.S_n + torch.pow(x, self.n))

    def force_2008(self, x):
        assert x.shape[1] == 2, "The dimension is not right."
        x1, x2 = torch.split(x, 1, dim=1)
        f1 = 100 * ((self.epsilon ** 2 + x1 ** 2) / (1 + x1 ** 2)) / (1 + x2) - 100 * self.a * x1
        f2 = 100 / self.tau_0 * (self.b - x2 / (1 + self.c * x1 ** 2))
        return torch.cat([f1, f2], dim=1)

    def force_2010(self, x):
        assert x.shape[1] == 8, "The dimension is not right."
        x1, x2, x3, x4, x5, x6, x7, x8 = torch.split(x, 1, dim=1)
        Keq = 1000
        Sm = x1 + x6 + 1 / Keq
        Trimer = (2 * x1 * x6) / (Sm + torch.sqrt(Sm ** 2 - 4 * x1 * x6))
        self.Cycb = x1 - Trimer

        f1 = self.k1 - (self.k21 + self.k22 * x2 + self.k23 * x4) * x1
        f2 = (self.k31 + self.k32 * x4) * (1 - x2) / (self.J3 + 1 - x2) - (
                self.k4 * self.m * self.Cycb + self.k41 * x7) * x2 / (self.J4 + x2)
        f3 = self.k51 + self.k52 * ((self.m * self.Cycb) ** self.n) / (
                self.J5 ** self.n + (self.m * self.Cycb) ** self.n) - self.k6 * x3
        f4 = self.k7 * x5 * (x3 - x4) / (self.J7 + (x3 - x4)) - self.k8 * self.Mad * x4 / (self.J8 + x4) - self.k6 * x4
        f5 = self.k9 * self.m * self.Cycb * (1 - x5) - self.k10 * x5
        f6 = self.k11 - (self.k121 + self.k122 * x7 + self.k123 * self.m * self.Cycb) * x6
        f7 = self.k131 + self.k132 * x8 - self.k14 * x7
        f8 = (self.k151 * self.m + self.k152 * x7) * (1 - x8) / (self.J15 + 1 - x8) - (
                self.k161 + self.k162 * self.m * self.Cycb) * x8 / (self.J16 + x8)
        return 1000 * torch.cat([f1, f2, f3, f4, f5, f6, f7, f8], dim=1)

    def force_2011(self, x):  # 3d Bio cycle, 2011
        assert x.shape[1] == 3, "The dimension is not right."
        CDK1, Plk1, APC = torch.split(x, 1, dim=1)
        APC_n1 = torch.pow(APC, self.n1)
        CDK1_n2 = torch.pow(CDK1, self.n2)
        Plk1_n3 = torch.pow(Plk1, self.n3)
        K1_n1 = self.K1 ** self.n1
        K2_n2 = self.K2 ** self.n2
        K3_n3 = self.K3 ** self.n3
        f1 = self.alpha1 - self.beta1 * CDK1 * APC_n1 / (K1_n1 + APC_n1)
        f2 = self.alpha2 * (1 - Plk1) * CDK1_n2 / (K2_n2 + CDK1_n2) - self.beta2 * Plk1
        f3 = self.alpha3 * (1 - APC) * Plk1_n3 / (K3_n3 + Plk1_n3) - self.beta3 * APC
        return 10 * torch.cat([f1, f2, f3], dim=1)

    def force_2013(self, x):
        f_new = -self.k * x + torch.mm(self.activate(x), self.matrix_act) + torch.mm(self.restrict(x),
                                                                                     self.matrix_res)
        return f_new.to(self.device)

    def force_double_well(self, x):  # Double well in x1, Gaussian in x2
        assert x.shape[1] == 2, "The dimension is not right."
        x1, x2 = torch.split(x, 1, dim=1)
        # derivative of double well potential ((x1 - x_max/2)^2 - 1)^2
        dvdx = 4.0 * (x1 - self.x_max * 0.5) * ((x1 - self.x_max * 0.5) ** 2 - 1.0)
        # derivative of gaussian centered at self.x_max/2
        dvdy = (x2 - self.x_max * 0.5)
        f1 = -dvdx - self.c * dvdy
        f2 = self.c * dvdx - dvdy
        return torch.cat([f1, f2], dim=1)

    def force_12d_dw_non(self, x):
        assert x.shape[1] == self.dimension, "The dimension is not right."
        x.requires_grad_()
        log_p = self.distribution.log_prob(x)
        log_p_x = torch.autograd.grad(outputs=log_p.sum(), inputs=x, create_graph=True, retain_graph=True)[0]  # N*D
        f = self.D * log_p_x
        f = -f @ self.mat_J_I.T
        return f

    def force_lorenz(self, state):
        """
        https://en.wikipedia.org/wiki/Lorenz_system
        """
        assert state.shape[1] == 3, "The dimension is not right."
        x, y, z = torch.split(state, 1, dim=1)
        f1 = self.sigma * (y - x)
        f2 = x * (self.rho - z) - y
        f3 = x * y - self.beta * z
        f = torch.cat([f1, f2, f3], dim=1)
        return f

    def force(self, x):
        if self.problem_id == 2008:
            return self.force_2008(x)
        elif self.problem_id == 2010:
            return self.force_2010(x)
        elif self.problem_id == 2011:
            return self.force_2011(x)
        elif self.problem_id == 2013:
            return self.force_2013(x)
        elif self.problem_id == 202201:
            return self.force_double_well(x)
        elif self.problem_id == 202202:
            return self.force_12d_dw_non(x)
        elif self.problem_id == 202203:
            return self.force_lorenz(x)
        else:
            raise "Not Implemented Problem."

    def eval_net(self, dnn, nf_flag=False):
        if self.problem_id == 202201:
            X = np.arange(0.01, self.x_max, 0.05)
            Y = np.arange(0.01, self.x_max, 0.05)
            X, Y = np.meshgrid(X, Y)
            x1 = torch.tensor(X, requires_grad=False).float().to(self.device).reshape(-1, 1)
            x2 = torch.tensor(Y, requires_grad=False).float().to(self.device).reshape(-1, 1)
            x = torch.cat([x1, x2], dim=1)
            U_True = ((x1 - self.x_max * 0.5) ** 2 - 1.0) ** 2 + 0.5 * (x2 - self.x_max * 0.5) ** 2
        elif self.problem_id == 202202:
            x = self.distribution.sample((self.sample_size,))
            U_True = - self.D * self.distribution.log_prob(x)
            U_True = U_True.reshape(-1, 1) - U_True.min()  # not needed.
        elif (self.problem_id == 2013) and (self.dimension == 2):
            assert self.x_max == 3.0
            X = np.linspace(0., 3.0, 301)
            Y = np.linspace(0., 3.0, 301)
            X, Y = np.meshgrid(X, Y)
            x = get_grid_data(X, Y, self.device)
            U_True = self.U_True.detach().float().reshape(-1, 1)
        elif self.problem_id == 2008:
            X = np.linspace(0., 8., 401)
            Y = np.linspace(0., 6., 301)
            X, Y = np.meshgrid(X, Y)
            x = get_grid_data(X, Y, self.device)
            U_True = self.U_True.detach().float().reshape(-1, 1)
        else:
            neg_tensor = torch.tensor(-1)
            return neg_tensor, neg_tensor, neg_tensor

        with torch.no_grad():
            if nf_flag:
                log_p = dnn.log_prob(x).reshape(-1, 1)
                U = - self.D * log_p.cpu()
            else:
                U = dnn(x).detach().cpu()

        U = U - U.min()
        U_True = U_True - U_True.min()
        U_True[U_True >= 20 * self.D] = torch.nan
        U_True = U_True.cpu()

        MSE = ((U - U_True) ** 2).nanmean()
        rRMSE = torch.sqrt(((U - U_True) ** 2).nansum()) / torch.sqrt((U_True ** 2).nansum())
        rMAE = (torch.abs(U - U_True)).nansum() / (torch.abs(U_True)).nansum()
        return MSE, rRMSE, rMAE
