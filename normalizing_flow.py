import os
import time

import normflows as nf
import torch

from gen_data import SimDataset
from plot import plot_eval_u
from problems import Problem
from utils import save_model, write_info


class NormalizingFlow:
    """
    https://github.com/VincentStimper/normalizing-flows
    """

    def __init__(self, model_path, args):
        self.device = args.device
        self.dimension = args.dimension

        # misc
        self.save_ckpt = args.save_ckpt
        self.log_interval = args.log_interval
        self.save_interval = args.save_interval
        self.model_path = model_path

        # problem
        self.problem = Problem(model_path, args)
        self.writer = self.problem.writer
        self.dataset = SimDataset(self.problem)
        if self.save_ckpt:
            write_info(args, os.path.join(self.model_path, 'info.log'))

        # network
        self.dr = args.dr
        self.num_potential_epochs = args.num_potential_epochs
        dimension = 2 if args.dr else self.problem.dimension
        self.dnn = self.set_nf_model(dimension)
        self.optimizer = torch.optim.Adam(self.dnn.parameters(), lr=args.lr)

    def set_nf_model(self, dimension):
        K = 4
        latent_size = dimension
        hidden_units = 64
        hidden_layers = 3

        flows = []
        for i in range(K):
            flows += [nf.flows.AutoregressiveRationalQuadraticSpline(latent_size, hidden_layers, hidden_units)]
            flows += [nf.flows.LULinearPermute(latent_size)]

        q0 = nf.distributions.DiagGaussian(dimension, trainable=False)
        return nf.NormalizingFlow(q0=q0, flows=flows).to(self.problem.device)

    def get_potential(self, x):
        log_p = self.dnn.log_prob(x).reshape(-1, 1)
        return - self.problem.D * log_p

    def train(self):
        start_time = time.time()
        num_epochs = self.num_potential_epochs
        for epoch in range(num_epochs):
            self.dnn.train()

            data = self.dataset.x_sde.detach()
            if self.dr:
                data = data[:, [self.problem.index_1, self.problem.index_2]]
            loss = self.dnn.forward_kld(data)

            self.optimizer.zero_grad()
            if ~(torch.isnan(loss) | torch.isinf(loss)):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.dnn.parameters(), max_norm=5, norm_type=2)
                self.optimizer.step()

            self.writer.add_scalar('total loss', loss.item(), global_step=epoch)

            if epoch % self.log_interval == 0:
                print(f"========= Epoch: {epoch} Spent Time: {time.time() - start_time}(s) ==========")
                print(f"Epoch {epoch}, Loss_NF: {loss.item()}")
                MSE, rRMSE, rMAE = self.problem.eval_net(self.dnn, nf_flag=True)
                print(f"MSE: {MSE}, rRMSE: {rRMSE}, rMAE: {rMAE}")
                self.writer.add_scalar('MSE', MSE.item(), global_step=epoch)
                self.writer.add_scalar('rRMSE', rRMSE.item(), global_step=epoch)
                self.writer.add_scalar('rMAE', rMAE.item(), global_step=epoch)

                if self.save_ckpt and (epoch % self.save_interval == 0):
                    save_model(self.dnn, os.path.join(self.model_path, 'model.pkl'))

                    if self.dimension == 2 or self.dr:
                        plot_eval_u(self.problem, self.get_potential, epoch, index_1=self.problem.index_1,
                                    index_2=self.problem.index_2)
