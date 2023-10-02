import os
import time

import numpy as np
import torch
import torch.utils.data.dataloader as dataloader

from gen_data import SimDataset
from network import DNN
from plot import plot_eval_u, plot_high_line_in_training, plot_projected_force
from problems import Problem
from utils import write_info, save_model


class EnergyLandscape:
    def __init__(self, layers, model_path, args):
        self.device = args.device
        self.sample_size = args.sample_size
        self.dimension = args.dimension

        # loss parameters
        self.rho_1 = args.rho_1
        self.rho_2 = args.rho_2

        # misc
        self.save_ckpt = args.save_ckpt
        self.log_interval = args.log_interval
        self.save_interval = args.save_interval
        self.model_path = model_path
        self.dr = args.dr

        self.problem = Problem(model_path, args)
        self.writer = self.problem.writer
        self.dataset = SimDataset(self.problem)
        if self.save_ckpt:
            write_info(args, os.path.join(self.model_path, 'info.log'))

        # network
        self.layers = layers
        self.dnn = DNN(layers).to(self.device)
        self.optimizer = torch.optim.Adam(self.dnn.parameters(), lr=args.lr)

    def loss_epr(self, x):
        u = self.dnn(x)
        f = self.problem.force(x).detach()
        u_x = torch.autograd.grad(outputs=u.sum(), inputs=x, create_graph=True, retain_graph=True)[0]
        loss_f = (((f + u_x) ** 2).sum(1)).mean()
        return loss_f

    def get_hjb_residual(self, x, u, f):
        u_x = torch.autograd.grad(outputs=u.sum(), inputs=x, create_graph=True, retain_graph=True)[0]
        f_x = torch.zeros_like(x[:, 0:1])
        u_xx = torch.zeros_like(x[:, 0:1])
        for i in range(f.shape[1]):
            f_x += torch.autograd.grad(outputs=f[:, i].sum(), inputs=x, create_graph=True, retain_graph=True,
                                       allow_unused=True)[0][:, i:i + 1]
            u_xx += torch.autograd.grad(outputs=u_x[:, i].sum(), inputs=x, create_graph=True, retain_graph=True,
                                        allow_unused=True)[0][:, i:i + 1]
        hjb_residual = - (f.detach() * u_x).sum(dim=1, keepdims=True) + self.problem.D * (u_xx + f_x.detach()) - (
                u_x ** 2).sum(dim=1, keepdims=True)
        return hjb_residual

    def loss_hjb(self, x):
        u = self.dnn(x)
        f = self.problem.force(x)
        return (self.get_hjb_residual(x, u, f) ** 2).mean()

    def train_potential(self, num_epochs):
        train_loader = dataloader.DataLoader(self.dataset, batch_size=self.problem.batch_size, shuffle=False)
        start_time = time.time()

        for epoch in range(num_epochs):
            self.dnn.train()

            # NOTE: We use a fixed dataset for 8D problem.
            if self.problem.problem_id != 2010:
                self.dataset.update_enh_data()
                self.dataset.update_sde_data()

            total_loss_list = []
            epr_loss_list = []
            hjb_loss_list = []

            for data_sde, data_enh in train_loader:
                data_sde = data_sde.to(self.problem.device)
                data_enh = data_enh.to(self.problem.device)

                data_sde.requires_grad_()
                data_enh.requires_grad_()

                loss_epr = self.loss_epr(data_sde)
                loss_hjb = self.loss_hjb(data_enh)

                loss = self.rho_1 * loss_epr + self.rho_2 * loss_hjb
                if ~(torch.isnan(loss) | torch.isinf(loss)):
                    total_loss_list.append(loss.item())
                    epr_loss_list.append(loss_epr.item())
                    hjb_loss_list.append(loss_hjb.item())
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(parameters=self.dnn.parameters(), max_norm=5, norm_type=2)
                    self.optimizer.step()

            total_loss = np.mean(total_loss_list)
            epr_loss = np.mean(epr_loss_list)
            hjb_loss = np.mean(hjb_loss_list)

            self.writer.add_scalar('total loss', total_loss, global_step=epoch)
            self.writer.add_scalar('epr loss', epr_loss, global_step=epoch)
            self.writer.add_scalar('hjb loss', hjb_loss, global_step=epoch)

            if (epoch + 1) % self.log_interval == 0:
                print(f"========= Epoch: {epoch} Spent Time: {time.time() - start_time}(s) ==========")
                print(f"Epoch {epoch}, Total Loss: {total_loss}, Loss_EPR: {epr_loss}, Loss_HJB: {hjb_loss}")
                MSE, rRMSE, rMAE = self.problem.eval_net(self.dnn, nf_flag=False)
                print(f"MSE: {MSE}, rRMSE: {rRMSE}, rMAE: {rMAE}")
                self.writer.add_scalar('MSE', MSE.item(), global_step=epoch)
                self.writer.add_scalar('rRMSE', rRMSE.item(), global_step=epoch)
                self.writer.add_scalar('rMAE', rMAE.item(), global_step=epoch)

                if self.dimension == 2 or self.dr:
                    plot_eval_u(self.problem, self.dnn, epoch, index_1=self.problem.index_1,
                                index_2=self.problem.index_2, samples=data_sde)
                elif self.problem.problem_id == 202202:
                    plot_high_line_in_training(self.problem, self.dnn, epoch)

            if self.save_ckpt and (epoch % self.save_interval == 0):
                save_model(self.dnn, os.path.join(self.model_path, f'model_{epoch}.pkl'))

    def train(self):
        self.train_potential(self.problem.num_potential_epochs)


class DimensionReduction(EnergyLandscape):
    def __init__(self, layers, model_path, args):
        super().__init__(layers, model_path, args)
        print(f"Dimension Reduction: The origin dim is {self.problem.dimension}.")
        print(f"INDEX: {self.problem.index_1}, {self.problem.index_2}.")

        layers_f = layers
        layers_f[-1] = 2
        self.dnn_f = DNN(layers_f).to(self.device)
        self.optimizer_f = torch.optim.Adam(self.dnn_f.parameters(), lr=args.lr)

    def loss_force(self, data_high):
        data_dr = data_high[:, [self.problem.index_1, self.problem.index_2]]
        f_high = self.problem.force(data_high).detach()
        f_dr = self.dnn_f(data_dr)
        loss_f = ((f_dr - f_high[:, [self.problem.index_1, self.problem.index_2]].detach()) ** 2).mean()
        return loss_f

    def loss_epr(self, data_high):
        data_vis = data_high[:, [self.problem.index_1, self.problem.index_2]]
        U_vis = self.dnn(data_vis)
        U_vis_x = torch.autograd.grad(outputs=U_vis.sum(), inputs=data_vis, create_graph=True, retain_graph=True)[0]
        f_high = self.problem.force(data_high).detach()
        loss_epr = ((U_vis_x + f_high[:, [self.problem.index_1, self.problem.index_2]]) ** 2).mean()
        return loss_epr

    def loss_hjb(self, x):
        x = x[:, [self.problem.index_1, self.problem.index_2]]
        x.requires_grad_()
        f = self.dnn_f(x)
        u = self.dnn(x)
        return (self.get_hjb_residual(x, u, f) ** 2).mean()

    def train_force(self, num_epochs):
        train_loader = dataloader.DataLoader(self.dataset, batch_size=self.problem.batch_size, shuffle=True)
        start_time = time.time()
        for epoch in range(num_epochs):
            self.dnn_f.train()
            force_loss_list = []
            for data_sde, _ in train_loader:
                data_sde = data_sde.to(self.problem.device)
                data_sde.requires_grad_()

                loss = self.loss_force(data_sde)

                self.optimizer_f.zero_grad()
                if ~(torch.isnan(loss) | torch.isinf(loss)):
                    force_loss_list.append(loss.item())
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(parameters=self.dnn_f.parameters(), max_norm=5, norm_type=2)
                    self.optimizer_f.step()

            force_loss = np.mean(force_loss_list)
            self.writer.add_scalar('force loss', force_loss, global_step=epoch)
            if epoch % self.log_interval == 0:
                print(f"========= Epoch: {epoch} Spent Time: {time.time() - start_time}(s) ==========")
                print(f"Epoch {epoch}, Force Loss: {force_loss}")
                plot_projected_force(self.problem, self.dnn_f, epoch, index_1=self.problem.index_1,
                                     index_2=self.problem.index_2, samples=data_sde)

            if self.save_ckpt and (epoch % self.save_interval == 0):
                save_model(self.dnn_f, os.path.join(self.model_path, f'force_{epoch}.pkl'))

    def train(self):
        self.train_force(self.problem.num_force_epochs)
        self.train_potential(self.problem.num_potential_epochs)
