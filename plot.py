import cv2 as cv
import matplotlib
import numpy as np
import torch

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def get_grid_data(x, y, device):
    x1 = torch.tensor(x, requires_grad=False).float().to(device).reshape(-1, 1)
    x2 = torch.tensor(y, requires_grad=False).float().to(device).reshape(-1, 1)
    return torch.cat([x1, x2], dim=1)


def predict_2d(problem, dnn, x, y):
    x, y = np.meshgrid(x, y)
    with torch.no_grad():
        u = dnn(get_grid_data(x, y, problem.device)).detach().cpu().numpy()
    return u.reshape(*x.shape, -1)


def plot_eval_u(problem, dnn, epoch, index_1=0, index_2=1, samples=None):
    color_map = "rainbow"

    if problem.problem_id == 2008:
        print("2008 plot with 8, 6")
        X = np.linspace(0., 8, 801)
        Y = np.linspace(0., 6, 601)
    elif problem.problem_id == 2010:
        print("2010 plot with 0.75, 1.5")
        X = np.linspace(0., 0.75, 801)
        Y = np.linspace(0., 1.25, 801)
    elif problem.problem_id == 2011:
        print("2011 3d plot with 0.8, 0.8")
        X = np.linspace(0., 0.8, 801)
        Y = np.linspace(0., 0.8, 801)
    elif problem.problem_id == 202203:
        print("202203 Lorenz plot with -20, 20")
        X = np.linspace(-20., 20., 1000)
        Y = np.linspace(0., 50., 1000)
    else:
        X = np.linspace(0., problem.x_max, 801)
        Y = np.linspace(0., problem.x_max, 801)
    U = predict_2d(problem, dnn, X, Y)[..., 0]

    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes()

    U = U - U.min()
    U[U >= 20 * problem.D] = 20 * problem.D

    X, Y = np.meshgrid(X, Y)
    surf = ax.pcolormesh(X, Y, U, cmap=color_map, shading='auto', vmax=U.max())
    ax.contourf(X, Y, U, 50, cmap=color_map)
    ax.set_aspect('equal')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.title(f'Potential $U=-Dln(P)$, D={problem.D}, Index={index_1}, {index_2}')
    if samples != None:
        samples = samples.detach().cpu().numpy()
        plt.scatter(samples[:, index_1], samples[:, index_2], s=0.1, c='k')
    plt.savefig(f"{problem.model_path}/U_{epoch}.png", dpi=100)
    plt.close()
    problem.writer.add_image(f'landscape {index_1}-{index_2}',
                             cv.cvtColor(cv.imread(f'{problem.model_path}/U_{epoch}.png'), cv.COLOR_BGR2RGB),
                             global_step=epoch,
                             dataformats='HWC')


def plot_high_line_in_training(problem, dnn, epoch, step=20):
    U = dnn(problem.draw_x.to(problem.device)).detach().cpu().numpy()
    U = U - U.min()

    U[U > 20 * problem.D] = 20 * problem.D
    plt.figure(figsize=(10, 10))
    plt.axes()

    plt.plot(problem.t_list, problem.line_U, label='true')
    plt.plot(problem.t_list, U, label='predict')

    labels = [str(round(x, 1)) for x in problem.draw_x[:, problem.index_1].numpy()[::step]]
    plt.xticks(problem.t_list[::step], labels)
    plt.legend()
    plt.title(
        f"Potential $U$, x{problem.index_1} from {problem.draw_x[0][problem.index_1]:.1f} to {problem.draw_x[-1][problem.index_1]:.1f}")
    plt.xlabel(f"$x{problem.index_1}$")
    plt.savefig(f"{problem.model_path}/{problem.dimension}D_line_U_{epoch}.png", dpi=100)
    plt.close()
    problem.writer.add_image(f'line',
                             cv.cvtColor(cv.imread(f'{problem.model_path}/{problem.dimension}D_line_U_{epoch}.png'),
                                         cv.COLOR_BGR2RGB),
                             global_step=epoch,
                             dataformats='HWC')


def plot_samples(problem, sde_samples, enh_samples):
    sde_samples = sde_samples.detach().cpu().numpy()
    enh_samples = enh_samples.detach().cpu().numpy()
    plt.xlim(0, problem.x_max)
    plt.ylim(0, problem.x_max)
    plt.scatter(enh_samples[:, problem.index_1], enh_samples[:, problem.index_2], s=0.5, c="green", label="enhanced")
    plt.scatter(sde_samples[:, problem.index_1], sde_samples[:, problem.index_2], s=0.5, c="orange", label="sde")
    plt.legend()
    plt.title(
        f'Dimension={problem.dimension}, D={problem.D}, Index={problem.index_1}, {problem.index_2}')
    plt.savefig(f"{problem.model_path}/samples_{problem.index_1}_{problem.index_2}.png", dpi=100)
    plt.close()


def plot_projected_force(problem, dnn, epoch, index_1=0, index_2=1, samples=None):
    if problem.problem_id == 2008:
        print("2008 plot with 8, 6")
        X = np.linspace(0., 8, 801)
        Y = np.linspace(0., 6, 601)
    elif problem.problem_id == 2014:
        print("2014 plot with 0.8, 0.8")
        X = np.linspace(0., 0.8, 801)
        Y = np.linspace(0., 0.8, 801)
    elif problem.problem_id == 20113:
        print("2011 3d plot with 0.8, 0.8")
        X = np.linspace(0., 0.8, 801)
        Y = np.linspace(0., 0.8, 801)
    elif problem.problem_id == 2010:
        print("2010 plot with 0.75, 1.5")
        X = np.linspace(0., 0.75, 801)
        Y = np.linspace(0., 1.5, 801)
    elif problem.problem_id == 202201:
        print("202201 Lorenz plot with -20, 20")
        X = np.linspace(-20., 20., 1000)
        Y = np.linspace(0., 50., 1000)
    else:
        X = np.linspace(0., problem.x_max, 801)
        Y = np.linspace(0., problem.x_max, 801)

    force = predict_2d(problem, dnn, X, Y)

    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes()

    f_X = force[:, :, 0]
    f_Y = force[:, :, 1]

    if problem.problem_id == 2010:
        lw = 0.1 * (np.sqrt(f_X ** 2 + f_Y ** 2))
    elif problem.problem_id == 202203:
        lw = 0.05 * (np.sqrt(f_X ** 2 + f_Y ** 2))
    else:
        lw = np.sqrt(f_X ** 2 + f_Y ** 2)
    strm = ax.streamplot(X, Y, f_X, f_Y, density=2., color=lw, linewidth=lw, arrowsize=1.5, cmap='rainbow')
    fig.colorbar(strm.lines)
    ax.set_title('Varying Line Width')
    ax.set_aspect('equal')
    plt.title(f'Projected Force, D={problem.D}, Index={index_1}, {index_2}')

    if samples != None:
        samples = samples.detach().cpu().numpy()
        plt.scatter(samples[:, index_1], samples[:, index_2], s=0.1, c='k')

    plt.savefig(f"{problem.model_path}/force_{epoch}.png", dpi=100)
    plt.close()
    problem.writer.add_image(f'force {index_1}-{index_2}',
                             cv.cvtColor(cv.imread(f'{problem.model_path}/force_{epoch}.png'), cv.COLOR_BGR2RGB),
                             global_step=epoch,
                             dataformats='HWC')
