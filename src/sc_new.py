"""Particle CNF trainer using the same network architecture as `particle.py`.

Discretises the PDE-constrained objective

    J = \int_0^T \int \lambda_L 1/2 ||\chi||^2 d\mu dt + \lambda_M KL(target||\mu_T)

and parameterises \chi(x,t) using the same residual self-attention network as
`particle.py`. The script is minimal but runnable and reuses sampling/loss
helpers from `particle.py`.
"""

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from particle import (
    ResNetWithAttention,
    generate_gaussian_samples,
    generate_target_samples,
    terminal_loss,
)


def velocity_field(model, x, time_value, velocity_scale=1.0):
    """Evaluate a time-conditioned velocity field using the shared ResNet backbone."""
    t = torch.full((x.shape[0], 1), time_value, device=x.device, dtype=x.dtype)
    xt = torch.cat((x, t), dim=1)
    return velocity_scale * model(xt)[:, :2]


def rollout_particles(model, x0, n_steps, T, velocity_scale=1.0):
    """Roll particles forward with explicit Euler using the learned velocity field."""
    x = x0.clone()
    dt = T / n_steps

    for step in range(n_steps):
        time_value = step * dt
        x.requires_grad_(True)
        v = velocity_field(model, x, time_value, velocity_scale=velocity_scale)
        x = x + v * dt

    return x


def train_transformer_flow(
    n_particles=1024,
    n_steps=50,
    T=1.0,
    lambda_L=1.0,
    lambda_M=10.0,
    n_epochs=200,
    lr=1e-3,
    device='cpu',
    target_type='two_moons',
    loss_type='kl',
    hidden_dim=64,
    n_blocks=4,
    grad_clip=1.0,
    velocity_scale=1.0,
):
    device = torch.device(device)

    # initial and target samples
    x0 = generate_gaussian_samples(n_particles).to(device)
    target = generate_target_samples(target_type, n_particles, seed=0).to(device)

    model = ResNetWithAttention(input_dim=3, hidden_dim=hidden_dim, n_blocks=n_blocks).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    dt = T / n_steps
    total_losses = []
    kl_losses = []

    for epoch in range(1, n_epochs + 1):
        x = x0.clone()
        total_loss = 0.0

        for step in range(n_steps):
            time_value = step * dt
            x.requires_grad_(True)
            v = velocity_field(model, x, time_value, velocity_scale=velocity_scale)

            # cost: kinetic energy term averaged over particles
            step_cost = 0.5 * (v ** 2).sum(dim=1).mean()
            total_loss = total_loss + lambda_L * step_cost * dt

            # Euler step
            x = x + v * dt

        terminal_match = terminal_loss(
            x,
            target,
            loss_type=loss_type,
            target_type=target_type,
            n_eval_samples=min(1000, target.size(0)),
        )
        total_loss = total_loss + lambda_M * terminal_match

        opt.zero_grad()
        total_loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        opt.step()
        total_losses.append(total_loss.item())
        kl_losses.append(terminal_match.item())

        if epoch % 20 == 0 or epoch == 1:
            print(
                f"Epoch {epoch}/{n_epochs}  loss={total_loss.item():.6f}  "
                f"{loss_type.upper()}={terminal_match.item():.6f}  target={target_type}"
            )

    return model, x.detach(), target.detach(), total_losses, kl_losses


def visualize_results(
    model,
    n_samples=500,
    device='cpu',
    target_type='annulus',
    loss_type='kl',
    n_blocks=4,
    n_epochs=200,
    n_steps=50,
    T=1.0,
    velocity_scale=1.0,
    run_index=None,
):
    """Visualize model output and target distribution."""
    model.eval()
    with torch.no_grad():
        gaussian_samples = generate_gaussian_samples(n_samples).to(device)
        model_samples = rollout_particles(
            model,
            gaussian_samples,
            n_steps=n_steps,
            T=T,
            velocity_scale=velocity_scale,
        ).cpu().numpy()
        target_samples = generate_target_samples(target_type, n_samples).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].scatter(model_samples[:, 0], model_samples[:, 1], alpha=0.5, s=10)
    axes[0].set_title("SC Output")
    axes[0].set_xlim(-2.5, 2.5)
    axes[0].set_ylim(-2.5, 2.5)

    axes[1].scatter(target_samples[:, 0], target_samples[:, 1], alpha=0.5, s=10)
    axes[1].set_title(f"Target ({target_type.replace('_', ' ').title()})")
    axes[1].set_xlim(-2.5, 2.5)
    axes[1].set_ylim(-2.5, 2.5)

    plt.tight_layout()
    run_suffix = f'_k{run_index}' if run_index is not None else ''
    output_path = f'sc_{n_blocks}_{n_epochs}_{n_samples}_{loss_type}_{target_type}{run_suffix}.png'
    plt.savefig(output_path)
    print(f'saved {output_path}')
    plt.close(fig)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device =', device)
    num_run = 1
    target_type = 'checkerboard'  # Options: 'annulus', 'two_moons', 'checker', 'checkerboard', 'gaussian_mixture'
    loss_type = 'mmd'  # Options: 'kl', 'mmd'
    n_particles = 3000
    hidden_dim = 64
    n_blocks = 5
    n_epochs = 2500
    n_steps = 60
    T = 1.0   
    lr = 1e-4
    lambda_L = 0.1
    lambda_M = 20.0
    grad_clip = 1.0
    velocity_scale = 0.25
    print(f'target = {target_type}')
    print(f'loss = {loss_type}')

    for k in range(num_run):
        n_particles = n_particles * (k + 1)  # Scale up particles for each run
        print(f'\nRun k={k} with n_particles={n_particles}')

        model, final_particles, target, total_losses, kl_losses = train_transformer_flow(
            n_particles=n_particles,
            n_steps=n_steps,
            T=T,
            n_epochs=n_epochs,
            device=device,
            target_type=target_type,
            loss_type=loss_type,
            hidden_dim=hidden_dim,
            n_blocks=n_blocks,
            lr=lr,
            lambda_L=lambda_L,
            lambda_M=lambda_M,
            grad_clip=grad_clip,
            velocity_scale=velocity_scale,
        )

        visualize_results(
            model,
            n_samples=n_particles,
            device=device,
            target_type=target_type,
            loss_type=loss_type,
            n_blocks=n_blocks,
            n_epochs=n_epochs,
            n_steps=n_steps,
            T=T,
            velocity_scale=velocity_scale,
            run_index=k,
        )

        plt.figure(figsize=(10, 5))
        plt.plot(kl_losses)
        plt.xlabel('Epoch')
        plt.ylabel(f'{loss_type.upper()} Loss')
        plt.title(f'Training {loss_type.upper()} Loss (k={k}, n_particles={n_particles})')
        plt.grid(True)
        loss_path = f'sc_loss_{loss_type}_{target_type}_k{k}.png'
        plt.savefig(loss_path)
        print(f'saved {loss_path}')
        plt.close()
