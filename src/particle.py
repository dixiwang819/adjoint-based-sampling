import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


class SelfAttention(nn.Module):
    """Self-attention layer that learns feature reweighting."""
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim // 4)
        self.fc2 = nn.Linear(dim // 4, dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x shape: (batch_size, dim)
        # Channel-wise (feature-wise) attention
        # This learns to reweight features based on sample content
        attn = self.fc1(x)  # (batch_size, dim//4)
        attn = torch.relu(attn)
        attn = self.fc2(attn)  # (batch_size, dim)
        attn = self.sigmoid(attn)  # (batch_size, dim)
        
        # Scale input by attention weights
        out = x * attn
        return out


class ResidualBlockWithAttention(nn.Module):
    """Residual block with self-attention."""
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim * 2
            
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, dim)
        self.attention = SelfAttention(dim)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        # Residual connection with attention
        residual = x
        
        # MLP part
        out = self.linear1(x)
        out = self.activation(out)
        out = self.linear2(out)
        
        # Self-attention part
        attn_out = self.attention(x)
        
        # Combine MLP and attention
        out = out + attn_out
        
        # Add residual connection
        out = out + residual
        
        return out


class ResNetWithAttention(nn.Module):
    """ResNet architecture with self-attention blocks for density estimation."""
    def __init__(self, input_dim=2, hidden_dim=64, n_blocks=4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Stack of residual blocks with attention
        self.residual_blocks = nn.ModuleList([
            ResidualBlockWithAttention(hidden_dim)
            for _ in range(n_blocks)
        ])
        
        # Output projection (map to density space)
        self.output_proj = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        # Project input to hidden space
        out = self.input_proj(x)
        
        # Apply residual blocks
        for block in self.residual_blocks:
            out = block(out)
        
        # Project output
        out = self.output_proj(out)
        
        return out


def generate_gaussian_samples(n_samples, dim=2, seed=None):
    """Generate samples from a standard Gaussian distribution."""
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    # Rejection-sample from a standard normal but only keep samples inside [-3, 3]^dim
    low, high = -3.0, 3.0
    samples = []
    tries = 0
    max_tries = 100
    # generate in batches until we have enough valid samples
    while len(samples) < n_samples and tries < max_tries:
        remaining = n_samples - len(samples)
        # oversample a bit to improve acceptance
        batch_size = max(remaining * 2, 64)
        gen = np.random.randn(batch_size, dim).astype(np.float32)
        mask = np.all((gen >= low) & (gen <= high), axis=1)
        good = gen[mask]
        if good.shape[0] > 0:
            take = min(good.shape[0], remaining)
            samples.append(good[:take])
        tries += 1

    if len(samples) == 0:
        # fallback: uniform samples in the box if rejection failed
        arr = np.random.uniform(low, high, size=(n_samples, dim)).astype(np.float32)
    else:
        arr = np.concatenate(samples, axis=0)
        if arr.shape[0] < n_samples:
            # fill remaining with uniform-in-box
            extra = np.random.uniform(low, high, size=(n_samples - arr.shape[0], dim)).astype(np.float32)
            arr = np.concatenate([arr, extra], axis=0)

    return torch.from_numpy(arr[:n_samples])


def generate_annulus_samples(n_samples, seed=None):
    """Generate samples from annulus distribution (uniform in area), scaled to [-3, 3]^2."""
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    # Parameters for annulus
    r_inner = 1.0
    r_outer = 2.0
    
    # Sample radii uniformly in area
    # For uniform sampling in annulus, pdf proportional to r
    # cdf = (r^2 - r_inner^2) / (r_outer^2 - r_inner^2)
    # So r^2 = r_inner^2 + u * (r_outer^2 - r_inner^2)
    u = np.random.uniform(0, 1, n_samples)
    r_squared = r_inner**2 + u * (r_outer**2 - r_inner**2)
    r = np.sqrt(r_squared)
    
    # Sample angles uniformly
    theta = np.random.uniform(0, 2*np.pi, n_samples)
    
    # Convert to Cartesian
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    # Scale to fit within [-3, 3]^2 if needed
    # Current range is roughly [-2, 2] x [-2, 2], already within [-3, 3]
    
    return torch.from_numpy(np.array([x, y], dtype=np.float32).T)


def generate_two_moons_samples(n_samples, noise=0.08, seed=None, horizontal_shift=1.8):
    """Generate a smooth two-moons target using noisy semicircular arcs."""
    generator = None
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        generator = torch.Generator().manual_seed(seed)

    n_upper = n_samples // 2
    n_lower = n_samples - n_upper

    theta_upper = np.pi * torch.rand(n_upper, generator=generator)
    theta_lower = np.pi * torch.rand(n_lower, generator=generator)

    upper = torch.stack((torch.cos(theta_upper), torch.sin(theta_upper)), dim=1)
    # Move the lower arc slightly right so the two inner tips sit closer together.
    lower = torch.stack((horizontal_shift - torch.cos(theta_lower), -torch.sin(theta_lower) - 0.5), dim=1)

    samples = torch.cat((upper, lower), dim=0)

    # Add Gaussian thickness so the target density is smooth rather than concentrated
    # on a piecewise-defined curve.
    samples = samples + noise * torch.randn(samples.shape, generator=generator)

    # Shuffle the combined sample set and rescale for a comparable plotting/training range.
    permutation = torch.randperm(samples.shape[0], generator=generator)
    samples = samples[permutation]
    samples = samples - samples.mean(dim=0, keepdim=True)
    max_abs = samples.abs().max()
    if max_abs > 0:
        samples = 2.2 * samples / max_abs

    return samples.to(dtype=torch.float32)


def generate_checker_samples(n_samples, seed=None, grid_size=4, extent=2.0):
    """Generate a continuous checkerboard distribution in alternating squares."""
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    cell_size = (2.0 * extent) / grid_size
    samples = []

    while len(samples) < n_samples:
        remaining = n_samples - len(samples)
        batch_size = max(remaining * 2, 64)
        candidates = np.random.uniform(-extent, extent, size=(batch_size, 2)).astype(np.float32)

        x_index = np.floor((candidates[:, 0] + extent) / cell_size).astype(int)
        y_index = np.floor((candidates[:, 1] + extent) / cell_size).astype(int)

        x_index = np.clip(x_index, 0, grid_size - 1)
        y_index = np.clip(y_index, 0, grid_size - 1)

        mask = (x_index + y_index) % 2 == 0
        accepted = candidates[mask]
        if accepted.shape[0] > 0:
            take = min(accepted.shape[0], remaining)
            samples.append(accepted[:take])

    return torch.from_numpy(np.concatenate(samples, axis=0)[:n_samples])


def generate_gaussian_mixture_samples(n_samples, seed=None):
    """Generate a four-component Gaussian mixture with separated modes."""
    generator = None
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        generator = torch.Generator().manual_seed(seed)

    centers = torch.tensor(
        [
            [-1.6, -1.2],
            [-1.0, 1.1],
            [1.1, -1.0],
            [1.7, 1.3],
        ],
        dtype=torch.float32,
    )
    n_components = centers.shape[0]
    base_count = n_samples // n_components
    counts = [base_count] * n_components
    for index in range(n_samples - base_count * n_components):
        counts[index] += 1

    component_samples = []
    component_std = 0.22
    for center, count in zip(centers, counts):
        noise = component_std * torch.randn((count, 2), generator=generator)
        component_samples.append(center.unsqueeze(0) + noise)

    samples = torch.cat(component_samples, dim=0)
    permutation = torch.randperm(samples.shape[0], generator=generator)
    samples = samples[permutation]
    return samples.to(dtype=torch.float32)


def generate_target_samples(target_type, n_samples, seed=None):
    """Dispatch helper for supported target distributions."""
    generators = {
        "annulus": generate_annulus_samples,
        "two_moons": generate_two_moons_samples,
        "checker": generate_checker_samples,
        "checkerboard": generate_checker_samples,
        "gaussian_mixture": generate_gaussian_mixture_samples,
        "gmm": generate_gaussian_mixture_samples,
    }

    if target_type not in generators:
        supported = ", ".join(sorted(generators))
        raise ValueError(f"Unknown target_type '{target_type}'. Supported values: {supported}")

    return generators[target_type](n_samples, seed=seed)


def adaptive_bandwidth(samples):
    """Compute adaptive bandwidth using Scott's rule."""
    n = samples.shape[0]
    d = samples.shape[1]
    return n ** (-1 / (d + 4))


def default_target_bandwidth(target_type):
    """Return a task-specific target KDE bandwidth when a fixed value is helpful."""
    if target_type == "two_moons":
        return 0.12
    return None


def default_mmd_bandwidths(target_type):
    """Return kernel scales for MMD."""
    if target_type == "two_moons":
        return [0.08, 0.12, 0.2, 0.4, 0.8, 1.6, 3.2]
    return [0.1, 0.2, 0.4, 0.8, 1.6, 3.2]


def squared_pairwise_distances(x, y):
    """Compute squared pairwise Euclidean distances without a sqrt singularity."""
    diff = x[:, None, :] - y[None, :, :]
    return (diff ** 2).sum(dim=-1)


def kl_divergence_loss(
    model_output,
    target_samples,
    n_eval_samples=1000,
    use_symmetric=True,
    target_bandwidth=None,
    model_bandwidth=None,
):
    """
    Compute KL divergence with adaptive bandwidth.
    
    Args:
        model_output: Samples from the model
        target_samples: Samples from the target distribution
        n_eval_samples: Number of samples to evaluate at
        use_symmetric: If True, use symmetric KL: KL(p||q) + KL(q||p)
        target_bandwidth: Optional fixed KDE bandwidth for target samples
        model_bandwidth: Optional fixed KDE bandwidth for model samples
    
    Returns:
        KL divergence value
    """
    # Simple approach: use kernel density estimation
    # KL(p || q) = E_p[log p(x) - log q(x)]
    
    device = model_output.device
    batch_size = min(n_eval_samples, target_samples.shape[0])
    
    # Adaptive bandwidth
    if target_bandwidth is None:
        bandwidth_target = max(0.05, adaptive_bandwidth(target_samples.cpu().numpy()))
    else:
        bandwidth_target = float(target_bandwidth)

    if model_bandwidth is None:
        bandwidth_model = max(0.05, adaptive_bandwidth(model_output.detach().cpu().numpy()))
    else:
        bandwidth_model = float(model_bandwidth)
    
    # ========== KL(target || model) ==========
    # Evaluate at target points
    eval_indices = torch.randperm(target_samples.shape[0])[:batch_size]
    eval_points_target = target_samples[eval_indices].to(device)
    
    # p(x) - density under target distribution
    dist2_to_target = squared_pairwise_distances(eval_points_target, target_samples.to(device))
    log_p = torch.logsumexp(-dist2_to_target / (2 * bandwidth_target**2), dim=1) - np.log(target_samples.shape[0])
    
    # q(x) - density under model distribution
    dist2_to_model = squared_pairwise_distances(eval_points_target, model_output)
    log_q = torch.logsumexp(-dist2_to_model / (2 * bandwidth_model**2), dim=1) - np.log(model_output.shape[0])
    
    kl_forward = (log_p - log_q).mean()
    
    # ========== KL(model || target) - Reverse KL for mode coverage ==========
    if use_symmetric:
        # Evaluate at model points
        eval_indices_model = torch.randperm(model_output.shape[0])[:batch_size]
        eval_points_model = model_output[eval_indices_model]
        
        # q(x) - density under model
        dist2_to_model_rev = squared_pairwise_distances(eval_points_model, model_output)
        log_q_rev = torch.logsumexp(-dist2_to_model_rev / (2 * bandwidth_model**2), dim=1) - np.log(model_output.shape[0])
        
        # p(x) - density under target
        dist2_to_target_rev = squared_pairwise_distances(eval_points_model, target_samples.to(device))
        log_p_rev = torch.logsumexp(-dist2_to_target_rev / (2 * bandwidth_target**2), dim=1) - np.log(target_samples.shape[0])
        
        kl_reverse = (log_q_rev - log_p_rev).mean()
        
        # Symmetric KL
        kl = 0.5 * kl_forward + 0.5 * kl_reverse
    else:
        kl = kl_forward
    
    return kl


def mmd_loss(model_output, target_samples, bandwidths=None):
    """Compute Gaussian-kernel MMD between model and target samples."""
    target_samples = target_samples.to(model_output.device)

    if bandwidths is None:
        combined = torch.cat((model_output.detach(), target_samples.detach()), dim=0)
        base_bandwidth = max(0.05, adaptive_bandwidth(combined.cpu().numpy()))
        bandwidths = [
            0.25 * base_bandwidth,
            0.5 * base_bandwidth,
            base_bandwidth,
            2.0 * base_bandwidth,
            4.0 * base_bandwidth,
            8.0 * base_bandwidth,
        ]

    xx_dist2 = squared_pairwise_distances(model_output, model_output)
    yy_dist2 = squared_pairwise_distances(target_samples, target_samples)
    xy_dist2 = squared_pairwise_distances(model_output, target_samples)

    k_xx = 0.0
    k_yy = 0.0
    k_xy = 0.0
    for bandwidth in bandwidths:
        bandwidth = float(bandwidth)
        scale = 2.0 * bandwidth ** 2
        k_xx = k_xx + torch.exp(-xx_dist2 / scale)
        k_yy = k_yy + torch.exp(-yy_dist2 / scale)
        k_xy = k_xy + torch.exp(-xy_dist2 / scale)

    num_kernels = float(len(bandwidths))
    k_xx = k_xx / num_kernels
    k_yy = k_yy / num_kernels
    k_xy = k_xy / num_kernels

    return k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()


def terminal_loss(
    model_output,
    target_samples,
    loss_type="kl",
    target_type="annulus",
    n_eval_samples=1000,
):
    """Dispatch between supported terminal matching losses."""
    if loss_type == "kl":
        return kl_divergence_loss(
            model_output,
            target_samples,
            n_eval_samples=n_eval_samples,
            target_bandwidth=default_target_bandwidth(target_type),
        )

    if loss_type == "mmd":
        return mmd_loss(
            model_output,
            target_samples,
            bandwidths=default_mmd_bandwidths(target_type),
        )

    raise ValueError(f"Unknown loss_type '{loss_type}'. Supported values: kl, mmd")


def train(
    model,
    n_samples=1024,
    n_blocks=4,
    n_epochs=1000,
    learning_rate=0.001,
    device='cpu',
    target_type='two_moons',
    loss_type='kl',
):
    """
    Train the ResNet model with self-attention.
    
    Args:
        model: The model to train
        n_samples: Number of samples per batch
        n_blocks: Number of residual blocks
        n_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to train on ('cpu' or 'cuda')
        target_type: Target distribution to learn ('annulus' or 'two_moons')
        loss_type: Terminal loss to optimize ('kl' or 'mmd')
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Generate a fixed target set for the chosen distribution.
    target_samples = generate_target_samples(target_type, n_samples * 2)
    
    losses = []
    
    for epoch in range(n_epochs):
        # Generate input samples (Gaussian)
        input_samples = generate_gaussian_samples(n_samples).to(device)
        
        # Forward pass
        output_samples = model(input_samples)
        
        # Compute terminal loss
        loss = terminal_loss(
            output_samples,
            target_samples,
            loss_type=loss_type,
            target_type=target_type,
            n_eval_samples=n_samples,
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (epoch + 1) % 20 == 0:
            print(
                f"Epoch {epoch + 1}/{n_epochs}, Target: {target_type}, Loss: {loss_type.upper()}, Value: {loss.item():.6f}"
            )
    
    return losses


def visualize_results(model, n_samples=500, device='cpu', target_type='annulus', loss_type='kl', n_blocks=4, n_epochs=200, run_index=None):
    """Visualize the model output only."""
    model.eval()
    with torch.no_grad():
        gaussian_samples = generate_gaussian_samples(n_samples).to(device)
        model_samples = model(gaussian_samples).cpu().numpy()

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax.scatter(model_samples[:, 0], model_samples[:, 1], alpha=0.5, s=10)
    ax.set_title("SARes Output")
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    
    plt.tight_layout()
    run_suffix = f"_k{run_index}" if run_index is not None else ""
    output_path = f'/Users/dixiwang/a/{loss_type}_{n_blocks}_{n_epochs}_{target_type}{run_suffix}.png'
    plt.savefig(output_path, dpi=100)
    print(f"Results saved to {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    # Configuration
    num_run = 1
    n_blocks = 5  # Number of residual blocks
    hidden_dim = 64
    n_epochs = 2500
    learning_rate = 1e-4
    target_type = "checkerboard"  # Options: "annulus", "two_moons", "checker", "checkerboard", "gaussian_mixture"
    loss_type = "mmd"  # Options: "kl", "mmd"
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print(f"Target distribution: {target_type}")
    print(f"Terminal loss: {loss_type}")
    
    for k in range(num_run):
        n_samples = 3000 + 100 * k
        print(f"\nRun k={k} with n_samples={n_samples}")

        model = ResNetWithAttention(input_dim=2, hidden_dim=hidden_dim, n_blocks=n_blocks)
        print(f"Model created with {n_blocks} residual blocks")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

        print("Training...")
        losses = train(
            model,
            n_samples=n_samples,
            n_blocks=n_blocks,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            device=device,
            target_type=target_type,
            loss_type=loss_type,
        )

        print("Visualizing results...")
        visualize_results(
            model,
            n_samples=n_samples,
            device=device,
            target_type=target_type,
            loss_type=loss_type,
            n_blocks=n_blocks,
            n_epochs=n_epochs,
            run_index=k,
        )

        plt.figure(figsize=(10, 5))
        plt.plot(losses)
        plt.xlabel("Epoch")
        plt.ylabel(f"{loss_type.upper()} Loss")
        plt.title(f"Training Loss ({loss_type.upper()}, k={k}, n_samples={n_samples})")
        plt.grid(True)
        loss_path = f'/Users/dixiwang/a/loss_{loss_type}_{target_type}_k{k}.png'    
        plt.savefig(loss_path, dpi=100)
        print(f"Loss plot saved to {loss_path}")
        plt.close()
