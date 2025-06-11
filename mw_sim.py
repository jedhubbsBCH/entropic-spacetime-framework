#!/usr/bin/env python3
# eeg_spatiotemporal_pinn_optimized.py — complete, mask‑safe, plotted build (10 Jun 2025)

from __future__ import annotations
import argparse, pathlib, datetime, random, math, csv
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
import mne
from scipy.signal import savgol_filter, find_peaks

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
CANONICAL_BANDS: Dict[str, Tuple[int, int]] = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta':  (13, 30),
    'gamma': (30, 80)
}

# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def seed_everything(seed: int):
    """
    Seeds all relevant random number generators for reproducibility.
    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device() -> torch.device:
    """
    Gets the appropriate device (CUDA or CPU) for tensor operations.
    Returns:
        torch.device: The selected device.
    """
    if torch.cuda.is_available():
        dev = torch.device('cuda')
        print(f"[CUDA] Using {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device('cpu')
        print('[CPU] CUDA unavailable – expect slow training')
    return dev

def ensure_psd_shape(psd: torch.Tensor, target_bins: int, device: torch.device) -> torch.Tensor:
    """
    Resamples PSD to target_bins if needed.
    Args:
        psd (torch.Tensor): The input Power Spectral Density tensor. Expected shape (batch, channels, freqs).
        target_bins (int): The desired number of frequency bins.
        device (torch.device): The device the tensor is on.
    Returns:
        torch.Tensor: The resampled PSD tensor.
    """
    if psd.shape[-1] == target_bins:
        return psd
    # F.interpolate with mode='linear' expects a 3D input (N, C, L_in) and interpolates along L_in.
    # Our `psd` tensor is already (batch, channels, freqs), so it fits this requirement directly.
    psd_resampled = F.interpolate(psd, size=target_bins, mode='linear', align_corners=False)
    return psd_resampled

# ─────────────────────────────────────────────────────────────────────────────
# Band discovery helper
# ─────────────────────────────────────────────────────────────────────────────

def find_prominent_bands(freqs: np.ndarray, mean_psd: np.ndarray, prominence_std_thresh: float, min_freq_filter: float) -> Dict[str, Tuple[float, float]]:
    """
    Identifies statistically prominent, non-overlapping frequency bands based on peak prominence.
    Args:
        freqs (np.ndarray): Array of frequencies.
        mean_psd (np.ndarray): Mean Power Spectral Density across channels.
        prominence_std_thresh (float): Threshold for peak prominence in noise standard deviations.
        min_freq_filter (float): The minimum frequency used for filtering, affects band boundary search.
    Returns:
        Dict[str, Tuple[float, float]]: Dictionary of discovered bands with their frequency ranges.
    """
    # Debug print: Show the frequency range being processed by this function
    print(f"DEBUG: find_prominent_bands received freqs from {freqs[0]:.1f} to {freqs[-1]:.1f} Hz. Min filter: {min_freq_filter} Hz")

    # Use a Savitzky-Golay filter to estimate the noise floor
    noise_floor = savgol_filter(mean_psd, window_length=51, polyorder=3)
    noise = mean_psd - noise_floor
    noise_std = np.std(noise)
    
    min_prominence = prominence_std_thresh * noise_std
    peaks, _ = find_peaks(mean_psd, prominence=min_prominence)
    
    # Debug print: Show how many peaks were found
    print(f"DEBUG: Found {len(peaks)} peaks in the filtered range.")

    if not len(peaks):
        print(f'[WARN] No peaks found with prominence > {prominence_std_thresh} std in the filtered range. Falling back to canonical bands (filtered).')
        # Return canonical bands, but ensure they are within the filtered range if applicable
        filtered_canonical_bands = {}
        for band_name, (f_low, f_high) in CANONICAL_BANDS.items():
            if f_high >= min_freq_filter: # Only include bands that overlap with the filtered range
                filtered_canonical_bands[band_name] = (max(f_low, min_freq_filter), f_high)
        return filtered_canonical_bands
    
    troughs, _ = find_peaks(-mean_psd)
    # Ensure band boundaries respect the overall min_freq_filter
    # This finds the first index >= min_freq_filter in the *already filtered* freqs array
    min_freq_idx_for_bands = np.argmax(freqs >= min_freq_filter) 

    discovered_bands: Dict[str, Tuple[float, float]] = {}
    for i, peak_idx in enumerate(peaks):
        # Find the nearest troughs to define band boundaries
        # Filter troughs to be only those valid within the current frequency range being processed
        left_troughs = troughs[(troughs < peak_idx) & (freqs[troughs] >= min_freq_filter)]
        left_boundary_idx = left_troughs[-1] if any(left_troughs) else min_freq_idx_for_bands

        right_troughs = troughs[(troughs > peak_idx) & (freqs[troughs] <= 100)] # Upper limit is 100 Hz
        right_boundary_idx = right_troughs[0] if any(right_troughs) else len(freqs) - 1
        
        f_low = freqs[left_boundary_idx]
        f_high = freqs[right_boundary_idx]
        
        # Ensure discovered band is within the overall filtered range and valid
        if f_high > f_low and f_high >= min_freq_filter: 
            band_name = f"Band {i+1} ({f_low:.1f}-{f_high:.1f} Hz)"
            discovered_bands[band_name] = (max(f_low, min_freq_filter), f_high) # Clamp low end to min_freq_filter
    
    print(f"Discovered {len(discovered_bands)} prominent bands within the filtered range.")
    return discovered_bands

# ─────────────────────────────────────────────────────────────────────────────
# Laplacian conv (used in the PDE)
# ─────────────────────────────────────────────────────────────────────────────
class Laplacian2D(nn.Module):
    """
    Computes the 2D Laplacian of a tensor using a fixed convolution kernel.
    Applies circular padding to handle boundary conditions.
    """
    def __init__(self):
        super().__init__()
        # Define the 2D Laplacian kernel
        k = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
        # Register it as a buffer so it's moved to the correct device with the model
        self.register_buffer('k', k.view(1, 1, 3, 3)) # Reshape for conv2d

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Applies the Laplacian operation.
        Args:
            u (torch.Tensor): Input tensor (batch, height, width).
        Returns:
            torch.Tensor: Tensor after Laplacian operation.
        """
        # Add a channel dimension for conv2d (batch, 1, height, width)
        u_4d = u.unsqueeze(1) 
        # Apply circular padding to simulate periodic boundary conditions
        u_padded = F.pad(u_4d, (1, 1, 1, 1), mode='circular')
        # Perform 2D convolution
        lap = F.conv2d(u_padded, self.k, padding=0)
        # Remove the channel dimension
        return lap.squeeze(1)

# ─────────────────────────────────────────────────────────────────────────────
# Entropic field model (PDE)
# ─────────────────────────────────────────────────────────────────────────────
class EntropicFieldNN(nn.Module):
    """
    Implements the reaction-diffusion PDE for activator (u) and inhibitor (v) fields.
    The parameters D, gamma, kappa, tau are learned and constrained within physical ranges.
    """
    def __init__(self):
        super().__init__()
        # Raw, unbounded parameters for stable optimization. These will be transformed
        # into physical parameters using sigmoid activations.
        self.raw_D = nn.Parameter(torch.randn(())) # Diffusion coefficient
        self.raw_gamma = nn.Parameter(torch.randn(())) # Activator feed/growth rate
        self.raw_kappa = nn.Parameter(torch.randn(())) # Inhibitor removal/damping rate
        self.raw_tau = nn.Parameter(torch.randn(())) # Inhibitor timescale
        
        # Instantiate the Laplacian operator
        self.lap = Laplacian2D()

    def physical(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Transforms raw parameters using sigmoid activation to constrain them within
        physically meaningful ranges.
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                D (diffusion), gamma (activator growth), kappa (inhibitor decay),
                tau (inhibitor timescale).
        """
        # Ranges based on typical reaction-diffusion system behavior and biological plausibility
        D = 0.1 + 0.9 * torch.sigmoid(self.raw_D)     # D in [0.1, 1.0]
        gamma = 0.5 + 1.5 * torch.sigmoid(self.raw_gamma) # gamma in [0.5, 2.0]
        kappa = 0.05 + 0.2 * torch.sigmoid(self.raw_kappa) # kappa in [0.05, 0.25]
        tau = 0.5 + 1.5 * torch.sigmoid(self.raw_tau)   # tau in [0.5, 2.0]
        return D, gamma, kappa, tau

    def step(self, u: torch.Tensor, v: torch.Tensor, dt: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs one time step of the reaction-diffusion PDE.
        Equations from EEG (2).tex:
        du/dt = D * nabla^2 u - u * v^2 + gamma * (1 - u)
        dv/dt = (1/tau) * (nabla^2 v + u * v^2 - (gamma + kappa) * v)
        Args:
            u (torch.Tensor): Current state of the activator field.
            v (torch.Tensor): Current state of the inhibitor field.
            dt (float): Time step size.
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                New u field, New v field, and the calculated du (change in u).
        """
        D, gamma, kappa, tau = self.physical()
        
        lap_u, lap_v = self.lap(u), self.lap(v)
        uv2 = u * v.pow(2) # u * v^2 term

        # Calculate changes for u and v
        du = dt * (D * lap_u - uv2 + gamma * (1 - u))
        dv = (dt / tau) * (lap_v + uv2 - (gamma + kappa) * v)
        
        # Update fields
        return u + du, v + dv, du

# ─────────────────────────────────────────────────────────────────────────────
# Electrode placement
# ─────────────────────────────────────────────────────────────────────────────

def place_electrodes(nx: int, n_ch: int, mode: str) -> Tuple[list[int], list[int]]:
    """
    Calculates virtual electrode coordinates on the 2D grid.
    Args:
        nx (int): Size of the square grid (nx x nx).
        n_ch (int): Number of channels (virtual electrodes).
        mode (str): 'circular' or 'grid' placement strategy.
    Returns:
        Tuple[list[int], list[int]]: Lists of x and y coordinates for electrodes.
    """
    if mode == 'grid':
        # Arrange electrodes in a square grid formation
        grid_size = int(math.ceil(math.sqrt(n_ch)))
        if grid_size == 0: return [], []
        xs = torch.linspace(nx * 0.2, nx * 0.8, grid_size).long()
        ys = xs.clone()
        coords = [(int(x), int(y)) for x in xs for y in ys][:n_ch]
    else: # 'circular'
        # Arrange electrodes in a circle
        center = nx // 2
        radius = nx // 3
        coords = [
            (int(center + radius * math.cos(2 * math.pi * i / n_ch)),
             int(center + radius * math.sin(2 * math.pi * i / n_ch)))
            for i in range(n_ch)
        ]
    # Return as separate lists of x and y coordinates, suitable for advanced indexing
    return list(zip(*coords))  # Returns (xs_list, ys_list)

# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def plot_eeg_timeseries(out_dir: pathlib.Path, eeg_features: Dict):
    """
    Plots a snippet of the raw input EEG time series for visualization.
    Args:
        out_dir (pathlib.Path): Directory to save the plot.
        eeg_features (Dict): Dictionary containing EEG data and metadata.
    """
    data = eeg_features['raw_data'][:, :int(min(5.0, eeg_features['raw_data'].shape[1] / eeg_features['sfreq']) * eeg_features['sfreq'])]
    time_vector = np.arange(data.shape[1]) / eeg_features['sfreq']
    
    fig, axs = plt.subplots(eeg_features['n_channels'], 1, figsize=(12, eeg_features['n_channels'] * 1.3), sharex=True)
    if eeg_features['n_channels'] == 1: 
        axs = [axs] # Ensure axs is iterable even for a single channel
    
    fig.suptitle('Raw Input Time Series (First 5 seconds or less)', fontsize=16)
    for i, ax in enumerate(axs):
        ax.plot(time_vector, data[i], linewidth=0.7)
        ax.set_ylabel(eeg_features['ch_names'][i], rotation=0, labelpad=20, ha='right')
        ax.grid(True, linestyle=':', alpha=0.6)
    
    axs[-1].set_xlabel('Time (s)')
    output_path = out_dir / 'input_timeseries.png'
    fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent suptitle overlap
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Saved input time series plot to {output_path}")

def plot_signature(out_dir: pathlib.Path, epoch: int, model: EntropicFieldNN, sim_fields: dict, virtual_electrodes: Tuple[torch.Tensor, torch.Tensor]):
    """
    Generates a 2x2 plot summarizing the learned parameters and the simulated field states.
    Args:
        out_dir (pathlib.Path): Directory to save the plot.
        epoch (int): Current training epoch.
        model (EntropicFieldNN): The trained PINN model.
        sim_fields (dict): Dictionary containing mean simulated field states (u, v, uv2, du).
        virtual_electrodes (Tuple[torch.Tensor, torch.Tensor]): x and y coordinates of virtual electrodes.
    """
    D, gamma, kappa, tau = [x.item() for x in model.physical()]
    
    fig, axs = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)
    axs = axs.flatten()
    
    param_string = f"D={D:.4f}, γ={gamma:.4f}, κ={kappa:.4f}, τ={tau:.4f}"
    fig.suptitle(f"Dynamical Signature after {epoch} Epochs\nParameters: {param_string}", fontsize=18)
    
    field_data_and_titles = [
        (sim_fields['u'], 'Activator Field ($u$)'),
        (sim_fields['v'], 'Inhibitor Field ($v$)'),
        (sim_fields['uv2'], 'Interaction Term ($u \\cdot v^2$)'), # Fixed: Escaped backslash
        (sim_fields['du'], 'Activator Change ($du/dt$)')
    ]
    
    cmaps = ['viridis', 'plasma', 'cividis', 'magma'] # Choose perceptually uniform colormaps
    
    # Plotting each field
    for i, (field_tensor, title) in enumerate(field_data_and_titles):
        ax = axs[i]
        field_np = field_tensor.cpu().numpy()
        
        # Plot the field as an image
        im = ax.imshow(field_np, cmap=cmaps[i], origin='lower', extent=[0, field_np.shape[1], 0, field_np.shape[0]])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04) # Add colorbar
        
        # Overlay electrode positions if they exist
        if virtual_electrodes[0] is not None and len(virtual_electrodes[0]) > 0:
            ax.scatter(virtual_electrodes[0].cpu().numpy(), virtual_electrodes[1].cpu().numpy(), 
                       color='red', marker='o', s=50, edgecolors='black', label='Virtual Electrodes')
            if i == 0: # Add legend only once
                ax.legend(loc='upper right')

        ax.set_title(title, fontsize=14)
        ax.set_xticks([]) # Remove x-axis ticks
        ax.set_yticks([]) # Remove y-axis ticks
        ax.set_aspect('equal') # Ensure aspect ratio is square

    output_path = out_dir / f"signature_epoch_{epoch}.png"
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Saved dynamical signature plot to {output_path}")

def plot_spectral_fit(out_dir: pathlib.Path, epoch: int, args: argparse.Namespace, 
                      eeg_mean_psd: torch.Tensor, eeg_freqs: torch.Tensor, 
                      sim_mean_psd: torch.Tensor, sim_freqs: torch.Tensor):
    """
    Plots the mean power spectral density (PSD) of real EEG data and simulated data
    for comparison, focusing on the specified frequency range.
    Args:
        out_dir (pathlib.Path): Directory to save the plot.
        epoch (int): Current training epoch.
        args (argparse.Namespace): Command-line arguments containing min_freq_hz.
        eeg_mean_psd (torch.Tensor): Normalized mean PSD from real EEG data.
        eeg_freqs (torch.Tensor): Frequencies corresponding to eeg_mean_psd.
        sim_mean_psd (torch.Tensor): Normalized mean PSD from simulated data.
        sim_freqs (torch.Tensor): Frequencies corresponding to sim_mean_psd.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)
    
    ax.plot(eeg_freqs.cpu().numpy(), eeg_mean_psd.cpu().numpy(), 
            label='Real EEG Mean PSD', color='blue', linewidth=2)
    ax.plot(sim_freqs.cpu().numpy(), sim_mean_psd.cpu().numpy(), 
            label='Simulated Mean PSD', color='red', linestyle='--', linewidth=2)
    
    ax.set_title(f"Spectral Fit at Epoch {epoch}", fontsize=16)
    ax.set_xlabel("Frequency (Hz)", fontsize=12)
    ax.set_ylabel("Normalized Power Spectral Density", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.7)
    
    # Set X-axis limits based on min_freq_hz and 100 Hz
    ax.set_xlim(args.min_freq_hz, 100)
    
    output_path = out_dir / f"spectral_fit_epoch_{epoch}.png"
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Saved spectral fit plot to {output_path}")

# ─────────────────────────────────────────────────────────────────────────────
# EEG loader
# ─────────────────────────────────────────────────────────────────────────────

def load_and_prepare_eeg(args: argparse.Namespace, device: torch.device) -> Dict:
    """
    Loads multi-channel EEG/MEA data, detrends it, computes PSDs, and extracts
    mean power spectrum and topographic maps for specified frequency bands.
    Args:
        args (argparse.Namespace): Command-line arguments.
        device (torch.device): The device (CPU/CUDA) to load data onto.
    Returns:
        Dict: A dictionary containing prepared EEG features and metadata.
    """
    filepath = args.eeg_file
    if not filepath.is_file():
        raise FileNotFoundError(f"EEG file not found at {filepath}")

    # Handle .csv vs MNE-supported formats
    if filepath.suffix.lower() == '.csv':
        if args.sfreq is None:
            raise ValueError("--sfreq argument is required for CSV files.")
        df = pd.read_csv(filepath, header=0, index_col=0)
        data_np = df.values.T # Transpose to (channels, samples)
        ch_names = df.columns.tolist()
        sfreq = args.sfreq
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
        raw = mne.io.RawArray(data_np, info, verbose=False)
    else:
        raw = mne.io.read_raw(filepath, preload=True, verbose=False)

    # Pick channels based on user input, or all EEG channels by default
    if args.channels:
        # Validate channel indices
        if any(ch_idx >= len(raw.ch_names) for ch_idx in args.channels):
            raise ValueError(f"Invalid channel index in {args.channels}. Max index is {len(raw.ch_names) - 1}.")
        ch_names_to_pick = [raw.ch_names[i] for i in args.channels]
        raw.pick(ch_names_to_pick)
    else:
        raw.pick('eeg') # Default to all EEG channels

    print(f"Selected {len(raw.ch_names)} channels: {', '.join(raw.ch_names)}")
    
    # Get data and metadata
    data_np = raw.get_data() # (n_channels, n_samples)
    sfreq = raw.info['sfreq']
    n_channels = data_np.shape[0]
    n_samples = data_np.shape[1]

    # Convert to PyTorch tensor and detrend (remove mean)
    eeg_data_tensor = torch.tensor(data_np, dtype=torch.float32, device=device)
    eeg_data_tensor -= eeg_data_tensor.mean(dim=1, keepdim=True)

    # Compute Power Spectral Density (PSD) using PyTorch's rfft
    fft_result = torch.fft.rfft(eeg_data_tensor, n=n_samples, dim=-1)
    psds_per_channel = torch.abs(fft_result).pow(2) # PSD = |FFT|^2

    # Get frequencies for the FFT result
    fft_freqs = torch.fft.rfftfreq(n_samples, 1.0 / sfreq, device=device)

    # Mask frequencies based on min_freq_hz and upper limit (100 Hz)
    freq_mask = (fft_freqs >= args.min_freq_hz) & (fft_freqs <= 100)
    psds_filtered = psds_per_channel[:, freq_mask]
    freqs_filtered = fft_freqs[freq_mask]

    # Compute mean PSD across channels
    mean_psd = torch.mean(psds_filtered, dim=0)
    # Normalize mean PSD to its maximum for consistent loss calculation
    mean_psd_norm = mean_psd / (torch.max(mean_psd) + 1e-9)

    # Determine frequency bands to use, passing the new min_freq_hz
    if args.mode == 'discovered':
        bands_to_use = find_prominent_bands(freqs_filtered.cpu().numpy(), mean_psd.cpu().numpy(), args.prominence_std, args.min_freq_hz)
    else:
        # Filter canonical bands to respect min_freq_hz
        bands_to_use = {}
        for band_name, (f_low, f_high) in CANONICAL_BANDS.items():
            # Only include bands if their upper bound is above min_freq_hz
            # and clamp their lower bound to min_freq_hz if necessary
            if f_high >= args.min_freq_hz:
                bands_to_use[band_name] = (max(f_low, args.min_freq_hz), f_high)

    # Compute topographic maps for each band
    topo_maps = {}
    if not bands_to_use: # Handle case where no bands are found after filtering
        print("[WARN] No valid frequency bands found or selected after filtering. Topographic loss will be 0.")
    for band_name, band_freqs in bands_to_use.items():
        band_mask = (freqs_filtered >= band_freqs[0]) & (freqs_filtered <= band_freqs[1])
        # Sum power within the band for each channel
        power_per_channel_in_band = torch.sum(psds_filtered[:, band_mask], dim=1)
        # Normalize topographic map for consistent loss calculation
        total_power_in_band = torch.sum(power_per_channel_in_band)
        topo_map_norm = power_per_channel_in_band / (total_power_in_band + 1e-9)
        topo_maps[band_name] = topo_map_norm

    return {
        "mean_psd": mean_psd_norm, # Normalized mean power spectral density
        "topo_maps": topo_maps,     # Dictionary of normalized topographic maps per band
        "freqs": freqs_filtered,    # Frequencies corresponding to the PSDs
        "ch_names": raw.ch_names,   # Names of selected channels
        "n_channels": n_channels,   # Number of selected channels
        "raw_data": data_np,        # Raw data (numpy array, for initial plotting)
        "sfreq": sfreq,             # Sampling frequency
        "bands_used": bands_to_use  # Actual bands used for analysis
    }

# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace):
    """
    Main training loop for the EEG Spatiotemporal PINN.
    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    device = get_device()
    # Enable cuDNN auto-tuner for optimal performance if on CUDA
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        if args.tf32: # Allow TF32 computation if requested (for Ampere+ GPUs)
            torch.backends.cuda.matmul.allow_tf32 = True

    # Seed for reproducibility if specified
    if args.seed:
        seed_everything(args.seed)

    # Setup output directory for this run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir / f"run_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving results to: {out_dir}")

    # Load and prepare EEG data
    eeg_features = load_and_prepare_eeg(args, device)
    
    # Plot initial EEG time series if not disabled
    if not args.no_plots:
        plot_eeg_timeseries(out_dir, eeg_features)

    # Place virtual electrodes on the grid
    ve_x_list, ve_y_list = place_electrodes(args.nx, eeg_features['n_channels'], args.placement)
    ve_x = torch.tensor(ve_x_list, device=device)
    ve_y = torch.tensor(ve_y_list, device=device)
    
    # Instantiate the PINN model
    model = EntropicFieldNN().to(device)
    # Compile the model for performance if requested (PyTorch 2.0+)
    if args.compile:
        model = torch.compile(model, fullgraph=True)
        print('[torch.compile] enabled for the model.')

    # Setup optimizer, learning rate scheduler, and gradient scaler for AMP
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=args.scheduler_patience, factor=args.scheduler_factor)
    scaler = GradScaler(enabled=args.use_amp)

    # Setup CSV logging for training progress
    csv_path = out_dir / 'training_log.csv'
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['epoch', 'total_loss', 'spectral_loss', 'topographic_loss', 'D', 'gamma', 'kappa', 'tau', 'lr'])
    print(f"Training log will be saved to: {csv_path}")

    print(f"\n--- Starting Spatiotemporal Training for {args.epochs} epochs ---")
    # Training loop with tqdm for progress bar
    for epoch in tqdm(range(1, args.epochs + 1), ncols=80, desc="Training Progress"):
        optimizer.zero_grad(set_to_none=True) # Reset gradients
        
        # Initialize u and v fields with some noise for batch diversity
        noise = 0.1 * (torch.rand(args.batch, args.nx, args.nx, device=device) - 0.5)
        u_sim = 0.5 + noise
        v_sim = 0.5 - noise

        # Enable Automatic Mixed Precision if use_amp is true
        with autocast(device_type=device.type, enabled=args.use_amp):
            # Simulate the PDE for 'steps' timesteps
            # Allocate tensor to store simulated time series at virtual electrodes
            simulated_time_series_at_ve = torch.zeros(args.batch, eeg_features['n_channels'], args.steps, device=device)
            last_du_field = None # To capture the du field from the last timestep for plotting

            for t in range(args.steps):
                u_sim, v_sim, du = model.step(u_sim, v_sim, args.dt)
                simulated_time_series_at_ve[:, :, t] = u_sim[:, ve_x, ve_y]
                last_du_field = du # Keep track of du from the last step

                # Implement Backpropagation Through Time (BPTT)
                if (t + 1) % args.bptt == 0:
                    # Detach to cut the computational graph, then enable gradients again
                    # This prevents memory issues for very long simulations
                    u_sim = u_sim.detach().requires_grad_()
                    v_sim = v_sim.detach().requires_grad_()

            # Compute simulated PSDs and frequencies from virtual electrode time series
            sim_fft_ve = torch.fft.rfft(simulated_time_series_at_ve, n=args.steps, dim=-1)
            sim_psds_ve = torch.abs(sim_fft_ve).pow(2) # (batch, channels, freqs)
            
            sim_freqs = torch.fft.rfftfreq(args.steps, args.dt, device=device)
            
            # Filter simulated PSDs and frequencies to the desired range
            freq_mask_sim = (sim_freqs >= args.min_freq_hz) & (sim_freqs <= 100)
            sim_psds_filtered = sim_psds_ve[:, :, freq_mask_sim]
            sim_freqs_filtered = sim_freqs[freq_mask_sim]

            # Ensure simulated PSDs match the target frequency bins (from EEG data)
            # This is crucial for valid MSE loss calculation
            target_freq_bins = eeg_features['mean_psd'].shape[-1]
            if sim_psds_filtered.shape[-1] != target_freq_bins:
                sim_psds_filtered = ensure_psd_shape(sim_psds_filtered, target_freq_bins, device)
            
            # Calculate mean simulated PSD across channels and batches
            sim_mean_psd = torch.mean(sim_psds_filtered, dim=(0, 1)) # Mean across batch and channels
            sim_mean_psd_norm = sim_mean_psd / (torch.max(sim_mean_psd) + 1e-9)

            # Calculate topographic loss for each band
            topo_losses = []
            bands_used = eeg_features['bands_used']
            if not bands_used: # Handle case where no bands are discovered or valid after filtering
                loss_topo = torch.tensor(0.0, device=device)
            else:
                for band_name, band_freqs in bands_used.items():
                    # Create a mask for the current band in the simulated frequencies
                    sim_band_mask = (sim_freqs_filtered >= band_freqs[0]) & (sim_freqs_filtered <= band_freqs[1])
                    
                    # Sum power within the band for each simulated channel (averaged over batch)
                    sim_power_per_channel = torch.mean(torch.sum(sim_psds_filtered[:, :, sim_band_mask], dim=2), dim=0)
                    
                    # Normalize simulated topographic map
                    sim_topo_map_norm = sim_power_per_channel / (torch.sum(sim_power_per_channel) + 1e-9)
                    
                    # Get the target topographic map for this band from real EEG data
                    target_topo_map = eeg_features['topo_maps'][band_name]
                    
                    # Add MSE loss for this band's topography
                    topo_losses.append(F.mse_loss(sim_topo_map_norm, target_topo_map))
                
                loss_topo = torch.mean(torch.stack(topo_losses))

            # Calculate spectral loss
            loss_spectral = F.mse_loss(sim_mean_psd_norm, eeg_features['mean_psd'])
            
            # Calculate total loss with weights
            total_loss = args.w_spec * loss_spectral + args.w_topo * loss_topo

        # Backpropagation and optimizer step with gradient scaling for AMP
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step(total_loss.item()) # Update learning rate based on total loss

        # Log training progress to CSV
        D, gamma, kappa, tau = [x.item() for x in model.physical()]
        current_lr = optimizer.param_groups[0]['lr']
        csv_writer.writerow([
            epoch, total_loss.item(), loss_spectral.item(), loss_topo.item(),
            D, gamma, kappa, tau, current_lr
        ])
        csv_file.flush() # Ensure data is written immediately

        # Print progress and parameters every 100 epochs
        if epoch % 100 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{args.epochs} | Loss: {total_loss.item():.4e} "
                  f"| Spec Loss: {loss_spectral.item():.4e} | Topo Loss: {loss_topo.item():.4e} "
                  f"| LR: {current_lr:.2e} | D: {D:.3f}, γ: {gamma:.3f}, κ: {kappa:.3f}, τ: {tau:.3f}")

        # Plot dynamical signature and spectral fit every 'plot_interval' epochs or at epoch 1
        if not args.no_plots and (epoch % args.plot_interval == 0 or epoch == 1):
            with torch.no_grad(): # Disable gradient calculation for plotting
                # Average fields across the batch for visualization
                mean_u_field = torch.mean(u_sim, dim=0).detach()
                mean_v_field = torch.mean(v_sim, dim=0).detach()
                mean_uv2_field = (mean_u_field * mean_v_field**2).detach() # Recalculate based on mean u, v
                mean_du_field = torch.mean(last_du_field, dim=0).detach() if last_du_field is not None else torch.zeros_like(mean_u_field)

                sim_fields = {
                    'u': mean_u_field,
                    'v': mean_v_field,
                    'uv2': mean_uv2_field,
                    'du': mean_du_field
                }
                plot_signature(out_dir, epoch, model, sim_fields, (ve_x, ve_y))
                
                # Plot the spectral fit
                plot_spectral_fit(out_dir, epoch, args, 
                                  eeg_features['mean_psd'], eeg_features['freqs'], 
                                  sim_mean_psd_norm, sim_freqs_filtered)
                
    csv_file.close() # Close the log file at the end of training
    print(f"\n--- Training Finished ---")
    print(f"Results saved to: {out_dir}")

# ─────────────────────────────────────────────────────────────────────────────
# CLI (Command Line Interface)
# ─────────────────────────────────────────────────────────────────────────────

def parse():
    """
    Parses command-line arguments for the EEG PINN script.
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    p = argparse.ArgumentParser(
        description="Spatiotemporal PINN for constraining Entropic Field Theories with EEG/MEA data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Required argument
    p.add_argument('eeg_file', type=pathlib.Path,
                   help='Path to multi-channel data file (e.g., .fif, .csv, .edf, .bdf).')
    
    # Data loading arguments
    p.add_argument('--sfreq', type=float, default=None,
                   help='Sampling frequency (Hz). REQUIRED for .csv input files.')
    p.add_argument('--channels', type=int, nargs='+', default=None,
                   help='Indices of channels to use (e.g., 0 1 2 for first three). '
                        'If None, all EEG channels will be used.')
    p.add_argument('--min_freq_hz', type=float, default=0.0,
                   help='Minimum frequency (Hz) to include in analysis for both real and simulated data.')
    
    # Output and plotting arguments
    p.add_argument('--output_dir', type=pathlib.Path, default=pathlib.Path('./eeg_pinn_runs'),
                   help='Base directory where individual run folders will be created.')
    p.add_argument('--no_plots', action='store_true',
                   help='Disable all matplotlib plots during and after training.')
    p.add_argument('--plot_interval', type=int, default=100,
                   help='Interval (in epochs) at which to generate signature plots.')

    # Frequency band discovery arguments
    p.add_argument('--mode', type=str, default='canonical', choices=['canonical', 'discovered'],
                   help='Method for defining frequency bands: '
                        '"canonical" (delta, theta, alpha, beta, gamma) or '
                        '"discovered" (data-driven peak detection).')
    p.add_argument('--prominence_std', type=float, default=2.0,
                   help='For "discovered" mode: minimum peak prominence threshold '
                        '(in standard deviations of the noise floor).')
    
    # Loss function weights
    p.add_argument('--w_spec', type=float, default=1.0,
                   help='Weight for the spectral (mean PSD) loss component.')
    p.add_argument('--w_topo', type=float, default=2.0,
                   help='Weight for the topographic (spatial power map) loss component.')
    
    # Simulation and training parameters
    p.add_argument('--batch', type=int, default=4,
                   help='Number of independent PDE simulations per training batch.')
    p.add_argument('--epochs', type=int, default=5000,
                   help='Total number of training epochs.')
    p.add_argument('--steps', type=int, default=2048,
                   help='Number of simulation timesteps per epoch.')
    p.add_argument('--nx', type=int, default=64,
                   help='Spatial grid size (nx x nx).')
    p.add_argument('--dt', type=float, default=0.02,
                   help='Simulation time step size (e.g., 0.02 for 50 steps/second).')
    p.add_argument('--bptt', type=int, default=256,
                   help='Backpropagation Through Time (BPTT) interval. '
                        'Set to 0 or `steps` to disable chunking.')
    
    # Optimizer and device settings
    p.add_argument('--lr', type=float, default=1e-4,
                   help='Initial learning rate for the Adam optimizer.')
    p.add_argument('--scheduler_patience', type=int, default=200,
                   help='Patience for ReduceLROnPlateau scheduler.')
    p.add_argument('--scheduler_factor', type=float, default=0.1,
                   help='Factor by which LR is reduced on plateau for scheduler.')
    p.add_argument('--placement', type=str, default='circular', choices=['circular', 'grid'],
                   help='Virtual electrode placement strategy on the 2D grid.')
    p.add_argument('--use_amp', action='store_true',
                   help='Enable Automatic Mixed Precision (AMP) training for faster computation on CUDA.')
    p.add_argument('--compile', action='store_true',
                   help='Enable PyTorch 2.0 torch.compile for potential performance gains.')
    p.add_argument('--tf32', action='store_true',
                   help='Allow TF32 computation (for Ampere+ GPUs) if using CUDA.')
    p.add_argument('--seed', type=int, default=None,
                   help='Random seed for reproducibility.')

    return p.parse_args()

if __name__ == '__main__':
    args = parse()
    train(args)
