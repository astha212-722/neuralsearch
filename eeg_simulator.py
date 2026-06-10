"""
NeuralSearch - Step 1: EEG Signal Simulator
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# Constants
SAMPLE_RATE = 256
DURATION    = 4
N_CHANNELS  = 8

# Brainwave bands
BANDS = {
    "delta": (0.5,  4),
    "theta": (4,    8),
    "alpha": (8,   13),
    "beta":  (13,  30),
    "gamma": (30, 100),
}

# Mental states — each has a different brainwave mix
MENTAL_STATES = {
    "relaxed":        {"delta": 0.2, "theta": 0.4, "alpha": 0.9, "beta": 0.2, "gamma": 0.1},
    "focused":        {"delta": 0.1, "theta": 0.2, "alpha": 0.4, "beta": 0.9, "gamma": 0.5},
    "searching":      {"delta": 0.1, "theta": 0.3, "alpha": 0.5, "beta": 0.8, "gamma": 0.7},
    "recall":         {"delta": 0.2, "theta": 0.7, "alpha": 0.6, "beta": 0.5, "gamma": 0.4},
    "visual_imagery": {"delta": 0.1, "theta": 0.5, "alpha": 0.7, "beta": 0.6, "gamma": 0.8},
    "idle":           {"delta": 0.5, "theta": 0.3, "alpha": 0.6, "beta": 0.2, "gamma": 0.1},
}


def generate_band_signal(freq_low, freq_high, amplitude, duration, sample_rate):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = np.zeros_like(t)
    for freq in np.linspace(freq_low, freq_high, 6):
        phase = np.random.uniform(0, 2 * np.pi)
        signal += np.sin(2 * np.pi * freq * t + phase)
    return t, signal * amplitude


def add_noise(signal, noise_level=0.05):
    noise = np.random.normal(0, noise_level, len(signal))
    t = np.linspace(0, 1, len(signal))
    drift = 0.03 * np.sin(2 * np.pi * 0.1 * t + np.random.uniform(0, np.pi))
    return signal + noise + drift


def simulate_eeg(mental_state, duration=DURATION, sample_rate=SAMPLE_RATE, n_channels=N_CHANNELS):
    if mental_state not in MENTAL_STATES:
        raise ValueError(f"Unknown state. Choose from: {list(MENTAL_STATES.keys())}")
    profile = MENTAL_STATES[mental_state]
    n_samples = int(sample_rate * duration)
    eeg_data = np.zeros((n_channels, n_samples))
    for ch in range(n_channels):
        channel_signal = np.zeros(n_samples)
        for band, (flo, fhi) in BANDS.items():
            amp = profile[band] * np.random.uniform(0.85, 1.15)
            _, sig = generate_band_signal(flo, fhi, amp, duration, sample_rate)
            channel_signal += sig
        eeg_data[ch] = add_noise(channel_signal)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    return t, eeg_data, mental_state


def plot_eeg(t, eeg_data, mental_state, save_path=None):
    n_channels = eeg_data.shape[0]
    channel_names = ["Fp1","Fp2","F3","F4","C3","C4","P3","P4"][:n_channels]
    fig = plt.figure(figsize=(14, 8), facecolor="#0d0d0d")
    fig.suptitle(f"NeuralSearch — EEG | State: {mental_state.upper()}",
                 color="white", fontsize=14, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(n_channels, 1, hspace=0.08)
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, n_channels))
    for i in range(n_channels):
        ax = fig.add_subplot(gs[i])
        ax.plot(t, eeg_data[i], color=colors[i], linewidth=0.7, alpha=0.9)
        ax.set_facecolor("#0d0d0d")
        ax.set_ylabel(channel_names[i], color="white", fontsize=8, rotation=0, labelpad=25)
        ax.tick_params(colors="gray", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")
        if i < n_channels - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Time (seconds)", color="white", fontsize=9)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
        print(f"  Saved → {save_path}")
    plt.show()


# ── Run it ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    print("=" * 50)
    print("  NeuralSearch — Step 1: EEG Simulator")
    print("=" * 50)
    for state in MENTAL_STATES:
        print(f"\n  Simulating: [{state}]")
        t, eeg_data, label = simulate_eeg(state)
        print(f"    Samples : {eeg_data.shape[1]}  |  Channels : {eeg_data.shape[0]}")
        plot_eeg(t, eeg_data, label, save_path=f"output/eeg_{state}.png")
    print("\n  Step 1 complete. Ready for Step 2.") ^X


