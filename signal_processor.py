import numpy as np
from scipy.signal import butter, filtfilt, welch
from eeg_simulator import simulate_eeg, BANDS, MENTAL_STATES

def bandpass_filter(signal, low, high, sample_rate=256, order=4):
    nyq = sample_rate / 2
    low = max(low / nyq, 0.001)
    high = min(high / nyq, 0.999)
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def extract_band_power(signal, sample_rate=256):
    freqs, psd = welch(signal, fs=sample_rate, nperseg=256)
    band_powers = {}
    for band, (flo, fhi) in BANDS.items():
        idx = np.where((freqs >= flo) & (freqs <= fhi))
        band_powers[band] = float(np.trapezoid(psd[idx], freqs[idx]))
    return band_powers

def process_eeg(eeg_data, sample_rate=256):
    all_features = []
    for ch in range(eeg_data.shape[0]):
        filtered = bandpass_filter(eeg_data[ch], 0.5, 100, sample_rate)
        powers = extract_band_power(filtered, sample_rate)
        all_features.extend(powers.values())
    return np.array(all_features)

def normalize_features(feature_vector):
    min_val = feature_vector.min()
    max_val = feature_vector.max()
    if max_val - min_val == 0:
        return feature_vector
    return (feature_vector - min_val) / (max_val - min_val)

if __name__ == "__main__":
    print("=" * 50)
    print("  NeuralSearch - Step 2: Signal Processing")
    print("=" * 50)
    for state in MENTAL_STATES:
        print(f"\n  Processing state: [{state}]")
        t, eeg_data, label = simulate_eeg(state)
        features = normalize_features(process_eeg(eeg_data))
        print(f"    Raw EEG shape   : {eeg_data.shape}")
        print(f"    Feature vector  : {features.shape[0]} features")
        print(f"    Feature range   : [{features.min():.3f}, {features.max():.3f}]")
        print(f"    Sample features : {features[:5].round(3)}")
    print("\n  Step 2 complete. Ready for Step 3.")
