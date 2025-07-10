import numpy as np
from scipy import signal as scipysig
from scipy import stats
import math
import matplotlib.pyplot as plt
import torch

import numpy as np
from scipy.optimize import least_squares
from numpy.fft import fft, ifft
from scipy.signal import correlate

def get_optimal_device():
    """Get the best available device for PyTorch operations."""
    if torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon GPU
    elif torch.cuda.is_available():
        return torch.device("cuda")  # NVIDIA GPU
    else:
        return torch.device("cpu")

def cross_correlation(sig1, sig2, use_pytorch=True):
    """Compute cross-correlation between two signals using PyTorch FFT for Apple Silicon GPU acceleration.
    
    Args:
        sig1: First signal array
        sig2: Second signal array
        use_pytorch: Whether to use PyTorch FFT (default: True)
        
    Returns:
        cc: Cross-correlation array
        lags: Corresponding lag values
    """
    if use_pytorch:
        device = get_optimal_device()
        
        # Convert to PyTorch tensors if needed
        if isinstance(sig1, np.ndarray):
            sig1_torch = torch.from_numpy(sig1).float().to(device)
        else:
            sig1_torch = sig1.float().to(device)
            
        if isinstance(sig2, np.ndarray):
            sig2_torch = torch.from_numpy(sig2).float().to(device)
        else:
            sig2_torch = sig2.float().to(device)
        
        n = sig1_torch.shape[0] + sig2_torch.shape[0]
        
        # PyTorch FFT operations
        SIG1 = torch.fft.rfft(sig1_torch, n=n)
        SIG2 = torch.fft.rfft(sig2_torch, n=n)
        
        R = SIG1 * torch.conj(SIG2)
        cc = torch.fft.irfft(R, n=n)
        
        max_shift = int(n / 2)
        cc = torch.cat([cc[-max_shift:], cc[:max_shift+1]])
        
        # Convert back to numpy
        cc = cc.cpu().numpy()
        lags = np.arange(-max_shift, max_shift + 1)
        
    else:
        # Original NumPy implementation
        n = sig1.shape[0] + sig2.shape[0]
        SIG1 = np.fft.rfft(sig1, n=n)
        SIG2 = np.fft.rfft(sig2, n=n)
        
        R = SIG1 * np.conj(SIG2)
        cc = np.fft.irfft(R, n=n)
        
        max_shift = int(n / 2)
        cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))
        lags = np.arange(-max_shift, max_shift + 1)
    
    return cc, lags

def compute_tdoa(sig1, sig2, fs):
    """Estimate TDoA between two signals using cross-correlation.
    
    Args:
        sig1: First signal array
        sig2: Second signal array  
        fs: Sampling frequency
        
    Returns:
        tdoa: Time difference of arrival in seconds
    """
    cc, lags = cross_correlation(sig1, sig2)
    
    lag = lags[np.argmax(cc)]
    tdoa = lag / float(fs)

    return tdoa  # in seconds

def estimate_doa(tdoa, mic_1, mic_2, sound_speed=343.0):
    """Estimate the angle of arrival (in radians) from TDoA."""
    delta_d = tdoa * sound_speed
    # Clamp sin_theta to [-1, 1] to avoid NaNs

    mic_distance = np.linalg.norm(mic_2.position - mic_1.position)

    sin_theta = np.clip(delta_d / mic_distance, -1.0, 1.0)
    theta = np.arcsin(sin_theta)
    return theta  # in radians

def get_angle_adjustment_strategy(angle_offset_deg, min_doa_angle):
    """
    Determine how to combine angle offset and DoA based on the original logic.
    
    Args:
        angle_offset_deg: Angle offset in degrees
        min_doa_angle: DoA angle in radians
        
    Returns:
        str: Strategy identifier for angle calculation
    """
    if (angle_offset_deg == 90 and min_doa_angle >= 0):
        return "offset_minus_doa"
    elif (min_doa_angle <= 0 and angle_offset_deg == -90):
        return "offset_minus_doa"
    elif angle_offset_deg == 90 and min_doa_angle <= 0:
        return "offset_minus_doa"
    elif min_doa_angle < 0 and angle_offset_deg > 0:
        return "offset_plus_doa"
    elif min_doa_angle > 0 and angle_offset_deg < 0:
        return "offset_minus_doa"
    else:
        return "doa_minus_offset"

def apply_angle_strategy(strategy, angle_offset, min_doa_angle):
    """
    Apply the determined strategy to calculate final angle.
    
    Args:
        strategy: Strategy identifier from get_angle_adjustment_strategy
        angle_offset: Angle offset in radians
        min_doa_angle: DoA angle in radians
        
    Returns:
        float: Final angle in degrees
    """
    strategies = {
        "offset_minus_doa": lambda: np.degrees(angle_offset - min_doa_angle),
        "offset_plus_doa": lambda: np.degrees(angle_offset + min_doa_angle),
        "doa_minus_offset": lambda: np.degrees(min_doa_angle - angle_offset)
    }
    return strategies[strategy]()

def estimate_doa_from_recordings(mic_array, sound_speed=343.0):
    """
    Estimate 3D DOA (azimuth and elevation) using non-linear least squares optimization on TDOAs.
    
    Args:
        mic_array: MicrophoneArray object containing microphones with recordings
        sound_speed: Speed of sound in m/s (default: 343.0)
        
    Returns:
        dict: Dictionary containing 'azimuth_deg' and 'elevation_deg'
    """
    # Step 1: Collect TDOA measurements from all microphone pairs
    tdoa_data = []
    
    for i in range(len(mic_array.microphones)):
        for j in range(i+1, len(mic_array.microphones)):
            mic_1 = mic_array.microphones[i]
            mic_2 = mic_array.microphones[j]
            
            # Compute TDOA and cross-correlation
            cc, lags = cross_correlation(mic_1.recording, mic_2.recording)
            tdoa = compute_tdoa(mic_1.recording, mic_2.recording, mic_1.sample_rate)
            
            # Simple quality metric: peak value of correlation
            peak_value = np.max(cc)
            
            tdoa_data.append({
                'mic_i': i,
                'mic_j': j,
                'tdoa': tdoa,
                'quality': peak_value,
                'mic_1_pos': mic_1.position,
                'mic_2_pos': mic_2.position
            })
    
    # Step 2: Simple outlier rejection based on correlation quality
    qualities = [d['quality'] for d in tdoa_data]
    quality_threshold = np.median(qualities) * 0.5
    
    valid_tdoas = [d for d in tdoa_data if d['quality'] > quality_threshold]
    
    if len(valid_tdoas) < 2:
        valid_tdoas = tdoa_data
    
    # Step 3: Define objective function for 3D optimization
    def residuals_3d(params):
        """Compute residuals using PyTorch for vectorized operations."""
        device = get_optimal_device()
        
        azimuth = float(params[0])
        elevation = float(params[1])
        
        # 3D source direction vector
        source_dir = torch.tensor([
            np.cos(azimuth) * np.cos(elevation),
            np.sin(azimuth) * np.cos(elevation),
            np.sin(elevation)
        ], device=device, dtype=torch.float32)
        
        # Vectorize all TDOA calculations
        tdoa_measured = torch.tensor([data['tdoa'] for data in valid_tdoas], device=device, dtype=torch.float32)
        
        # Stack all microphone vectors
        mic_vectors = torch.stack([
            torch.tensor(data['mic_2_pos'] - data['mic_1_pos'], device=device, dtype=torch.float32) 
            for data in valid_tdoas
        ])
        
        # Vectorized dot product for all pairs at once
        tdoa_predicted = torch.matmul(mic_vectors, source_dir) / sound_speed
        
        # Vectorized weight calculation
        qualities_tensor = torch.tensor([data['quality'] for data in valid_tdoas], device=device, dtype=torch.float32)
        weights = torch.sqrt(qualities_tensor / torch.max(qualities_tensor))
        
        # Vectorized error calculation
        errors = weights * (tdoa_measured - tdoa_predicted)
        
        return errors.cpu().numpy()  # Convert back for least_squares
    
    # Step 4: Define 2D objective function for azimuth initialization
    def residuals_2d(params):
        """Compute residuals for 2D case (elevation = 0) to get good azimuth estimate."""
        errors = []
        
        azimuth = float(params[0])
        source_dir = np.array([np.cos(azimuth), np.sin(azimuth), 0.0])
        
        for data in valid_tdoas:
            tdoa_measured = data['tdoa']
            mic_vector = data['mic_2_pos'] - data['mic_1_pos']
            tdoa_predicted = np.dot(source_dir, mic_vector) / sound_speed
            
            # Weight by quality
            weight = np.sqrt(data['quality'] / max(qualities))
            error = weight * (tdoa_measured - tdoa_predicted)
            errors.append(error)
        
        return np.array(errors)
    
    # Step 5: Grid search for azimuth initialization
    # Step 5: Grid search for azimuth initialization (PyTorch vectorized)
    device = get_optimal_device()
    test_angles = torch.linspace(-np.pi, np.pi, 36, device=device, dtype=torch.float32)

    # Pre-compute all data as tensors for vectorized operations
    tdoa_measured = torch.tensor([data['tdoa'] for data in valid_tdoas], device=device, dtype=torch.float32)
    mic_vectors = torch.stack([
        torch.tensor(data['mic_2_pos'] - data['mic_1_pos'], device=device, dtype=torch.float32) 
        for data in valid_tdoas
    ])
    qualities_tensor = torch.tensor([data['quality'] for data in valid_tdoas], device=device, dtype=torch.float32)
    max_quality = torch.max(qualities_tensor)
    weights = torch.sqrt(qualities_tensor / max_quality)

    # Vectorized cost calculation for all angles at once
    def vectorized_grid_search(angles, tdoa_measured, mic_vectors, weights, sound_speed):
        """Compute costs for all angles simultaneously."""
        num_angles = len(angles)
        costs = torch.zeros(num_angles, device=angles.device, dtype=torch.float32)
        
        for i, angle in enumerate(angles):
            # 2D source direction vector for this angle
            source_dir = torch.tensor([torch.cos(angle), torch.sin(angle), 0.0], device=angles.device, dtype=torch.float32)
            
            # Vectorized TDOA prediction for all microphone pairs
            tdoa_predicted = torch.matmul(mic_vectors, source_dir) / sound_speed
            
            # Vectorized error calculation
            errors = weights * (tdoa_measured - tdoa_predicted)
            costs[i] = torch.sum(errors**2, dtype=torch.float32)
        
        return costs

    # Compute all costs at once
    costs = vectorized_grid_search(test_angles, tdoa_measured, mic_vectors, weights, sound_speed)

    # Find the best angle
    best_idx = torch.argmin(costs)
    best_azimuth = test_angles[best_idx].item()
    best_cost = costs[best_idx].item()
    
    # Step 6: Try multiple elevation starting points with best azimuth
    # Step 6: Try multiple elevation starting points with best azimuth
    elevation_candidates = [0, np.pi/6, -np.pi/6, np.pi/3, -np.pi/3]  # 0°, ±30°, ±60°
    best_initial_guess = None
    best_3d_cost = np.inf

    # Ensure azimuth is within bounds first
    best_azimuth = np.clip(best_azimuth, -np.pi + 1e-6, np.pi - 1e-6)

    for test_elevation in elevation_candidates:
        # Clamp elevation to valid range
        clamped_elevation = np.clip(test_elevation, -np.pi/2 + 1e-6, np.pi/2 - 1e-6)
        
        initial_guess = [best_azimuth, clamped_elevation]
        
        try:
            cost = np.sum(residuals_3d(initial_guess)**2)
            if cost < best_3d_cost:
                best_3d_cost = cost
                best_initial_guess = initial_guess
        except Exception as e:
            print(f"Warning: Failed to evaluate initial guess {initial_guess}: {e}")
            continue

    # Fallback if no valid initial guess found
    if best_initial_guess is None:
        print("Warning: No valid initial guess found, using fallback [0, 0]")
        best_initial_guess = [0.0, 0.0]

    # Final safety check - ensure initial guess is strictly within bounds
    best_initial_guess[0] = np.clip(best_initial_guess[0], -np.pi + 1e-6, np.pi - 1e-6)
    best_initial_guess[1] = np.clip(best_initial_guess[1], -np.pi/2 + 1e-6, np.pi/2 - 1e-6)

    # Debug output (remove after testing)
    # print(f"Final initial guess: {best_initial_guess}")
    # print(f"Azimuth in bounds: {-np.pi < best_initial_guess[0] < np.pi}")
    # print(f"Elevation in bounds: {-np.pi/2 < best_initial_guess[1] < np.pi/2}")

    # Step 7: Run 3D optimization
    result = least_squares(
        residuals_3d, 
        best_initial_guess,
        bounds=([-np.pi + 1e-6, -np.pi/2 + 1e-6], [np.pi - 1e-6, np.pi/2 - 1e-6]),
        method='trf'
    )
    # Convert to degrees
    azimuth_deg = np.degrees(result.x[0])
    elevation_deg = np.degrees(result.x[1])
    
    # Apply the same 180° flip correction for azimuth as in original code
    # azimuth_deg = azimuth_deg + 180
    # if azimuth_deg > 180:
    #     azimuth_deg -= 360
    
    return {
        'azimuth_deg': azimuth_deg,
        'elevation_deg': elevation_deg
    }

def convert_to_b_format(recordings, sample_rate=44100):
    """
    Convert recordings to B-format (W, X, Y, Z).
    
    Args:
        recordings: Numpy array of shape (num_mics, num_samples)
    Returns:
        b_format: Numpy array of shape (4, num_samples) containing B-format signals
    """
    b_transform = np.asarray([[1,1,1,1],
                [1, 1, -1, -1],
                [1, -1, 1, -1],
                [1, -1, -1, 1]])
    recordings_b = b_transform @ recordings
    return recordings_b, sample_rate

def estimate_doa_using_aim(mic_array, sound_speed=343.0, angle_offset_deg=0.0):
    """
    Estimate 3D DOA using the AIM method with angle offset.
    
    Args:
        mic_array: MicrophoneArray object containing microphones with recordings
        sound_speed: Speed of sound in m/s (default: 343.0)
        angle_offset_deg: Angle offset in degrees (default: 0.0)
        
    Returns:
        dict: Dictionary containing 'azimuth_deg' and 'elevation_deg'
    """
    recordings = []
    for i in mic_array.microphones:
        if i.recording is None:
            raise ValueError("All microphones must have recordings to estimate DOA.")
        recordings.append(i.recording)
    recordings = np.array(recordings)
    recordings_b, sample_rate = convert_to_b_format(recordings, mic_array.microphones[0].sample_rate)

    # make b-format spectrogram
    specs = []
    for num in np.arange(4):
        freqs, inds, spec = scipysig.stft(recordings_b[num,:], fs=sample_rate, nperseg=1024)
        nf_full = len(freqs)
        freqs = freqs[0:160]
        specs.append(spec[0:160, :].T)

    # directly get the three components
    w = specs[0]
    x = specs[1]
    y = specs[2]

    # azimuth values for all pixels
    azimuth = np.arctan2(np.real(w.conj() * y), np.real(w.conj() * x))

    # weight the azimuth values by the intensity
    weights = np.abs(w)**2
    
    # need to set these parameters for histogram
    duration = len(recordings_b[0])/sample_rate
    time_step = 0.05
    num_time = int(duration * 1/time_step)
    num_azim = 60

    azimuth_flat = azimuth.flatten()
    weights_flat = weights.flatten()

    # Convert azimuth from radians to degrees and wrap to [-180, 180]
    azimuth_deg = np.degrees(azimuth_flat)
    azimuth_deg = (azimuth_deg + 180) % 360 - 180

    # Calculate weighted circular mean
    # Convert angles to radians for circular averaging
    azimuth_rad = np.radians(azimuth_deg)
    
    # Convert to unit vectors on the unit circle
    cos_angles = np.cos(azimuth_rad)
    sin_angles = np.sin(azimuth_rad)
    
    # Calculate weighted mean of the unit vectors
    weighted_cos_sum = np.sum(weights_flat * cos_angles)
    weighted_sin_sum = np.sum(weights_flat * sin_angles)
    total_weight = np.sum(weights_flat)
    
    # Avoid division by zero
    if total_weight == 0:
        raise ValueError("Total weight is zero - unable to compute circular mean")
    
    mean_cos = weighted_cos_sum / total_weight
    mean_sin = weighted_sin_sum / total_weight
    
    # Calculate the circular mean angle
    circular_mean_rad = np.arctan2(mean_sin, mean_cos)
    circular_mean_deg = np.degrees(circular_mean_rad)
    
    # Apply angle offset
    final_azimuth = circular_mean_deg + angle_offset_deg
    
    # Wrap the final result to [-180, 180] range
    final_azimuth = (final_azimuth + 180) % 360 - 180
    
    # Return dictionary with azimuth (elevation would need additional processing)
    return {
        'azimuth_deg': final_azimuth,
        'elevation_deg': 0.0  # Placeholder - original code doesn't compute elevation
    }