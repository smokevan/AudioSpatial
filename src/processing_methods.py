import numpy as np
from scipy import signal as scipysig
from scipy import stats
import math
import matplotlib.pyplot as plt

import numpy as np
from scipy.optimize import least_squares
from numpy.fft import fft, ifft
from scipy.signal import correlate


def cross_correlation(sig1, sig2):
    """Compute cross-correlation between two signals using PHAT weighting.
    
    Args:
        sig1: First signal array
        sig2: Second signal array
        
    Returns:
        cc: Cross-correlation array
        lags: Corresponding lag values
    """
    # n = sig1.shape[0] + sig2.shape[0]
    # SIG1 = np.fft.rfft(sig1, n=n)
    # SIG2 = np.fft.rfft(sig2, n=n)
    
    # R = SIG1 * np.conj(SIG2)
    # R /= np.abs(R) + 1e-15  # PHAT weighting: keep phase, normalize magnitude

    # cc = np.fft.irfft(R, n=n)
    # max_shift = int(n / 2)
    # cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))  # wrap negative lags

    # lags = np.arange(-max_shift, max_shift + 1)

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
        """Compute residuals between measured and predicted TDOAs for 3D case."""
        errors = []
        
        azimuth = float(params[0])
        elevation = float(params[1])
        
        # 3D source direction vector
        source_dir = np.array([
            np.cos(azimuth) * np.cos(elevation),
            np.sin(azimuth) * np.cos(elevation),
            np.sin(elevation)
        ])
        
        for data in valid_tdoas:
            tdoa_measured = data['tdoa']
            mic_vector = data['mic_2_pos'] - data['mic_1_pos']
            tdoa_predicted = np.dot(source_dir, mic_vector) / sound_speed
            
            # Weight by quality
            weight = np.sqrt(data['quality'] / max(qualities))
            error = weight * (tdoa_measured - tdoa_predicted)
            errors.append(error)
        
        return np.array(errors)
    
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
    test_angles = np.linspace(-np.pi, np.pi, 36)
    best_azimuth = None
    best_cost = np.inf
    
    for test_angle in test_angles:
        cost = np.sum(residuals_2d([test_angle])**2)
        if cost < best_cost:
            best_cost = cost
            best_azimuth = test_angle
    
    # Step 6: Try multiple elevation starting points with best azimuth
    elevation_candidates = [0, np.pi/6, -np.pi/6, np.pi/3, -np.pi/3]  # 0°, ±30°, ±60°
    best_initial_guess = None
    best_3d_cost = np.inf
    
    for test_elevation in elevation_candidates:
        initial_guess = [best_azimuth, test_elevation]
        cost = np.sum(residuals_3d(initial_guess)**2)
        if cost < best_3d_cost:
            best_3d_cost = cost
            best_initial_guess = initial_guess
    
    # Step 7: Run 3D optimization
    result = least_squares(
        residuals_3d, 
        best_initial_guess,
        bounds=([-np.pi, -np.pi/2], [np.pi, np.pi/2]),
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
    # Convert recordings to B-format

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

    # Create histogram bins
    num_azim = 60
    bins = np.linspace(-180, 180, num_azim + 1)

    # Weighted histogram
    hist, bin_edges = np.histogram(azimuth_deg, bins=bins, weights=weights_flat)

    # Find the bin with the maximum weighted count
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.figure(figsize=(10, 5))
    plt.bar(bin_centers, hist, width=bin_edges[1] - bin_edges[0], align='center', alpha=0.7, color='royalblue')
    plt.xlabel('Azimuth (degrees)')
    plt.ylabel('Weighted Count')
    plt.title('Azimuthal Histogram (AIM Method)')
    plt.grid(True, alpha=0.3)
    plt.xlim([-180, 180])
    plt.show()