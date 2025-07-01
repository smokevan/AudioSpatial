import numpy as np
from scipy import signal as scipysig
from scipy import stats
from scipy.io.wavfile import write
import time
from IPython.display import Audio
import librosa
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from source_objects import pureTone
from source_objects import noiseSource
from source_objects import fileSource
from source_objects import sweepSource

from microphone_objects import omnidirectionalMic
from microphone_objects import microphoneArray

from processing_methods import estimate_doa_from_recordings
from processing_methods import estimate_doa_using_aim


### ======HELPER FUNCTIONS FOR TESTS ###

def plot_recordings(recording, sample_rate=441000, title="Microphone Array Recordings"):
    """
    Plot all four audio recordings from the microphone array
    
    Parameters:
    recording: list of 4 numpy arrays (audio recordings)
    sample_rate: sample rate in Hz
    title: plot title
    """
    
    # Create time axis based on recording length
    recording_length = len(recording[0])  # All recordings should be same length
    time_axis = np.linspace(0, recording_length / sample_rate, recording_length)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Colors for each microphone
    colors = ['blue', 'red', 'green', 'orange']
    labels = ['Mic 1', 'Mic 2', 'Mic 3', 'Mic 4']
    
    # Plot each recording
    for i in range(len(recording)):
        plt.plot(time_axis, recording[i], color=colors[i], label=labels[i], linewidth=1, alpha=0.8)
    
    # Set plot properties
    plt.ylim(-max(recording[1]) - 0.1, max(recording[1]) + 0.1)
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Amplitude', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Show the plot
    plt.show()

def run_multi_snr_ratios():

    global sample_rate

    a_signal = 10


    snr_db_values = np.linspace(0, 40, 25)
    noise_amps = a_signal * 10 ** (-snr_db_values / 20)

    mic_1 = omnidirectionalMic(position=[0.1, -0.1, 0.0], recording_length=5, sample_rate=sample_rate)
    mic_2 = omnidirectionalMic(position=[0.1, 0.1, 0.0], recording_length=5, sample_rate=sample_rate)
    mic_3 = omnidirectionalMic(position=[-0.1, 0.1, 0.0], recording_length=5, sample_rate=sample_rate)
    mic_4 = omnidirectionalMic(position=[-0.1, -0.1, 0.0], recording_length=5, sample_rate=sample_rate)

    mic_array = microphoneArray([mic_1, mic_2, mic_3, mic_4])

    source1 = pureTone(distance=100, angle=-120, frequency=440, amplitude=10.0)
    source2 = noiseSource(distance=100, angle=-120, type="gauss", amplitude = 0.25)
    results = []

    for snr, amp in zip(snr_db_values, noise_amps):
        source2.updateAmp(amp)
        sources = [source1, source2]
        signals = [source1.generate_signal(duration=1, sample_rate=sample_rate, windowing="hann"), source2.generate_signal(5, sample_rate=sample_rate)]
        
        recordings = mic_array.record(sources, signals)
        angle = estimate_doa_from_recordings(mic_array)

        results.append([snr, angle])
        print(snr, angle)

    print(results)

def plot_3d_setup(mic_array, sources, title="3D Audio Setup Visualization"):
    """
    Plot 3D positions of microphones and sound sources
    
    Parameters:
    mic_array: microphoneArray object containing microphones
    sources: list of source objects (pureTone, noiseSource, fileSource)
    title: plot title
    """
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract microphone positions
    mic_positions = np.array([mic.position for mic in mic_array.microphones])
    mic_x = mic_positions[:, 0]
    mic_y = mic_positions[:, 1] 
    mic_z = mic_positions[:, 2]
    
    # Plot microphones
    ax.scatter(mic_x, mic_y, mic_z, c='blue', s=100, marker='o', 
               label=f'Microphones ({len(mic_array.microphones)})', alpha=0.8)
    
    # Label each microphone
    for i, (x, y, z) in enumerate(mic_positions):
        ax.text(x, y, z + 0.01, f'Mic {i+1}', fontsize=10, ha='center')
    
    # Extract and plot source positions
    source_colors = ['red', 'green', 'orange', 'purple', 'brown']
    source_markers = ['s', '^', 'D', 'v', 'p']
    
    for i, source in enumerate(sources):
        # Convert polar coordinates (distance, angle, elevation) to Cartesian
        distance = source.distance
        angle_rad = np.radians(source.angle)
        elevation_rad = np.radians(source.elevation)
        
        # Convert to Cartesian coordinates
        x = distance * np.cos(elevation_rad) * np.cos(angle_rad)
        y = distance * np.cos(elevation_rad) * np.sin(angle_rad)
        z = distance * np.sin(elevation_rad)
        
        # Determine source type for label
        source_type = type(source).__name__
        if hasattr(source, 'frequency'):
            label = f'{source_type} ({source.frequency}Hz)'
        elif hasattr(source, 'type'):
            label = f'{source_type} ({source.type})'
        elif hasattr(source, 'filename'):
            label = f'{source_type} ({source.filename})'
        else:
            label = f'{source_type} {i+1}'
        
        # Plot source
        color = source_colors[i % len(source_colors)]
        marker = source_markers[i % len(source_markers)]
        ax.scatter(x, y, z, c=color, s=150, marker=marker, 
                   label=label, alpha=0.8, edgecolors='black', linewidth=1)
        
        # Add source label
        ax.text(x, y, z + 0.1, f'S{i+1}', fontsize=10, ha='center', 
                weight='bold', color=color)
        
        # Draw line from origin to source to show direction
        ax.plot([0, x], [0, y], [0, z], '--', color=color, alpha=0.5, linewidth=1)
    
    # Plot origin
    ax.scatter([0], [0], [0], c='black', s=50, marker='x', 
               label='Origin', alpha=0.8)
    
    # Connect microphones to show array geometry
    if len(mic_array.microphones) > 2:
        # Create lines between adjacent microphones to show array shape
        for i in range(len(mic_positions)):
            for j in range(i+1, len(mic_positions)):
                ax.plot([mic_positions[i, 0], mic_positions[j, 0]], 
                       [mic_positions[i, 1], mic_positions[j, 1]], 
                       [mic_positions[i, 2], mic_positions[j, 2]], 
                       'b-', alpha=0.3, linewidth=0.5)
    
    # Set labels and title
    ax.set_xlabel('X (meters)', fontweight='bold')
    ax.set_ylabel('Y (meters)', fontweight='bold') 
    ax.set_zlabel('Z (meters)', fontweight='bold')
    ax.set_title(title, fontweight='bold', fontsize=14)
    
    # Set equal aspect ratio
    max_range = max(np.max(np.abs(mic_positions)), 
                   max([s.distance for s in sources]) if sources else 1)
    ax.set_xlim([-max_range*1.1, max_range*1.1])
    ax.set_ylim([-max_range*1.1, max_range*1.1])
    ax.set_zlim([-max_range*0.2, max_range*0.2])
    
    # Add legend
    ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98))
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add angle annotation for sources
    info_text = "Source Information:\n"
    for i, source in enumerate(sources):
        info_text += f"S{i+1}: {source.angle}° azimuth, {source.elevation}° elevation, {source.distance}m\n"
    
    # Add text box with source info
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text2D(0.02, 0.02, info_text, transform=ax.transAxes, fontsize=9,
              verticalalignment='bottom', bbox=props)
    
    plt.tight_layout()
    plt.show()

def run_multi_sample_rates(): # Store results for all sample rates
    global sources
    all_results = {}

    print("Running DOA simulation with multiple sample rates...")
    total_start_time = time.time()

    for sr_idx, sample_rate in enumerate(sample_rates):
        print(f"\nTesting sample rate: {sample_rate} Hz")
        
        # Create microphone array for current sample rate
        mic_1 = omnidirectionalMic(position=[0.1, -0.1, 0.0], recording_length=2, sample_rate=sample_rate)
        mic_2 = omnidirectionalMic(position=[0.1, 0.1, 0.0], recording_length=2, sample_rate=sample_rate)
        mic_3 = omnidirectionalMic(position=[-0.1, 0.1, 0.0], recording_length=2, sample_rate=sample_rate)
        mic_4 = omnidirectionalMic(position=[-0.1, -0.1, 0.0], recording_length=2, sample_rate=sample_rate)
        
        mic_array = microphoneArray([mic_1, mic_2, mic_3, mic_4])
        
        # Store results for this sample rate
        true_angles = []
        estimated_angles = []
        
        start_time = time.time()
        for i, angle in enumerate(angle_range):
            sources[0].updateAngle(angle)
            signals = [sources[0].generate_signal(duration=0.25, sample_rate=sample_rate), sources[1].generate_signal(10, sample_rate=sample_rate)]
            mic_array.record(sources, signals)
            final_angle = estimate_doa_from_recordings(mic_array)
            
            true_angles.append(angle)
            estimated_angles.append(final_angle)
            
            print(f"True: {angle:6.1f}°, Estimated: {final_angle:6.1f}°")
        
        end_time = time.time()
        computation_time = end_time - start_time
        
        # Convert to numpy arrays
        true_angles = np.array(true_angles)
        estimated_angles = np.array(estimated_angles)
        
        # Handle angle wrapping for error calculation
        def angle_difference(angle1, angle2):
            """Calculate the smallest angle difference between two angles"""
            diff = angle1 - angle2
            # Wrap to [-180, 180]
            diff = ((diff + 180) % 360) - 180
            return diff
        
        # Calculate errors
        errors = np.array([angle_difference(est, true) for est, true in zip(estimated_angles, true_angles)])
        
        # Calculate statistics
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors**2))
        mean_error = np.mean(errors)
        median_error = np.median(errors)
        std_error = np.std(errors)
        max_error = np.max(np.abs(errors))
        min_error = np.min(np.abs(errors))
        
        # Store all results
        all_results[sample_rate] = {
            'true_angles': true_angles,
            'estimated_angles': estimated_angles,
            'errors': errors,
            'mae': mae,
            'rmse': rmse,
            'mean_error': mean_error,
            'median_error': median_error,
            'std_error': std_error,
            'max_error': max_error,
            'min_error': min_error,
            'computation_time': computation_time
        }

    total_end_time = time.time()
    total_computation_time = total_end_time - total_start_time

    # Calculate maximum error across all sample rates for dynamic y-limit
    max_error_overall = max([results['max_error'] for results in all_results.values()])

    # Create single polar plot with both sample rates
    fig = plt.figure(figsize=(12, 10))
    fig.suptitle('Direction of Arrival (DOA) Error Analysis - Sample Rate Comparison', fontsize=18, fontweight='bold')

    # Polar plot of Error vs True Angle
    ax = fig.add_subplot(111, projection='polar')

    # Plot data for each sample rate
    for sr_idx, sample_rate in enumerate(sample_rates):
        results = all_results[sample_rate]
        true_angles = results['true_angles']
        errors = results['errors']
        
        # Create interpolated data for smoother curve
        interp_angles = np.linspace(-180, 180, 500)  # Dense angle array for smooth curve
        f_interp = interp1d(true_angles, errors, kind='cubic', fill_value='extrapolate')
        interp_errors = f_interp(interp_angles)
        
        # Convert angles to radians for polar plot
        theta_rad = np.radians(true_angles)
        theta_interp_rad = np.radians(interp_angles)
        
        # Plot the thin interpolated curve with offset
        ax.plot(theta_interp_rad, np.abs(interp_errors) + 5, color=colors[sr_idx], 
                linewidth=1, alpha=0.7, label=f'{labels[sr_idx]} - Absolute Error')
        
        # Plot the actual data points on top with offset
        ax.plot(theta_rad, np.abs(errors) + 5, 'o', color=colors[sr_idx], 
                markersize=3, alpha=0.9)

    # Customize polar plot
    ax.set_theta_zero_location('N')  # 0° at top
    ax.set_theta_direction(-1)  # Clockwise
    ax.set_title('Error Magnitude vs. True Angle (Polar View)', fontweight='bold', fontsize=16, pad=30)
    ax.set_ylim(0, 15)  # Dynamic range: offset by 5 for visibility
    ax.grid(True, alpha=0.4)
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.0))

    # Add custom radial tick labels to show the actual error values
    max_tick = int(np.ceil(max_error_overall + 5))
    radial_ticks = np.arange(5, max_tick + 1, 2)  # Every 2 degrees, starting from 5 (0° actual)
    radial_labels = [f'{tick-5}°' for tick in radial_ticks]  # Convert back to actual error values
    ax.set_rticks(radial_ticks)
    ax.set_yticklabels(radial_labels)

    # Add radial labels
    ax.set_ylabel('Absolute Error (degrees)', labelpad=50, fontweight='bold')

    plt.tight_layout()

    # Print statistics for all sample rates
    print("\n" + "="*60)
    print("DOA ESTIMATION ERROR STATISTICS - SAMPLE RATE COMPARISON")
    print("="*60)

    for sr_idx, sample_rate in enumerate(sample_rates):
        results = all_results[sample_rate]
        print(f"\nSample Rate: {sample_rate} Hz ({labels[sr_idx]})")
        print("-" * 40)
        print(f"Mean Absolute Error (MAE):     {results['mae']:.2f}°")
        print(f"Root Mean Square Error (RMSE): {results['rmse']:.2f}°")
        print(f"Mean Error (Bias):             {results['mean_error']:.2f}°")
        print(f"Median Error:                  {results['median_error']:.2f}°")
        print(f"Standard Deviation:            {results['std_error']:.2f}°")
        print(f"Maximum Absolute Error:        {results['max_error']:.2f}°")
        print(f"Minimum Absolute Error:        {results['min_error']:.2f}°")
        print(f"Computation Time:              {results['computation_time']:.3f} seconds")
        print(f"Average Time per Estimate:     {results['computation_time']/len(angle_range):.3f} seconds")

    print(f"\nNumber of Test Points:         {len(angle_range)}")
    print(f"Total Computation Time:        {total_computation_time:.3f} seconds")
    print("="*60)

    plt.show()

def run_spacing_vs_frequency():
    """
    Run a simulation to analyze the effect of microphone spacing on DOA estimation accuracy
    at different frequencies. Returns percent error with proper angle wrapping.
    """
    
    global sample_rate

    # Define frequencies and spacings
    frequencies = np.arange(100, 20001, 500)  # Frequencies from 100 Hz to 20000 Hz
    spacings = np.arange(0.01, 0.21, 0.01)  # Spacings from 0.01 m to 0.20 m
    angle_range = np.arange(-180, 181, 10)  # Angle range in degrees
    results = np.zeros((len(frequencies), len(spacings)))
    source1 = sweepSource(50, 0, f0 = 0, f1 = 100, amplitude = 10.0)

    mic_array = microphoneArray(sample_rate, 0.070, geometry="tetra", mic_type="omni", noise=True, noiseAmplitude=0.0025)

    print("Running DOA simulation with varying microphone spacing and frequency...")

    def angle_difference(estimated, actual):
        """
        Calculate the smallest angle difference between two angles, handling wrapping.
        
        Args:
            estimated: Estimated angle in degrees
            actual: Actual angle in degrees
            
        Returns:
            Smallest angular difference in degrees (always positive)
        """
        # Calculate raw difference
        diff = estimated - actual
        
        # Wrap to [-180, 180] range
        diff = ((diff + 180) % 360) - 180
        
        # Return absolute value for error calculation
        return abs(diff)

    def calculate_percent_error(estimated, actual, max_error=180):
        """
        Calculate percent error for angle estimation.
        
        Args:
            estimated: Estimated angle in degrees
            actual: Actual angle in degrees
            max_error: Maximum possible error (180° for angles)
            
        Returns:
            Percent error
        """
        absolute_error = angle_difference(estimated, actual)
        percent_error = (absolute_error / max_error) * 100
        return percent_error

    for freq_idx, freq in enumerate(frequencies):
        for spacing_idx, spacing in enumerate(spacings):
            mic_array.updateSpacing(spacing)
            angle_errors = []
            for i, angle in enumerate(angle_range):
                source1.updateAngle(angle)
                source1.f0 = freq
                source1.f1 = freq + 500
                signals = [source1.generate_signal(windowing=None, duration=0.25, sample_rate=sample_rate)]
                recordings = mic_array.record([source1], signals, recording_length=4)
                # if freq_idx == 0 and spacing_idx == 0 and i == 0:
                #     plot_recordings(recordings)
                estimated_result = estimate_doa_from_recordings(mic_array)
                estimated_angle = estimated_result['azimuth_deg']
                
                # Calculate percent error with proper angle wrapping
                percent_error = calculate_percent_error(estimated_angle, angle)
                angle_errors.append(np.square(angle_difference(estimated_angle, angle)))

            # Store the mean percent error for this frequency and spacing
            results[freq_idx, spacing_idx] = np.sqrt(np.mean(angle_errors))
            
        # Progress indicator
        print(f"Completed frequency {freq} Hz ({freq_idx + 1}/{len(frequencies)})")

    print("DOA simulation completed.")

    return results, frequencies, spacings

def plot_2d_heatmap(results, y_ax, x_ax):
    """
    Plot a 2D heatmap of the results showing percent error.
    
    Parameters:
    results: 2D numpy array of results (now in percent error)
    frequencies: array of frequency values
    spacings: array of spacing values
    """
    
    plt.figure(figsize=(12, 8))
    
    # Create the heatmap
    im = plt.imshow(results, aspect='auto', cmap='viridis', origin='lower', 
                   extent=[x_ax[0], x_ax[-1], y_ax[0], y_ax[-1]])
    
    # Add colorbar
    cbar = plt.colorbar(im, label='Mean Percent Error (%)')
    
    # Set labels and title
    plt.xlabel('Microphone Spacing (meters)', fontweight='bold')
    plt.ylabel('Frequency (Hz)', fontweight='bold')
    plt.title('DOA Estimation RMS Angle Error vs Frequency and Spacing', fontweight='bold', fontsize=14)
    
    # Add grid
    plt.grid(True, alpha=0.3, color='white')
    
    # Format the plot
    plt.tight_layout()
    plt.show()

### ======TEST SETUP BELOW THIS LINE====== ###


# Test sample rates
sample_rates = [44100, 192000]  # Sample rates in Hz
sample_rate = 44100
colors = ['blue', 'red']  # Colors for different sample rates
labels = ['44.1 kHz', '192 kHz']  # Labels for legend

source1 = pureTone(distance=300, angle=-40, frequency=100, amplitude=10.0, elevation=0)

source3 = fileSource(distance=300, angle=-40, amplitude=10.0, filename='trimmed.wav', elevation=0)

sources = [source1]
signals = [source1.generate_signal(windowing="hann", duration=1, sample_rate=sample_rate)]


angle_range = np.arange(-180, 181, 1)  # Angle range in degrees

mic_array = microphoneArray(sample_rate, 0.070, geometry="tetra", mic_type="omni", noise=True, noiseAmplitude=0.001)

recordings = mic_array.record(sources, signals, recording_length = 4)

# Plot all four microphone recordings

# Call the function to plot your recordings

result, freq, space = run_spacing_vs_frequency()

plot_2d_heatmap(result, freq, space)    