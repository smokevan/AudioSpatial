import numpy as np
from scipy import signal as scipysig
from scipy import stats
import time

from source_objects import soundSource
from source_objects import noiseSource

sound_speed = 343.0  # Speed of sound in air at 20 degrees Celsius in m/s

class micCapsule():
    """
    A class to represent a microphone capsule in a simulation.
    """

    def __init__(self, position, sample_rate=44100, noise = False, noiseAmplitude = 0.01):
        """
        Initialize microphone capsule with position, recording length, and sample rate.

        :param position: Position of the microphone capsule in 3D space (x, y, z) as a list or array
        :param recording_length: Length of the recording in seconds (default is 1.0)
        :param sample_rate: Sample rate in Hz (default is 44100)
        """
        self.position = np.array(position)
        self.sample_rate = sample_rate
        self.recording = None
        self.noise = noise
        self.noiseAmplitude = noiseAmplitude
    
    def __repr__(self):
        return f"MicCapsule object at position {self.position}, recording length: {self.recording_length} s, sample rate: {self.sample_rate} Hz"
    
class omnidirectionalMic(micCapsule):
    """
    A class to represent an omnidirectional microphone capsule.
    """

    def __init__(self, position, sample_rate=44100, noise = False, noiseAmplitude = 0.01):
        """
        Initialize an omnidirectional microphone capsule.

        :param position: Position of the microphone capsule in 3D space (x, y, z) as a list or array
        :param recording_length: Length of the recording in seconds (default is 1.0)
        :param sample_rate: Sample rate in Hz (default is 44100)
        """
        super().__init__(position, sample_rate, noise=noise, noiseAmplitude=noiseAmplitude)
    
    def __repr__(self):
        return f"OmnidirectionalMic object at position {self.position}, recording length: {self.recording_length} s, sample rate: {self.sample_rate} Hz"
    
    def record(self, sources, signals, recording_length = 1.0):
        """
        Record signals from multiple sound sources into the microphone capsule.
    
        :param sources: List of sound source objects
        :param signals: List of numpy arrays containing the signals to be recorded

        """
        num_samples = int(recording_length * self.sample_rate)

        self.recording = np.zeros(num_samples)  # Initialize recording buffer
        
        for source, signal in zip(sources, signals):
            # Check if this is a noise source
            if isinstance(source, noiseSource):
                travel_distance = np.linalg.norm(source.position - self.position)
                travel_time = travel_distance / sound_speed
                delay_samples = int(travel_time * self.sample_rate)

                attenuation_factor = 1.0 / (travel_distance*2) # Simple attenuation model
                # For noise sources, add directly without distance effects
                signal_length = min(len(signal), num_samples)
                self.recording[:signal_length] += signal[:signal_length] * attenuation_factor
            else:
                # For regular sources, apply distance-based delay and attenuation
                travel_distance = np.linalg.norm(source.position - self.position)
                travel_time = travel_distance / sound_speed
                delay_samples = int(travel_time * self.sample_rate)

                attenuation_factor = 1.0 / travel_distance # Simple attenuation model

                if delay_samples < num_samples:
                    received_signal = signal[:num_samples - delay_samples] * attenuation_factor  # Apply attenuation
                    self.recording[delay_samples:delay_samples + len(received_signal)] += received_signal
                    if self.noise:
                        noise_signal = np.random.normal(0, self.noiseAmplitude, len(self.recording))
                        self.recording += noise_signal
                else:
                    print(f"Signal from source at {source.position} is too far away to be recorded within the recording length.")
        
        return self.recording

class directionalMic(micCapsule):
    """
    A class to represent a directional microphone capsule.
    """

    def __init__(self, position, direction, sample_rate=44100, noise = False, noiseAmplitude = 0.01):
        """
        Initialize a directional microphone capsule.

        :param position: Position of the microphone capsule in 3D space (x, y, z) as a list or array
        :param direction: Direction of the microphone's sensitivity in 3D space (x, y, z) as a list or array
        :param recording_length: Length of the recording in seconds (default is 1.0)
        :param sample_rate: Sample rate in Hz (default is 44100)
        """
        super().__init__(position, sample_rate, noise=noise, noiseAmplitude=noiseAmplitude)
        self.direction = np.array(direction) / np.linalg.norm(direction)  # Normalize direction vector
    
    def __repr__(self):
        return (f"DirectionalMic object at position {self.position}, "
                f"direction: {self.direction}, recording length: {self.recording_length} s, sample rate: {self.sample_rate} Hz")
    def record(self, sources, signals, recording_length = 1.0):
        """
        Record signals from multiple sound sources into the microphone capsule.
    
        :param sources: List of sound source objects
        :param signals: List of numpy arrays containing the signals to be recorded

        """
        num_samples = int(recording_length * self.sample_rate)

        self.recording = np.zeros(num_samples)  # Initialize recording buffer
        
        for source, signal in zip(sources, signals):
            # Check if this is a noise source
            if isinstance(source, noiseSource):
                travel_distance = np.linalg.norm(source.position - self.position)
                travel_time = travel_distance / sound_speed
                delay_samples = int(travel_time * self.sample_rate)

                attenuation_factor = 1.0 / (travel_distance*2) # Simple attenuation model
                # For noise sources, add directly without distance effects
                signal_length = min(len(signal), num_samples)
                self.recording[:signal_length] += signal[:signal_length] * attenuation_factor
            else:
                # For regular sources, apply distance-based delay and attenuation
                travel_distance = np.linalg.norm(source.position - self.position)

                travel_direction = source.position/np.linalg.norm(source.position)

                travel_time = travel_distance / sound_speed
                delay_samples = int(travel_time * self.sample_rate)

                attenuation_factor = 1.0 / travel_distance # Simple attenuation model

                directionality_factor = (1 + np.dot(travel_direction, self.direction))/2

                if delay_samples < num_samples:
                    received_signal = signal[:num_samples - delay_samples] * attenuation_factor * directionality_factor # Apply attenuation and direction
                    self.recording[delay_samples:delay_samples + len(received_signal)] += received_signal
                    if self.noise:
                        noise_signal = np.random.normal(0, self.noiseAmplitude, len(self.recording))
                        self.recording += noise_signal
                else:
                    print(f"Signal from source at {source.position} is too far away to be recorded within the recording length.")
        
        return self.recording

class microphoneArray():
    """
    A class to represent a microphone array in a simulation.
    """

    def __init__(self, sample_rate=44100, spacing=0.035,num_mics = 4, geometry = "planar", mic_type = "omni", noise = False, noiseAmplitude = 0.01):
        """
        Initialize a microphone array with a list of microphone capsules.

        :param microphones: List of microphone capsule objects (e.g., omnidirectionalMic, directionalMic)
        """
        self.sample_rate = sample_rate
        self.spacing = spacing
        self.num_mics = num_mics
        self.geometry = geometry
        self.mic_type = mic_type
        self.noise = noise
        self.noiseAmplitude = noiseAmplitude

        if self.geometry == "planar" and self.mic_type == "omni":
            angle_diff = 360/self.num_mics
            circle_radius = self.spacing/(2 * np.sin(np.pi/self.num_mics))

            mic_list = []

            for i in range(self.num_mics):
                position = [circle_radius * np.cos(i * angle_diff), circle_radius * np.sin(i * angle_diff), 0]
                mic = omnidirectionalMic(position, sample_rate=self.sample_rate, noise=self.noise, noiseAmplitude=self.noiseAmplitude)
                mic_list.append(mic)
        elif self.geometry == "tetra" and self.mic_type == "omni":
            sphere_radius = 0.612 * self.spacing
            mic_list = [
                omnidirectionalMic([sphere_radius, sphere_radius, sphere_radius], sample_rate=self.sample_rate, noise=self.noise, noiseAmplitude=self.noiseAmplitude),
                omnidirectionalMic([sphere_radius, -sphere_radius, -sphere_radius], sample_rate=self.sample_rate, noise=self.noise, noiseAmplitude=self.noiseAmplitude),
                omnidirectionalMic([-sphere_radius, sphere_radius, -sphere_radius], sample_rate=self.sample_rate, noise=self.noise, noiseAmplitude=self.noiseAmplitude),
                omnidirectionalMic([-sphere_radius, -sphere_radius, sphere_radius], sample_rate=self.sample_rate, noise=self.noise, noiseAmplitude=self.noiseAmplitude)
            ]
        elif self.geometry == "tetra" and self.mic_type == "directional":
            sphere_radius = 0.612 * self.spacing
            mic_list = [
                directionalMic([sphere_radius, sphere_radius, sphere_radius], [1, 1, 1], sample_rate=self.sample_rate, noise=self.noise, noiseAmplitude=self.noiseAmplitude),
                directionalMic([sphere_radius, -sphere_radius, -sphere_radius], [1, -1, -1], sample_rate=self.sample_rate, noise=self.noise, noiseAmplitude=self.noiseAmplitude),
                directionalMic([-sphere_radius, sphere_radius, -sphere_radius], [-1, 1, -1], sample_rate=self.sample_rate, noise=self.noise, noiseAmplitude=self.noiseAmplitude),
                directionalMic([-sphere_radius, -sphere_radius, sphere_radius], [-1, -1, 1], sample_rate=self.sample_rate, noise=self.noise, noiseAmplitude=self.noiseAmplitude)
            ]
        self.microphones = mic_list

    def __repr__(self):
        return f"MicrophoneArray with {len(self.microphones)} microphones"
    
    def record(self, source, signal, recording_length = 1.0):
        """
        Record a signal into all microphones in the array.
        """
        recordings = []
        for mic in self.microphones:
            recording = mic.record(source, signal, recording_length = recording_length)
            recordings.append(recording)

        recording = np.array(recordings)
        return recording

    def updateSpacing(self, spacing):
        """
        Update the spacing between microphones in the array.
        """
        if self.geometry == "planar" and self.mic_type == "omni":
            angle_diff = 360/self.num_mics
            circle_radius = spacing/(2 * np.sin(np.pi/self.num_mics))

            for i, mic in enumerate(self.microphones):
                mic.position = np.array([circle_radius * np.cos(i * angle_diff), circle_radius * np.sin(i * angle_diff), 0])
        elif self.geometry == "tetra" and self.mic_type == "omni":
            sphere_radius = 0.612 * spacing
            self.microphones[0].position = np.array([sphere_radius, sphere_radius, sphere_radius])
            self.microphones[1].position = np.array([sphere_radius, -sphere_radius, -sphere_radius])
            self.microphones[2].position = np.array([-sphere_radius, sphere_radius, -sphere_radius])
            self.microphones[3].position = np.array([-sphere_radius, -sphere_radius, sphere_radius])
        elif self.geometry == "tetra" and self.mic_type == "directional":
            sphere_radius = 0.612 * spacing
            self.microphones[0].position = np.array([sphere_radius, sphere_radius, sphere_radius])
            self.microphones[1].position = np.array([sphere_radius, -sphere_radius, -sphere_radius])
            self.microphones[2].position = np.array([-sphere_radius, sphere_radius, -sphere_radius])
            self.microphones[3].position = np.array([-sphere_radius, -sphere_radius, sphere_radius])

