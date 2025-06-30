import numpy as np
from scipy.io import wavfile

class soundSource():
    """
    A class to represent a sound source in a simulation.
    """

    def __init__(self, distance, angle, elevation=0.0, amplitude = 0):
        """
        initialize sound source with distance, angle, amplitude, freq, and elevation. 
        (effectively a point source described by spherical coordinates)
        
        :param distance: Distance of the sound source to the array center (coordinate center)
        :param angle: Planar angle of the sound source in degrees
        :param elevation: Elevation angle of the sound source in degrees (default is 0.0)
        """
        self.distance = distance
        self.angle = angle
        self.elevation = elevation
        self.position = np.array([
            distance * np.cos(np.radians(angle)) * np.cos(np.radians(elevation)),
            distance * np.sin(np.radians(angle)) * np.cos(np.radians(elevation)),
            distance * np.sin(np.radians(elevation))
        ])
        self.amplitude = amplitude

    def __repr__(self):
        return f"SoundSource object {self} at position {self.position}"
    
    def updateAngle(self, angle):
        self.angle = angle
        self.position = np.array([
            self.distance * np.cos(np.radians(angle)),
            self.distance * np.sin(np.radians(angle)),
            self.distance * np.sin(np.radians(self.elevation))
        ])
    def updateAmp(self, amp):
        self.amplitude = amp
        
    
class pureTone(soundSource):
    
    """
    """
    def __init__(self, distance, angle, frequency, amplitude=1.0, elevation=0.0):
        """
        a pure tone sound source with parent SoundSource

        :param frequency: Frequency of the pure tone in Hz
        :param amplitude: Amplitude of the pure tone (default is 1.0)
        
        """
        super().__init__(distance, angle, elevation, amplitude=amplitude)
        self.frequency = frequency

    def __repr__(self):
        return (f"PureTone object {self} at position {self.position}, "
                f"frequency: {self.frequency} Hz, amplitude: {self.amplitude}")
    
    def generate_signal(self, duration, sample_rate=44100, windowing=None):
        """
        Generate a pure tone signal.

        :param duration: Duration of the signal in seconds
        :param sample_rate: Sample rate in Hz (default is 44100)
        :return: Numpy array containing the generated signal
        """
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        signal = self.amplitude * np.sin(2 * np.pi * self.frequency * t)

        if windowing is not None:
            if windowing == "hann":
                signal = signal * np.hanning(len(signal))
        return signal

class noiseSource(soundSource):

    def __init__(self, distance = 0, angle = 0, elevation = 0, type = "gauss", amplitude=1.0):
        """
        a pure tone sound source with parent SoundSource

        :param frequency: Frequency of the pure tone in Hz
        :param amplitude: Amplitude of the pure tone (default is 1.0)
        
        """
        super().__init__(distance = distance, angle = angle, elevation = elevation, amplitude=amplitude)
        self.type = type

    def __repr__(self):
        return (f"Noise {type} object {self} at position {self.position}, "
                f"frequency: {self.frequency} Hz, amplitude: {self.amplitude}")
    
    def generate_signal(self, duration, sample_rate=44100):
        """
        Generate a constant noise signal.

        :param duration: Duration of the signal in seconds
        :param sample_rate: Sample rate in Hz (default is 44100)
        :return: Numpy array containing the generated signal
        """
        if self.type == "gauss":
            num_samples = int(sample_rate * duration)
            # Generate Gaussian white noise
            noise = np.random.normal(0, self.amplitude, num_samples)
            return noise
        
class sweepSource(soundSource):
    
    def __init__(self, distance = 0, angle = 0, freq_min = 0, freq_max = 100, amplitude = 1, elevation = 0):
        super().__init__(self, distance, angle = angle, elevation = elevation, amplitude=amplitude)
        self.freq_min = freq_min
        self.freq_max = freq_max
        
class fileSource(soundSource):
    
    def __init__(self, distance = 0, angle = 0, elevation = 0, filename = '', amplitude = 1.0):
        super().__init__(distance = distance, angle=angle, elevation=elevation, amplitude=amplitude)
        self.filename = filename
    
    def generate_signal(self, sample_rate=44100, windowing = None):
        file_rate, data = wavfile.read(self.filename)
        signal_array = np.array(data, dtype=float)
        signal_array = (signal_array / np.max(np.abs(signal_array))) * self.amplitude
        if windowing is not None:
            if windowing == "hann":
                signal_array = signal_array * np.hanning(len(signal_array))
        return signal_array
        