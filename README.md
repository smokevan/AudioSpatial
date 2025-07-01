# AudioSpatial
Simulation toolkit for spatial sound using microphone arrays geared towards bioacoustics.

## Modules

The toolkit includes the following modules:

### Microphone Objects

Sound recievers, mainly initialized as microphone arrays. Planar and tetrahedral arrangements for geometry options, omnidirectional and cardioid microphones for reciever options. 

### Source Objects

Sound emitting objects, including noise sources, frequency sweeps, file sources, pure tones, etc.

### Processing Methods

These are methods used for the time-difference-of-arrival (TDOA) estimation, direction of arrival (DoA) estimation, ambisonics format conversion for acoustic vector sensing, and soon to include MUSIC and other methodology for DoA estimation.

### Simulation Methods

Prepackged functions for running particular simulations and plotting outputs such as heatmaps using matplotlib.