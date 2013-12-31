This folder contains samples of GIS data from the State Geographic Information Database (SGID) of the US State of Utah.
It also contains a sample of Turbulence data from the JHU Turbulence Database Cluster.

UtahTexSample1m.raw is a 2048x2048 3x8bit RGB image, extracted directly from the 1m color orthophotography data set.
UtahGeoSample1m.raw is a 2048x2048 16bit signed integer image, extracted from the 5m DEM data set, upsampled to 1m horizontal resolution, shifted to be zero-centered, and quantized to 1m vertical resolution.
Iso_256_256_256_t4000_VelocityX.raw is a 256x256x256 32bit float volume containing the x component of the velocity in a section of the first time step of the isotropic turbulence data set.
Iso_128_128_128_t4000_VelocityX/Y/Z.raw contain a 128x128x128 section of the x, y, and z components of the velocity.
All files are stored as raw data (no header).

The original data is available at:
http://gis.utah.gov/download
http://turbulence.pha.jhu.edu
