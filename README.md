# EINSfit
EINSfit is a class to fit Elastic Incoherent Neutron Scattering (EINS) data with the Gaussian Approximation (GA) model and additional models which can use a larger momentum transfer range Q. Developed for data obtained with the LAMP software of the back-scattering spectrometer IN13 (Institut Laue-Langevin / CRG, www.ill.eu) , but can also be used for any data in numpy format.

It should be used via an interactive python console (ipython) or jupyter and needs Python>=3.6 with following extra packages:

  - numpy
  - matplotlib (+cycler)
  - lmfit
