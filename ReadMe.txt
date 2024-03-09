This code was written to generate hypothetical single photon organic quantum emitter materials in SMILES notation. The training data was scraped from the literature to the train the model. The model is based on a variational autoencoder where the latent space is sampled to generate new SMILES or species of materials. The stability and emission properties were predicted by using 3DS Material Studios Semi-Empirical Quantum Chemistry package (VAMP). The data was processed through the quantum chemistry package in a high-throughput manner use 3DS Pipeline Pilot Package. 

This code was written by Dr. Robert Tempke and Dr. Terence Musho - West Virginia University.

AGoRaS_QM.ipynb - Main Machine Learning VAE Model.

Data Folder - Contains Input and Output Data
PipelinePilot - Contains high-throughput Script for 3DS Pipeline Pilot
AuxCode - Contains Auxiliary Code for Processing Data.