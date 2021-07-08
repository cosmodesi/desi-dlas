CNN Structure
The model we used is a standard CNN architecture, casting DLAs as a 1D image problem. There are three convolutional layers, three pooling layers and one fully connected layers. We have reset two parameters variable. One is n, it stands for how many pixels one window contains, it could be 400 or 600. The other one is m, it means the dimensions of the input data, the value of it is 1 or 4 (median smoothing for low S/N spectra). The three sub-fully connected layers are correspond to three labels: classification, offset(DLA location) and column density. 

Data Input
The input to the neural network consists of the raw flux values in the sightlines. The uncertainty values (i.e. error array) are entirely ignored by the algorithm.
We do not actually input all sightline pixels into our network because this would make identifying multiple DLAs along a given sightline too challenging4. Therefore, the input to the network is a n pixel region of the sightline in a sliding window. 
