## CNN Structure
The model we used is a standard CNN architecture, casting DLAs as a 1D image problem. There are three convolutional layers, three pooling layers and one fully connected layers. We have reset two parameters variable. One is n, it stands for how many pixels one window contains, it could be 400 or 600. The other one is m, it means the dimensions of the input data, the value of it is 1 or 4 (median smoothing for low S/N spectra). The three sub-fully connected layers are correspond to three labels: classification, offset(DLA location) and column density. 
![image](https://github.com/cosmodesi/desi-dlas/blob/training/desidlas/training/figures/CNN.png)

## Data Input
The input to the neural network consists of the raw flux values in the sightlines. The uncertainty values (error array) are ignored by the algorithm.
We do not actually input all sightline pixels into our network because this would make identifying multiple DLAs along a given sightline too challenging. Therefore, the input to the network is a n pixel region of the sightline in a sliding window. 

## Loss Function
In this model, three outputs are produced: classification, localization, and column density estimation. So there are three individual loss functions.

The loss function for DLA classification (![](http://latex.codecogs.com/svg.latex?\mathcal{L}_{c})) in a sample is the standard cross-entropy loss function

where ![](http://latex.codecogs.com/svg.latex?y_{c}) is the ground truth classification label ( 1 if DLA; 0 otherwise) for the sample; and ![](http://latex.codecogs.com/svg.latex?\hat{y_{c}}) is the model’s predicted classification ![](http://latex.codecogs.com/svg.latex?\hat{y_{c}}) is in the range (0,1) where ![](http://latex.codecogs.com/svg.latex?\hat{y_{c}}) > 0.5 indicates a positive DLA classification and ![](http://latex.codecogs.com/svg.latex?\hat{y_{c}}) < 0.5 negative classification).

## Parameter set
The parameters deciding the size of each layer is shown in parameterset.py 

## Output
The network produces three output labels for every n pixels window in the spectrum.If a DLA is predicted the label takes on a 1/true value, else 0/false. Second the algorithm estimates the DLA line-center by producing a value in the range -60~+60 that indicates how many pixels the window center is from the center of the DLA. Third, the algorithm produces a column density estimate that is only valid if a DLA actually exists in the current window.
