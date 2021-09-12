Installing
------

Download
------

- You can download the code by visiting : https://github.com/cosmodesi/desi-dlas
- You can also use the command below to download the code

   .. code:: bash
   git clone https://github.com/cosmodesi/desi-dlas.git
   
Environment
------
- To make sure you have all the required modules, you can run 

   .. code:: bash
   pip install -r requirements.txt
   
- The file requirements.txt is in desidlas/requirements.txt
- Using GPU can reduce the time for running the CNN model obviously, so please make sure your tensorflow version is available for the GPU.
- If you have problem to install the tensorflow_gpu by using the requirements.txt, you can try to install it using conda:


   .. code:: bash
   conda create --name tf_gpu
   
   conda activate tf_gpu
   
   conda install tensorflow-gpu


Get The Model File
------
- The model files are too large to upload to github, you can find the model files for high S/N spectra here:

- https://drive.google.com/drive/folders/1DYOE_k9S_F0JmnAdFbTmHkVqyxFlc4t-?usp=sharing

- The model files are too large to upload to github, you can find the model files for low S/N spectra here : 

- https://drive.google.com/drive/folders/1s5km1NAg5j0Y-tWI1q58Y09hjj0Jjc8C?usp=sharing

