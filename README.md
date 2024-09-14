# Digit-Recognizer-in-Python
&emsp; A simple digit recognizer written using Python. <br/>

<br/>

**Details:** <br/>

<br/>

&emsp; The first attempt used the K-Nearest Neighbors (KNN) Algorithm with K = 100. <br/>
&emsp; 10000 MNIST digits pictures were used as train data (28x28 pixel images of just one colour channel). <br/>
&emsp; L1 and L2 metrics were used, L2 seems to work better. <br/>
&emsp; Used PyGame for the graphical interface, along with Numpy for fast vectorization. <br/>
&emsp; Tried using a MLP Classifier (Multi-Layer Perceptron Classifier) with 2 hidden layers of size 64, learning rate 0.001 and early stopping. <br/>
&emsp; The MLP Classifier performed ok after being trained with 60000 images. <br/>
&emsp; Also tried using a SVC (Support Vector Classifier) with the RBF Kernel Function and 30000 images as train. <br/>
&emsp; The SVC model performed the best out of all the tried models until now. <br/>
&emsp; Tried a convolutional neural network using the TensorFlow library. It showed the best results so far. The accuracy is around 99.5%. <br/>
&emsp; The convolutional network had 3 convolutional layers, with max pooling 2D in between, followed by a dense layer and an output layer. <br/>

<br/>

**Example of usage:** <br/>

<p align = "center">
  <img width="800" height="533" src="https://github.com/Razvan48/Digit-Recognizer-in-Python/blob/main/demo/ezgif.com-video-to-gif-converter.gif">
</p>



