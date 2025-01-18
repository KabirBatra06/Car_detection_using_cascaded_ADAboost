## Implementation of AdaBoost

To begin my implementation of AdaBoost I begin by creating low level features for all the images. Haar filters are used to create these low-level features. I use harr filters of even length and width, ranging from 2 elements wide all the way to the size of the image. 
Next, I label all the images. Positive car images are labeled as 1 and negative car images are labeled as 0.
Once I have the labeled images and the low-level features I begin by assigning a weight to each datapoint. These weights are initially uniformly distributed. Each positive image and negative image have the same probability distribution. 
Once this distribution is created, I begin by finding weak classifiers in the images. I do this by picking one feature from all the images and then sorting it in ascending order. 

Then the sorted feature list is used to calculate errors for two polarities let’s call them polarity 1 and polarity 2.

My error for polarity 1 is calculated as follows:

$$
{e_1\=S_pos+A_neg-S_neg}
$$

Here S_pos is the cumulative sum of all the weights of positive labels less than the threshold  
S_neg is the cumulative sum of all the weights of negative labels less than the threshold  
A_neg is the sum of all the weights of negative labels less than the threshold  

My error for polarity 2 is calculated as follows:

$$
{e_2\=S_neg+A_pos-S_pos}
$$

Here S_neg is the cumulative sum of all the weights of negative labels less than the threshold  
S_pos is the cumulative sum of all the weights of positive labels less than the threshold  
A_pos is the sum of all the weights of positive labels less than the threshold  

Once I have found all the errors, I look through them all to find the lowest error. The polarity from which we get the lowest error is the polarity for that weak classifier and the feature value at the index of the lowest error becomes the new threshold value. We use this threshold value to label all the images. This becomes our weak classifier h_t

Then I calculate the trust factor for this classifier using the following formula:

$$
\alpha_t\=\frac{1}{2} ln⁡((1-ϵ_t)/ϵ_t )
$$

Here  

$$
ϵ_t\=\frac{1}{2} ΣD_t (x_i ).|h_t (x_i )-y_i |
$$  

D_t (x) are the weights and y_i are the true labels  

Next, I update the weights based on this classifier using the following formula:

$$
D_{t+1} x_i \=D_t (x_i ) e^{-αy_i h_t (x_i ) ) )} / Z_t
$$

And

$$
Z_t \= ΣD_t x_i e^{-αy_i h_t x_i}
$$

I do this for 7 iterations per cascade. The best weak classifier from all the cascades is used to classify the images for that cascade.
I calculate the False positive rate and the false negative rate from this classifier and store that in a list. 
The list of all the False positive rates and the false negative rates for each of the cascades is stored in a list and then plotted as seen in the outputs.

When I go from one cascade to the next, the only images that get passed through are the ones that are classified as positive images by the previous classifier. The same steps mentioned above are repeated at every cascade. I implemented 5 cascades in my code. The tolerance I choose for the false positive rate was 0.000001

Finally for testing, features are generated for the testing images and then they are put through the same weak classifiers that were previously deemed the best at every cascade. The threshold and polarity that was determined before are the ones that I use for the testing images as well. Similar to training I maintain a list of false negative rates and false positive rates for each cascade in the testing set.

# Results

<p align="center">
  <img src="https://github.com/KabirBatra06/panorama_generation/blob/main/1.jpg" width="350" title="img1">
</p>
