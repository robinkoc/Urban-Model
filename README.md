# Urban-Model
The model I trained and gathered statistics for the admission to research group

Uses Cuda if avalible, else CPU. 
I used about 5000 of the data for training and 1500 of it for testing.
Inputs are scaled and randomized.

There are 2 for loops, outer one is learning rates and inner one is 16 models I added. It tries all the combinations of these values and prints which combination gives the best r2 value on which epoch. I got learning rate: 0.0005 with mode(model): 8 combination gives best r2 value at around 700 epoch. 
16 modes (models) have: one 1, three 2, three 3 and one 4 layered model with one of each using hardwish or ReLU

Implemented an early stop mechanism to stop training when validation loss starts increasing again to keep model from overfitting. 

After choosing combination gives out statistics and plot of predictions(blue) and real values(orange). Writes some statistics and predicted values, real values and difference of each value pair to a .txt file called statistics.txt

#How to Run
just download the folder "/urban" and run the "urban_robin.py" file, code uses relative paths so no need to change anything 
