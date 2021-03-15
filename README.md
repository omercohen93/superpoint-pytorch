# superpoint-pytorch
This file is a pytorch implementation and evaluation of Superpoint model as described in https://arxiv.org/pdf/1712.07629v4.pdf.

We found great help in Rémi Pautrat’s tensorflow implementation: https://github.com/rpautrat/SuperPoint.


In interest point detection, our model seems to not fully converge:
![image](https://user-images.githubusercontent.com/73498160/111214173-4ca24600-85da-11eb-8fd3-2681b2f49719.png)

But still, the results of homogrphic addaptation combined with the system seems good:
![image](https://user-images.githubusercontent.com/73498160/111214201-55931780-85da-11eb-8001-57b1807bdb1b.png)

To see in comparison to other point detection models:

![image](https://user-images.githubusercontent.com/73498160/111214834-16b19180-85db-11eb-981e-29d950b2cf8a.png)

The overall results do not reach the tracking ability as of the original model.
with original model, the matching points are:

![image](https://user-images.githubusercontent.com/73498160/111215142-77d96500-85db-11eb-8fe2-c25bd7d8ee83.png)

with our implementation:

![image](https://user-images.githubusercontent.com/73498160/111215197-8c1d6200-85db-11eb-9e06-f04815a94b86.png)


Though overall results do not reach satisfying abilities, we hope the different blocks (data genereation, homographic adaptation and so on) can be of use to some future work.


## Within this file are all stages of implementation:
### 1)Generate synthetic dataset- 
creates a dataset containing 100000 images of self made synthetic shapes, together with the dataset file containing images names and labels
This part takes about 12 hours on tesla v-100
### 2)Magicpoint_training_with_synthetic_dataset- 
Training magic point model as described in the paper, with the exception that in our implementation we use premade data, and not created data on the fly. We train for about 40000 iterations. This part takes 7 hours
### 3)Homographic adaptation- 
creates pseudo ground truth for coco images, so we can train magic point on coco. This part takes around 14 hours for an 87000 images dataset.
### 4)Magic point training on coco- 
similar to 2, only with different transformations to the data.

3 and 4 can be used iteratively, just change to the right path in paths.around 20000 iterations where ran, which took around 12 hours. 

### 5)Super point training on coco- 
using the full super point architecture. Creating the twin image for training is within the train function. In this part we used a small minibatch of 3 images, due to large memory consumption by the descriptors. We trained for 250000 iterations, which took 8 hours.


All the described process was made in google cloud, and the resulting weights are presented at weights folder. Pre Trained original weights are presented at pretrained weights.
The Superpoint architecture can be viewed in ‘models’ folder.
The different outcome point detection of every stage can be viewed in ‘test pretrained model’, just change the paths in paths headline. 


The evaluation method we proposed, using a 10 seconds video with small movements, together with imaging of point matching ability of different models are presented at video evaluation. 

All the data can be saved using the 'download_dataset' file in datasets. 
For superpoint training We used coco train2014.

