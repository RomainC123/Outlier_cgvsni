# Outlier_cgvsni


Computer generation of photo-realistic images has become potent enough that some images can now pass as natural images to the human eye. In order to avoid potential abuses in the future, it has become necessary to come up with tools to realize that for us. CNN-based deep neural networks have shown good promises in achieving good classification accuracy. However, standard supervised training has proven to be very sensitive to domain shift. Through the use of semi-supervised learning, this article aims to study a potential solution to this issue, comparing the results on four different datasets of high quality CG images.

This project is the continuation of a work that was started throughout a 6 months internship. The base idea (flow models and outlier detection applied to image classification) started as an offshoot of a project realized at the start of the year (fillière recherche) which utilized these models to do outlier detection of fraudulent banking data.

## Context and objectives

State-ot-the-art for this problem is the following article : https://hal.archives-ouvertes.fr/hal-02929038/file/manuscript_di.pdf

As shown by the results of the experimentation in the above work, CG image detection is a task that is very sensible to domain shift (here, changing the algorithm that renders the CG images). To address this issue, a semi-supervised approach (only labeling part of the dataset, and leaving the rest unlabeled) is used.

The goal here is to use the supervised part of the dataset (containing natural and CG images) to train a "mapping" of images into a set number of features. This model will then be applied to all of the remaining unlabeled images, to turn them into their latent features ; and this mapping will then be used to train a flow model using Deep SVDD, in order to allow for outlier detection.

## Image map

To provide us with our mapping, we use the model studied in our reference article. This model, ENet (fig 1), is designed with and output layer to be used with a standard supervised learning scheme (use crossentropy loss). But we can see that the layers L8 and L9 are fully connected layers, at the end of the processing chain. Hence their outputs should be good mappings of the relevant features of each image. This is why the image map model is going to be a ENet mode, but where the actual mapping is taken from the second-to-last last layer (which is here L9).

![figure1](enet.png)
<b>Fig.1 - ENet base model used for direct supervised classification</b>

In order to train the mapping, we use the same training method as stated in the reference article :
- Each image is center cropped (to allow the use of any image of sufficient size, regardless of dimensions) and normalized
- Training is done over 60 epochs
- Loss is calculated using torch.nn.CrossEntropyLoss

We can do this supervised training step only with the labeled part of our dataset.

## Flow model and outlier detection

Now that we have access to a mapping of any image, we can train our outlier detection stage. Note that this stage has to be done independently from the previous stage, as one of the main requirements of FlowSVDD training is a model with constant jacobian for every entry possible, which would not be the case if we just trained a singular model combining our ENet stage with our flow model stage. The main ressource for flow SVDD is the following article : https://arxiv.org/pdf/2108.04907.pdf

The main idea in flow SVDD is to train a model to map our feature space to another, same dimensional space of latent features, such as out data is a tightly packed as possible. Jacobian being constant, the mapping keeps volumes unchanged, with a multiplicative constant. This forces the algorithm to actually learn something instead of producing trivial solutions (like mapping everything to the same point, which would indeed minimize the area of the sphere enclosing the data in our latent space).

This approach relies on the supposition that the feature mapping that we got out of the first step of our training will be able to be trained is such a way that CG images are actual outliers.

The following loss is used :

![\Large F(R,c,f)=R^2+frac{1}{\nu n}\sum_i max(0,\lVert f(x_i)-c\rVert^2-R^2)](<https://latex.codecogs.com/svg.latex?\Large&space;F%28R,c,f%29=R^2+\frac{1}{\nu%20n}\sum_i%20max%280,\lVert%20f%28x_i%29-c\rVert^2-R^2%29>)

The parameters c and R are learned through the training process, and f is our trained function. <span>&#957;</span> the hyperparamater controlling how much of the training data should be considered as outliers. This is usually set between 0.1 and 0.01 for our purposes here.

An important fact to note here is that we will only train this step using natural images. As we are trying to teach our model to map images such as natural images are tightly packed together, we don't need to use CG images, and we can set <span>&#957;</span> to be quite low (as none of our images are actually outliers). This approach is also relevant here as our dataset is imbalanced towards natural images (we train on a balanced dataset, but we have a lot more natural images to choose from than CG images). This allows us to leverage these readily available images without any efforts.

As for R and c, they are handled differently :
- c is initialized before the first epoch. As the value of c doesn't matter (it being the center of the ball, changing c would be equivalent to translating our latent space, which doesn't change anything) it's taken as the average of all features over the latent space, when passing though the randomly initialized flow model.
- R has to be updated, but only so often. This training method only updates it every two epochs. The update is made to put R as the value of the distance between c and the value of the dataset that lies at the position (1 - <span>&#957;</span>) * total number of images, when all distances are ordered in ascending order. This makes R the radius of the ball that includes exactly 1 - <span>&#957;</span> %  of the training images in our latent space, at the time of the update.

The full updating process and loss calculations for our flow model can be found in loss.py.

## Training

The dataset is quite massive, and cannot be uploaded to github. To fetch it in order to run the code, please go to https://drive.google.com/file/d/18-jdgU7OHj56bua4wMQ6yboMzWa_qXyz/view?usp=sharing. The archive should be decompressed into the dataset folder, so that the folder looks like

dataset    
├── Artlantis  
├── Autodesk  
├── Corona  
├── RAISE  
├── VISION  
├── VRay  
└── dataset.csv

In order to have comparable data to the reference article, the training is conducted using the same hyperparamaters (for the image mapping process). The idea is to train multiple mapping models, each using a dataset made up of natural images and of CG images from one rendering algorithm, then to train a flow model off of each of these mappings, to be able to see if the addition of the flow model improves the performances compared to the case of a more traditional classification using a fully-connected layer and a fully labeled dataset.

To run the training process, be sure to unpack everything in the right folders, and install all required dependancies (pip install -r requirements.txt). Then run train_img_map.py, which will create a model in the trained model folders. When that is done, run train_flow.py (you will need to provide the name of the model you just trained through the --model_name argument in your command, the model name being the datetime of the moment you started the first training step). Be aware that these models are pretty massive and take a very long time to train (~4h for img map training with 1260 images on an RTX 2070).

## Results

These will be updated as I get time to train different models, the time constraint here being a major player (roughly 20h of training on my computer to train an actual comparable img_map model in reference to the state-of-the-art).
