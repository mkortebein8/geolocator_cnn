# Geolocation Convolutional Neural Network
GEOG 5545 (Geospatial + X) Project

<img width="900" height="600" alt="image" src="https://github.com/user-attachments/assets/66739293-d79f-4d05-9ab9-31a17243ed4f" />


### **Description:**
In this project, I attempted to create a CNN which has the ability to guess the location of images taken on the ground (non-satellite). A significant inspiration was the game GeoGuessr, in which you are given a google streetview image and prompted to guess where the image is on Earth. I happen to not be very good at this game, and started wondering about how an AI might perform.

I chose to use a convolutional neural network, as there have been [successful implementations of CNNs tasked with geolocation in the past](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45488.pdf). I considered a transformer architecture, but as this was my first attempt at something of this scale I figured that starting with a simpler architecture would be best.

My implementation utilizes S2 Cells to classify areas of the Earth. Image classification is a supervised learning task, and in supervised learning training data need labels. S2 Cells are a way to split up a sphere (in our case the Earth) into separate cells, which are then used as labels/classes. I use a [Quadtree approach](https://en.wikipedia.org/wiki/Quadtree) to split S2 Cells, with cells being dense in areas where many pictures are located and sparse in areas where there are almost no pictures. I implement a recursive algorithm to keep splitting each S2 cell into four cells until one of two base cases is achieved: either the max cell level is reached or the number of pictures in a cell drops below a max photo threshold.

### **How to run:**
1. Download the shard files from [this website](https://www.kaggle.com/datasets/habedi/large-dataset-of-geotagged-images/data). Put these files into the directory "./data" relative to the script.
2. You are free to run the script as is, or change the parameters of the model.\
WARNING: If you are using the entirety of the data, training will take a long time. A training loop of five epochs took me ~40 GPU hours on one NVIDIA A100 GPU with 40 GB VRAM.

### **Discussion:**
The best accuracy I received on test data was 37%. As such, this is not a final implementation/deliverable. I hope to improve and possibly morph this model in the future, and eventually want to train for more epochs as time and GPU resources were a constraint. With that being said, 37% is decent for a problem of this complexity.

**Output example**
Here is an example image:
<img width="336" height="337" alt="image" src="https://github.com/user-attachments/assets/db1da373-d0b1-47ac-b1ad-fecb7bd07b3d" />

Using this image, I run it through the model and get a predicted class for which the image belongs to:
<img width="171" height="43" alt="image" src="https://github.com/user-attachments/assets/4d194f47-2887-40d7-8c00-f3fb258f0bda" />

The labelled class was 49, and the predicted class was 49. In this case the model guesses correctly.



### **Resources:**
Inspiration came from [PlaNet](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45488.pdf), [TransGeo](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhu_TransGeo_Transformer_Is_All_You_Need_for_Cross-View_Image_Geo-Localization_CVPR_2022_paper.pdf) and [Stanford](https://cs231n.stanford.edu/reports/2015/pdfs/CS231N_Final_Report_amanivp_jamesh93.pdf) papers.\
The Python [S2Sphere package](https://s2sphere.readthedocs.io/en/latest/) was used extensively.

Note: The best accuracy I received on test data was 37%. As such, this is not a final implementation/deliverable. I hope to improve and possibly morph this model in the future, and eventually want to train for more epochs as time and GPU resources were a constraint.
