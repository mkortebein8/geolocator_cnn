# Geolocation Convolutional Neural Network

## GEOG 5545 (Geospatial + X) Project

### **Description:** \
In this project, I attempted to create a CNN which has the ability to guess the location of images taken on the ground (non-satellite). A significant inspiration was the game GeoGuessr, in which you are given a google streetview image and prompted to guess where the image is on Earth. I happen to not be very good at this game, and started wondering about how an AI might perform.

I chose to use a convolutional neural network, as there have been [successful implementations of CNNs tasked with geolocation in the past](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45488.pdf). I considered a transformer architecture, but as this was my first attempt at something of this scale I figured that starting with a simpler architecture would be best.

My implementation utilizes S2 Cells to classify areas of the Earth; as such my script uses the [s2sphere Python package](https://s2sphere.readthedocs.io/en/latest/). 

**How to run:**
1. Download the shard files from [this website](https://www.kaggle.com/datasets/habedi/large-dataset-of-geotagged-images/data). Put these files into the directory /data relative to the script.
2. You are free to run the script as is, or change the parameters of the model.\
WARNING: If you are using the entirety of the data, training will take a long time. A training loop of five epochs took me ~40 GPU hours on one NVIDIA A100 GPU with 40 GB VRAM.

**Resources:**\
Inspiration came from the [PlaNet](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45488.pdf) and [TransGeo](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhu_TransGeo_Transformer_Is_All_You_Need_for_Cross-View_Image_Geo-Localization_CVPR_2022_paper.pdf) papers.
The Python [S2Sphere package](https://s2sphere.readthedocs.io/en/latest/) was used extensively.
