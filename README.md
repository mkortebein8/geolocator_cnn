# Geolocation Convolutional Neural Network

### GEOG 5545 (Geospatial + X) Project

**Description**\
**How to run**\
**Resources**

**Description** \
In this project, I attempted to create a CNN which has the ability to guess the location of images taken on the ground (non-satellite). A significant inspiration was the game GeoGuessr, in which you are given a google streetview image and prompted to guess where the image is on Earth. I happen to not be very good at this game, and started wondering about how an AI might perform. \

I chose to use a convolutional neural network, as there have been [successful implementations of CNNs tasked with geolocation in the past](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45488.pdf). I considered a transformer architecture, but as this was my first attempt at something of this scale I figured that starting with a simpler architecture would be best. \

My implementation utilizes S2 Cells to classify areas of the Earth; as such my script uses the [s2sphere Python package](https://s2sphere.readthedocs.io/en/latest/). 



