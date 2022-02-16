# Image Captioner

[LIVE DEVELOPMENT NOTEBOOK](https://seanvonb.github.io/image-captioner/)

This is another project that was part of my [Computer Vision Nanodegree](https://www.udacity.com/course/computer-vision-nanodegree--nd891) from 2020. In this notebook, I cover the process of developing an image captioner: a network that receives previously unseen images and returns written descriptions of their content. It combines Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs) with Long Short-Term Memory (LSTM) cells to create a network with both feedforward and feedback connections. These connections allow the network to maintain memory between steps and handle sequential data, like language, where the last output determines the next output as much as the current input does.

Beyond the obvious but incredible difference an image captioner can make in terms of user accessibility, this project demonstrates a network that's capable of infering *contextual nuance*, which has countless other applications and is simply fascinating. On the other hand, this project sometimes demonstrates a network that's *incapable* of infering contextual nuance; which, while disappointing, can also be pretty funny.

Probably the most valuable outcome of this project will be the [development notebook](https://seanvonb.github.io/image-captioner/), which I think could be a useful reference for future machine learning projects. Please take a look if this sounds interesting to you!

## Features

-   Generate a caption of about 10 words for any given image
-	Explore the key differences between CNNs and RNNs
-	Learn how to use and build a custom data loader for language tasks
-	Construct an encoder-decoder pipeline for sequential data inference

## Credits

-   This project was part of [Udacity's Computer Vision Nanodegree](https://www.udacity.com/course/computer-vision-nanodegree--nd891).
-   Image data is provided by Microsoft's [Common Objects in Context](https://cocodataset.org/#home) and [COCO API](https://github.com/cocodataset/cocoapi).

## License

Copyright © 2020-2022 Sean von Bayern  
Licensed under the [MIT License](LICENSE.md)
