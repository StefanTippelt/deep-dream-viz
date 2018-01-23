# Visual art generation with Deep Dream and Tensorflow

Creating visually appealing art by using deep neural nets and [Google’s TensorFlow](https://github.com/tensorflow/tensorflow) is in all mouth. Hence it is not always straight forward to adjust existing models to your own needs and the tuning of several parameters in different layers is still a mainly manual task.
Therefore, a method is developed to generate a lot of images with varying parameters and using different layers as activations for [Google’s deep dream](https://github.com/google/deepdream) inception model inception5h. The tool addresses the need for speeding up the exploration of model characteristics that make the result visually appealing. Finally, the exploration output is displayed in a parameter_a x parameter_b grid to semi-automate the exploration of visually interesting layers and parameters.

The code in this repository is based on [Hvass Labs Tutorials](https://github.com/Hvass-Labs/TensorFlow-Tutorials).

![Example 1 for urban people](https://github.com/StefanTippelt/deep-dream-viz/blob/master/src/img/blended_pictures/urban_people1.jpg)
![Example 2 for urban people](https://github.com/StefanTippelt/deep-dream-viz/blob/master/src/img/blended_pictures/urban_people2.jpg)
![Example 3 for urban people](https://github.com/StefanTippelt/deep-dream-viz/blob/master/src/img/blended_pictures/urban_people3.jpg)

## Required Packages

The tutorials require several Python packages to be installed, these are listed in requirements.txt
An easier way to run the project is to use provided docker file to create your own docker image. You do not have docker installed yet? Get if from here: [install docker](https://docs.docker.com/engine/installation/).
Then just run:

 ```make build```

 ```make run```


## License (MIT)
Published under the MIT License. See the file LICENSE for details.

The images used in the tutorials are from pixabay.com and made publicly available under [Creative Commons CC0 License](https://creativecommons.org/publicdomain/zero/1.0/deed.en)
