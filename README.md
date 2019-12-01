# AttentionCNN-MFOVCNN-HepatoCellularCarcinoma---Stanford-Work

Code for the paper : ANALYSIS OF MULTI FIELD OF VIEW CNN AND ATTENTION
CNN ON H&E STAINED WHOLE-SLIDE IMAGES ON HEPATOCELLULAR CARCINOMA

Full Paper : https://drive.google.com/open?id=1AQTnic4ZvNU431DyYPphshVZgEPuLuc5

Codes are provided in order to represent CNN Architectures since they require more than an explanation. Architectures targets 2 and 3 field of view and codes are adjusted according to that. 
Data loading systems are generally differ, thus only the code for architectures are published.
Takes two days to train 2 WSI images on high-end GPU of Stanford Sherlock.

Hepatocellular carcinoma (HCC) is a leading cause of cancer-related death worldwide[1]. Whole-slide
imaging which is a method of scanning glass slides have been employed for diagnosis of HCC. Using
high resolution Whole-slide images is infeasible for Convolutional Neural Network applications.
Hence tiling the Whole-slide images is a common methodology for assigning Convolutional Neural
Networks for classification and segmentation. Determination of the tile size affects the performance
of the algorithms since small field of view can not capture the information on a larger scale and
large field of view can not capture the information on a cellular scale. In this work, the effect of tile
size on performance for classification problem is analysed. In addition, Multi Field of View CNN is
assigned for taking advantage of the information provided by different tile sizes and Attention CNN
is assigned for giving the capability of voting most contributing tile size. It is found that employing
more than one tile size significantly increases the performance of the classification by 3.97% and
both algorithms are found successful over the algorithm which uses only one tile size.

