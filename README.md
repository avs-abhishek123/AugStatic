<img src = 'https://github.com/avs-abhishek123/AugStatic/blob/8809c0700d3a8900fbe3e92ebc47ca39f2304922/augstatic.png' width = 1020 height = 450 align = "center">

<!--<h1 align ="center" style="font-size: 1400px"> AugStatic - A Light-Weight Image Augmentation Library </h1>-->
<h1 align ="center" style="color: purple; font-size: 80px;"><b><u>AugStatic - A Light-Weight Image Augmentation Library</u></b></h1>

## Abstract
<p> <a href = 'https://www.jetir.org/papers/JETIR2205199.pdf'> [ Paper Link ] </a> </p>
The rapid exponential increase in the data led to an abrupt mix of various data types, leading to a deficiency of helpful information. Creating new data with the existing different types of data are presented in this paper. Augmentation is adding up or modifying the dataset with extra data. There are many types of augmentation done for various kinds of datasets. Augmentation has been widely used in multiple pre-processing steps of diverse machine learning pipelines. Many libraries or packages are made for augmentation called augmentation libraries. There are many salient features that each library supports. This paper seeks to enhance the library that makes the AugStatic library much more lightweight and efficient. AugStatic is a custom-built image augmentation library with lower computation costs and more extraordinary salient features compared to other image augmentation libraries. This framework can be used for NumPy array and tensors too.


---

## Brief Introduction:

* In this research, An light weight Efficient Augmentation library has been developed, named AugStatic
* AugStatic is a custom-built image augmentation library with lower computation costs and efficiency compared to other image augmentation libraries. 
* This framework can be used for NumPy arrays and tensors too.

---

## Background Research Work

### Various types of augmentations were researched and compiled into a compact, lightweight, and practical library.

* ### **Imgaug**  [ [Library-GitHub-Link](https://github.com/aleju/imgaug) | [Documentation-Link](https://imgaug.readthedocs.io/en/latest/) ] 


  <p>
    <img src = 'https://github.com/avs-abhishek123/AugStatic/blob/fed5dd54e822dac6ffc3dd37ede0e2b39ecfef8d/imgaug2.png' width = 350 height = 200 align = "right">
    <h4> The salient feature are – </h4>
    
    <ul>
      <li> It contains over forty image augmentation techniques. </li>
      <li> Functionality to augment images with masks, key points, bounding boxes, and heat maps. </li>
      <li> Easier to augment the image dataset for object detection and segmentation problems. </li>
      <li> Complex augmentation pipelines. </li>
      <li> Many helper functions for augmentation visualization, conversion, and more. </li> 
    </ul>
  </p>
  
* ### **Augmentor** [ [Library-GitHub-Link](https://github.com/mdbloice/Augmentor) | [Documentation-Link](https://augmentor.readthedocs.io/en/stable/) ]
  <p>
  <img src = 'https://github.com/avs-abhishek123/AugStatic/blob/8809c0700d3a8900fbe3e92ebc47ca39f2304922/augmentor.png' width = 350 height = 200 align = "right">
  <h4> The salient feature are – </h4>
  
  <ul>
    <li> It has fewer possible augmentations compared to other packages. </li>
    <li> It supports extra features like size-preserving shearing, size-preserving rotations, and cropping, which is beneficial for machine learning pipelines. </li>
    <li> It supports to compose augmentation pipelines. </li>
    <li> It supports usage with PyTorch [8] and Tensorflow [1]. </li>
  </ul>
  </p>
  
* ### **Albumentations** [ [Library-GitHub-Link](https://github.com/albumentations-team/albumentations) | [Documentation-Link](https://albumentations.ai/docs/) ] 

  <p>
  <img src = 'https://github.com/avs-abhishek123/AugStatic/blob/8809c0700d3a8900fbe3e92ebc47ca39f2304922/albumentations.png' width = 350 height = 200 align = "right">
  <h4> The salient feature are – </h4>
  
  <ul>
    <li> It contains over forty image augmentation techniques. </li>
    <li> Functionality to augment images with masks, key points, bounding boxes, and heat maps. </li>
    <li> Easier to augment the image dataset for object detection and segmentation problems. </li>
    <li> Complex augmentation pipelines. </li>
    <li> Many helper functions for augmentation visualization, conversion, and more. </li>
    </ul>
  </p>
  
---

## Results 

| Augmentation Technique | Output Image |
| --- | --- |
| Blur | ![Blur](https://github.com/avs-abhishek123/AugStatic/blob/0f12cb0adf7b0c9bd68074f8891907067604e79f/OutputImages/Blur.jpg) |
| CLAHE(Contrast Stretched Adaptive Histogram Equalization) | ![CLAHE](https://github.com/avs-abhishek123/AugStatic/blob/0f12cb0adf7b0c9bd68074f8891907067604e79f/OutputImages/CLAHE.jpg) |
| ChannelDropout | ![ChannelDropout](https://github.com/avs-abhishek123/AugStatic/blob/0f12cb0adf7b0c9bd68074f8891907067604e79f/OutputImages/ChannelDropout.jpg) |
| ChannelShuffle | ![ChannelShuffle](https://github.com/avs-abhishek123/AugStatic/blob/0f12cb0adf7b0c9bd68074f8891907067604e79f/OutputImages/ChannelShuffle.jpg) |
| ColorJitter | ![ColorJitter](https://github.com/avs-abhishek123/AugStatic/blob/0f12cb0adf7b0c9bd68074f8891907067604e79f/OutputImages/ColorJitter.jpg) |
| Downscale | ![Downscale](https://github.com/avs-abhishek123/AugStatic/blob/0f12cb0adf7b0c9bd68074f8891907067604e79f/OutputImages/Downscale.jpg) |
| Emboss | ![Emboss](https://github.com/avs-abhishek123/AugStatic/blob/0f12cb0adf7b0c9bd68074f8891907067604e79f/OutputImages/Emboss.jpg) |
| FancyPCA | ![FancyPCA](https://github.com/avs-abhishek123/AugStatic/blob/0f12cb0adf7b0c9bd68074f8891907067604e79f/OutputImages/FancyPCA.jpg) |
| GaussNoise | ![GaussNoise](https://github.com/avs-abhishek123/AugStatic/blob/0f12cb0adf7b0c9bd68074f8891907067604e79f/OutputImages/GaussNoise.jpg) |
| GaussianBlur | ![GaussianBlur](https://github.com/avs-abhishek123/AugStatic/blob/0f12cb0adf7b0c9bd68074f8891907067604e79f/OutputImages/GaussianBlur.jpg) |
| GlassBlur | ![GlassBlur](https://github.com/avs-abhishek123/AugStatic/blob/0f12cb0adf7b0c9bd68074f8891907067604e79f/OutputImages/GlassBlur.jpg) |
| HueSaturationValue | ![HueSaturationValue](https://github.com/avs-abhishek123/AugStatic/blob/0f12cb0adf7b0c9bd68074f8891907067604e79f/OutputImages/HueSaturationValue.jpg) |
| ISONoise | ![ISONoise](https://github.com/avs-abhishek123/AugStatic/blob/0f12cb0adf7b0c9bd68074f8891907067604e79f/OutputImages/ISONoise.jpg) |
| InvertImg | ![InvertImg](https://github.com/avs-abhishek123/AugStatic/blob/0f12cb0adf7b0c9bd68074f8891907067604e79f/OutputImages/InvertImg.jpg) |
| MedianBlur | ![MedianBlur](https://github.com/avs-abhishek123/AugStatic/blob/0f12cb0adf7b0c9bd68074f8891907067604e79f/OutputImages/MedianBlur.jpg) |
| MotionBlur | ![MotionBlur](https://github.com/avs-abhishek123/AugStatic/blob/0f12cb0adf7b0c9bd68074f8891907067604e79f/OutputImages/MotionBlur.jpg) |
| MultiplicativeNoise | ![MultiplicativeNoise](https://github.com/avs-abhishek123/AugStatic/blob/0f12cb0adf7b0c9bd68074f8891907067604e79f/OutputImages/MultiplicativeNoise.jpg) |
| Posterize | ![Posterize](https://github.com/avs-abhishek123/AugStatic/blob/0f12cb0adf7b0c9bd68074f8891907067604e79f/OutputImages/Posterize.jpg) |
| RGBShift | ![RGBShift](https://github.com/avs-abhishek123/AugStatic/blob/0f12cb0adf7b0c9bd68074f8891907067604e79f/OutputImages/RGBShift.jpg) |
| Sharpen | ![Sharpen](https://github.com/avs-abhishek123/AugStatic/blob/0f12cb0adf7b0c9bd68074f8891907067604e79f/OutputImages/Sharpen.jpg) |
| Solarize | ![Solarize](https://github.com/avs-abhishek123/AugStatic/blob/0f12cb0adf7b0c9bd68074f8891907067604e79f/OutputImages/Solarize.jpg) |
| Superpixels | ![Superpixels](https://github.com/avs-abhishek123/AugStatic/blob/0f12cb0adf7b0c9bd68074f8891907067604e79f/OutputImages/Superpixels.jpg) |
| ToGray | ![ToGray](https://github.com/avs-abhishek123/AugStatic/blob/0f12cb0adf7b0c9bd68074f8891907067604e79f/OutputImages/ToGray.jpg) |
| ToSepia | ![ToSepia](https://github.com/avs-abhishek123/AugStatic/blob/0f12cb0adf7b0c9bd68074f8891907067604e79f/OutputImages/ToSepia.jpg) |
| VerticalFlip | ![VerticalFlip](https://github.com/avs-abhishek123/AugStatic/blob/0f12cb0adf7b0c9bd68074f8891907067604e79f/OutputImages/VerticalFlip.jpg) |
| HorizontalFlip | ![HorizontalFlip](https://github.com/avs-abhishek123/AugStatic/blob/0f12cb0adf7b0c9bd68074f8891907067604e79f/OutputImages/HorizontalFlip.jpg) |
| Transpose | ![Transpose](https://github.com/avs-abhishek123/AugStatic/blob/0f12cb0adf7b0c9bd68074f8891907067604e79f/OutputImages/Transpose.jpg) |
| OpticalDistortion | ![OpticalDistortion](https://github.com/avs-abhishek123/AugStatic/blob/0f12cb0adf7b0c9bd68074f8891907067604e79f/OutputImages/OpticalDistortion.jpg) |
| GridDistortion | ![GridDistortion](https://github.com/avs-abhishek123/AugStatic/blob/0f12cb0adf7b0c9bd68074f8891907067604e79f/OutputImages/GridDistortion.jpg) |
| JpegCompression | ![JpegCompression](https://github.com/avs-abhishek123/AugStatic/blob/0f12cb0adf7b0c9bd68074f8891907067604e79f/OutputImages/JpegCompression.jpg) |
| Cutout | ![Cutout](https://github.com/avs-abhishek123/AugStatic/blob/0f12cb0adf7b0c9bd68074f8891907067604e79f/OutputImages/Cutout.jpg) |
| CoarseDropout | ![CoarseDropout](https://github.com/avs-abhishek123/AugStatic/blob/0f12cb0adf7b0c9bd68074f8891907067604e79f/OutputImages/CoarseDropout.jpg) |
| GridDropout | ![GridDropout](https://github.com/avs-abhishek123/AugStatic/blob/0f12cb0adf7b0c9bd68074f8891907067604e79f/OutputImages/GridDropout.jpg) |

---

## To cite my paper: 
|Citing Text|
|---|
| "AugStatic - A Light-Weight Image Augmentation Library", International Journal of Emerging Technologies and Innovative Research (www.jetir.org), ISSN:2349-5162, Vol.9, Issue 5, page no.b735-b742, May-2022, Available :http://www.jetir.org/papers/JETIR2205199.pdf |
