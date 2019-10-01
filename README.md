# MaskRcnn_tensorflow_cpp_inference
## inference mask_rcnn model with tensorflow c++ api
>this project  consists of three main operations
  >>1. keras model to tensorflow model: because we use front-end keras call the backend tensorflow,so we need to convert keras model to tensorflow model.  
  >>2. inference tensorflow  model with cpp,and  use `Eigen3` lib carefully.  
  >>3. note that such as  batch_size,the operation that save model in  gpu or cpu ,must be same as the config you set in the python call.  
  
## enviroment:  
  >tensorflow c++ library:(https://github.com/fo40225/tensorflow-windows-wheel)  this project's `tf version 1.8.0 avx2 gpu`.  
  >cuda(if use gpu): cuda 9.1   
  >protobuf(if use gpu)： protobuf 3.6   
  >opencv: 3.3.0  
  >system : win10  
  >gui tool :qt 5.8.0  
  >compile tool : msvc2015  
  
## tutorials:
  ### keras model to tensorflow model
    > download the front-end keras  mask_rcnn model and install it  https://github.com/matterport/Mask_RCNN  
    > download this https://github.com/parai/Mask_RCNN for converting keras model to tensorflow model 
    > modify matterport's Mask_RCNN/samples/coco/coco.py  
    
  
    
    
  
