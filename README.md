# MaskRcnn_tensorflow_cpp_inference
# If you have any good Suggestions, please give them to me. I would really appreciate it
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
    > 1.modify matterport's Mask_RCNN/samples/coco/coco.py
    
   ![Image](https://github.com/CasonTsai/MaskRcnn_tensorflow_cpp_inference/blob/master/images/1.png)
   
    > 2.modify inference config ,especially parameter GPU_COUNT and IMAGES_PER_GPU,these two parameter must be same as that of parai's Mask_RCNN-master/samples/demo.py ,otherwise will get error. parameter  IMAGES_PER_GPU   involve the image nums when we use batch inference 
    ,mine is 32 , 1080 ti could handle  32 images with  size is  512*512.  
    
   ![Image](https://github.com/CasonTsai/MaskRcnn_tensorflow_cpp_inference/blob/master/images/2.png)
   
    > 3.modify the  class nums ,and IMAGE_MIN_DIM ,IMAGE_MAX_DIM in class CcocoConfig,int this project IMAGE_MIN_DIM = 512,IMAGE_MAX_DIM = 512,class nums is 1+6,these parameter according to yours
    
   ![Image](https://github.com/CasonTsai/MaskRcnn_tensorflow_cpp_inference/blob/master/images/3.png)
   
    > 4.  run the coco.py ,we can get the keras model(model+weight),this project's keras model name is mask_rcnn_whole_batch32_new20.h5  
    > 5.  begin convert keras model to tensorflow model, the following operations are mainly make some config in `parai's Mask_RCNN-master/samples/demo.py` be same as  `matterport's Mask_RCNN/samples/coco/coco.py` 
    >>  5.1 modify parameter NUM_CLASSES and IMAGE_MIN_DIM  IMAGE_MAX_DIM of  CocoConfig in  `parai's Mask_RCNN-master/samples/demo.py` 
    
   ![Image](https://github.com/CasonTsai/MaskRcnn_tensorflow_cpp_inference/blob/master/images/4.png)  
   ![Image](https://github.com/CasonTsai/MaskRcnn_tensorflow_cpp_inference/blob/master/images/5.png)  
   
    >>  5.2 modify inferenceConfig in  `parai's Mask_RCNN-master/samples/demo.py`  
   ![Image](https://github.com/CasonTsai/MaskRcnn_tensorflow_cpp_inference/blob/master/images/6.png)  
   ![Image](https://github.com/CasonTsai/MaskRcnn_tensorflow_cpp_inference/blob/master/images/7.png)    
   
    >>  5.3 modify `parai's Mask_RCNN-master/scripts/export_model.py` 
   ![Image](https://github.com/CasonTsai/MaskRcnn_tensorflow_cpp_inference/blob/master/images/8.png)  
    
    >>  5.4 run `parai's Mask_RCNN-master/samples/demo.py` we will get the mask_rcnn tensorflow model finally,tf model file name in this project's is mask_rcnn_batch32_new20.pb ,in this step ,we finish conveter the keras model into tensorflow model,we  could  use `parai's Mask_RCNN-master/infere_from_pb.py` to test the pb file to check  whether result is correcet or not
    
  ### c++ inference  tensorflow model 
    > 1.  all we need cpp file are `data_format.h,detectbatch.cpp ,detectbatch.h`   
    > 2.  the main.cpp is show how to use `data_format.h,detectbatch.cpp ,detectbatch.h` to detect the image 
  
  ![Image](https://github.com/CasonTsai/MaskRcnn_tensorflow_cpp_inference/blob/master/images/result.png)  
   
  
    
    
    
  
    
    
  
