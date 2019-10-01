#include "mainwindow.h"
#include <QApplication>

#include "detectbatch.h"
#include <iostream>
#include <vector>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    detectBatch detectBatchTmp;//模型预测器 model predictor 
    std::vector<tensorflow::Tensor> outputs;//用于获取预测结果的tensor类型容器 vector for saving   model's output 
    std::vector<imageDetectInfo> outputsInfo;//存储从output tensor提取出来的输出 vector for extracting the result from output tensor
    int batch_size =32;//batch要与把keras模型转tensorflow模型时设置的数目一致,我的是32 batch size ,must be same as that time we convert keras model to tensorflow model
    size_t imgNum_actual=32;//actual image nums ,must be not more than batch_size
    int detect_size_w=512,detect_size_h=512,input_dim=3;//image size and channels
    detectBatchTmp.batch_size=32;
    std::vector<cv::Mat> imgData(batch_size);//用于临时存放img的容器 vector for storing the cv::mat image
    std::string imgPath1="E:\\SourceCode\\Qt\\QtLearn\\src\\testopencv\\untitled6\\test1.jpg";//image file name 
    std::string imgPath2="E:\\SourceCode\\Qt\\QtLearn\\src\\testopencv\\untitled6\\test2.jpg";//image file name2
    std::string modelPath="E:\\Maidipu\\code\\test_tensorflow_model\\mask_rcnn_batch32.pb";//tensorflow pb file path
	
	
	//get the image and put them to cv::Mat vector 
    cv::Mat imgTest1=cv::imread(imgPath1);
    cv::Mat imgTest2=cv::imread(imgPath2);
    outputsInfo.resize(batch_size);
    for(int i=0;i<batch_size;i++){
        if((i%2)==0)
            imgData[i]=imgTest1.clone();
        else
            imgData[i]=imgTest2.clone();

    }

    int class_num =7;//class_num,modify it according to yours 
    detectBatchTmp.num_classes=class_num;
    detectBatchTmp.TF_MASKRCNN_IMAGE_METADATA_LENGTH=12+class_num;
    detectBatchTmp.initConfig(detect_size_w,detect_size_h);
    detectBatchTmp.loadModel(modelPath);

	//create input_tensor
    tensorflow::Tensor input_tensor=tensorflow::Tensor(tensorflow::DT_FLOAT,{batch_size,detect_size_h,detect_size_w,input_dim});//input tensor维度设置为batch大小//用于临时存放输入图像的tensor
    detectBatchTmp.CVMats_to_Tensor(imgData,&(input_tensor),imgNum_actual);//convert cv::Mat vector  to  input_tensor

    detectBatchTmp.runBatch(input_tensor,&(outputs));//inference begin 
    detectBatchTmp.unmold_detections(outputs,outputsInfo);//get the output result and extract it to outputsInfo

	
	//show the detect result 
    for(int i=0;i<outputsInfo.size();i++){
        auto &detectInfo=outputsInfo[i].detectInfo;
        for(int j=0;j<detectInfo.size();j++){
            std::cout<<"j :"<<j<<" y1 :"<<detectInfo[j].y1<<" x1 :"<<detectInfo[j].x1 <<" y2 :"<<detectInfo[j].y2<<" x2 :"<<detectInfo[j].x2<<std::endl;
            std::cout<<"class id : "<<detectInfo[j].classId<<std::endl;
            std::cout<<"scores : "<<detectInfo[j].scores<<std::endl;
            }
    }
    MainWindow w;
    w.show();




    return a.exec();
}
