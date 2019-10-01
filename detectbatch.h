#ifndef DETECTBATCH_H
#define DETECTBATCH_H

//#include <Windows.h>
#include <math.h>
#include <iostream>
#include <iomanip>
#include <opencv2/core.hpp>
#include <vector>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/Dense>  //matrix library
#include "data_format.h"

class detectBatch
{
public:

    detectBatch();
    void loadModel(std::string model_path);
    ~detectBatch();
    /*---coco Config---*/
    //the below is only for deploy
    int RPN_ANCHOR_SCALES[5]={32, 64, 128, 256, 512}; //rpn阶段生成的anchor的尺度
    float RPN_ANCHOR_RATIOS[3]={0.5, 1, 2};//rpn阶段生成的anchor的缩放因子
    float BACKBONE_STRIDES[5]={4, 8, 16, 32, 64};//用于计算输入图像经过backbone的每一个阶段(可能是pooling或者conv等down sample操作导致feature map缩小后的尺寸,这一部分没细看)后feature图的长宽
    int  RPN_ANCHOR_STRIDE =1;//rpn阶段生成的anchor之间的间隔
    float MEAN_PIXEL[3]={123.7f,116.8f,103.9f};//图像三个通道对应的要减去的均值
    //下面是记录数组对应的个数,貌似可用sizeof来计算.....算了不想每次都计算,总觉得以后会用到...
    int rpn_anchor_scales_num=5;
    int backbone_strides_num =5;
    int rpn_anchor_ratios_num =3;

    //输入的图像数目和类别数目,下面只是默认的,根据需要来改
    int input_num=1;//input images
    int num_classes=6+1;//classes num + bg
    int TF_MASKRCNN_IMAGE_METADATA_LENGTH;//=12+num_classes;//这里的19是12+7(7是有七类)  //图像的meta数据的长度,一般
    /*---coco Config---*/

    std::string input_tensor_name[3]={"input_image_1","input_image_meta_1","input_anchors_1"};//网络输入的tensor的名字,对应于模型
    std::string output_tensor_name[5]={"output_detections","output_mrcnn_class",
                                       "output_mrcnn_bbox","output_mrcnn_mask","output_rois",

                                      };//网络输出的tensor的名字,对应于模型
    int input_width;//输入tensor的宽度
    int input_height;//输入tensor的高度
    int input_channels=3;//输入tensor的通道 input tensor channels,same as image channels
    int inputImg_w=512;//输入图像的宽度 image width
    int inputImg_h=512;//输入图像的高度 image height
    int inputImg_c=3;//输入图像的通道 image channels
    float image_meta[19]={};//for image_meta
    int backbone_shape[5][2];//for backbone_shape
    int _anchor_cache[2]={};//用于缓存 for cache 
    std::vector<tensorflow::Tensor> outputs;
    std::vector<imageDetectInfo> outputsInfo;

    bool session_open=false;
    tensorflow::Session* session;//tensorflow'ssession
    tensorflow::Session *session_gpuConfig=nullptr;//设置gpu的config gpu's config
    tensorflow::GraphDef graphdef;//模型的计算图
    void initConfig(int input_w, int input_h);
    std::vector<cv::Mat> inputImg_list;
    void CVMats_to_Tensor(std::vector<cv::Mat> &imgs,tensorflow::Tensor *input_tensor,size_t &imgNum_actual);
    void runBatch(tensorflow::Tensor&input_tensor,std::vector<tensorflow::Tensor>*output_tensors);
    std::string wholeImagePath;//大图路径
    Eigen::MatrixXf finalBox;
    Eigen::MatrixXf finalBox_norm;
    Eigen::MatrixXf finalboxMat;
    int batch_size=12; //batch size 
    int batch_num=0;//how many batches
    int batch_mod=0;//the rest images
    void unmold_detections(std::vector<tensorflow::Tensor>&output_tensors,std::vector<imageDetectInfo> &output_vec);
private:
    //detectBatch detectBatchTmp;
    tensorflow::Tensor resized_tensor;//输入的tensor
    tensorflow::Tensor inputMetadataTensor;//图像元数据
    tensorflow::Tensor inputAnchorsTensor;//图像anchors数据
    void compose_image_meta();
    void get_anchors();



};


#endif // DETECTBATCH_H
