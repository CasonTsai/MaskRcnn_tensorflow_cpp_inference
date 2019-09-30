#include "detectbatch.h"
detectBatch::detectBatch()
{

}
void detectBatch::loadModel(std::string model_path)
{
    tensorflow::SessionOptions opts;
    //
    //std::string aa="/gpu:0";
    //tensorflow::graph::SetDefaultDevice(aa, &graphdef);
    //opts.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(0.9);
   // opts.config.mutable_gpu_options()->set_allow_growth(true);
    std::cout<<"model path is !!!!!!!!! "<<model_path<<std::endl;
    tensorflow::Status status=NewSession(opts,&session);
    tensorflow::Status status_load=ReadBinaryProto(tensorflow::Env::Default(),model_path,&graphdef);//read model from pb file

    if (!status_load.ok())
    {
        std::cout<<"Error: loading model failed !"<<model_path<<std::endl;
        std::cout<<status_load.ToString()<<std::endl;
    }
     tensorflow::Status status_create=session->Create(graphdef);//import model to session

    if(!status_create.ok())
    {
        std::cout<<"Error: Creating graph in session failed !"<<status_create.ToString()<<std::endl;

    }
    session_open=true;//成功打开session
    std::cout<<"<-----Successfully created session and load graph---->"<<std::endl;

}

detectBatch::~detectBatch()
{

    std::vector<tensorflow::Tensor>().swap(outputs);
    std::vector<imageDetectInfo>().swap(outputsInfo);
    if(session_open){
        session->Close();
        std::cout<<"sessiong closed !!!!!"<<std::endl;
    }
}

void detectBatch::initConfig(int input_w,int input_h)
{
    input_width=input_w;
    input_height=input_h;
    compose_image_meta();
    get_anchors();
}

void detectBatch::CVMats_to_Tensor(std::vector<cv::Mat> &imgs, tensorflow::Tensor *input_tensor, size_t &imgNum_actual)
{
    /*
        *Function:  CVMats_to_Tensor
        *Description:  cv::mat图像容器转到tensorflow::tensor
        *Calls:
            1. ****
        *Called By:
          1. ****

        *InputList:
          1. imgs 存储cv::mat的图像容器 std::vector<cv::Mat> &
          2. input_tensor 要存储数据的tensor tensorflow::Tensor *
          3. imgNum_actual 实际要转换cv::mat张数 size_t &

        *OutPut:
          1. NULL
    */

    auto outputMap =input_tensor->tensor<float,4>();//获取tensor指针,注意这里outputMap是Eigen::tensor类型
    for(size_t b=0;b<imgNum_actual;b++)//遍历图像张数
    {

        for(int r=0;r<outputMap.dimension(1);r++)//遍历行数
        {
            for(int c=0;c<outputMap.dimension(2);c++)//遍历列数
            {
                //note that opencv mat image channel is B G R
                //减去均值
                outputMap(b,r,c,0)=imgs[b].at<cv::Vec3b>(r,c)[2]-MEAN_PIXEL[0];//R
                outputMap(b,r,c,1)=imgs[b].at<cv::Vec3b>(r,c)[1]-MEAN_PIXEL[1];//G
                outputMap(b,r,c,2)=imgs[b].at<cv::Vec3b>(r,c)[0]-MEAN_PIXEL[2];//B
            }

        }
    }

}

void detectBatch::runBatch(tensorflow::Tensor&input_tensor,std::vector<tensorflow::Tensor> *output_tensors)
{
    tensorflow::Status status_run=session->Run({{input_tensor_name[0],input_tensor},
                                    {input_tensor_name[1], inputMetadataTensor},
                                   {input_tensor_name[2], inputAnchorsTensor}},
                                    {output_tensor_name[0],output_tensor_name[1],
                                     output_tensor_name[2],output_tensor_name[3],
                                     output_tensor_name[4]},{},{output_tensors});

    if(!status_run.ok())
    {
        std::cout<<"Error: Run failed!"<<std::endl;
        std::cout<<"status: "<<status_run.ToString()<<std::endl;
    }

}
void detectBatch::compose_image_meta()
{
    /*
        *Function:  compose_image_meta
        *Description:  计算图像meta数据
        *Calls:
            1. ****
        *Called By:
          1. ****

        *InputList:
          1. NULL

        *OutPut:
          1. NULL
    */
    int imglongSide,inputlongSide;
    image_meta[0]=0;
    //original_image_shape: [H, W, C] before resizing or padding.
    image_meta[1]=inputImg_h;
    image_meta[2]=inputImg_w;
    image_meta[3]=inputImg_c;
    imglongSide=image_meta[1]>=image_meta[2]?image_meta[1]:image_meta[2];


    //image_shape: [H, W, C] after resizing and padding
    image_meta[4]=input_height;
    image_meta[5]=input_width;
    image_meta[6]=input_channels;
    inputlongSide=image_meta[4]>=image_meta[5]?image_meta[4]:image_meta[5];

    //window: (y1, x1, y2, x2) in pixels. The area of the image where the real image is (excluding the padding)
    image_meta[7]=0;
    image_meta[8]=0;
    image_meta[9]=input_height;//因为我的图像都是裁剪好再送进去的,所以窗口的长宽与实际图像长宽一致
    image_meta[10]=input_width;

    //scale: The scaling factor applied to the original image (float32)
    image_meta[11]=inputlongSide/imglongSide;

    //active_class_ids: List of class_ids available in the dataset from which the image came.
    for(int i=TF_MASKRCNN_IMAGE_METADATA_LENGTH-num_classes;i<TF_MASKRCNN_IMAGE_METADATA_LENGTH;i++)
    {
        image_meta[i]=0;
    }

    inputMetadataTensor=tensorflow::Tensor(tensorflow::DT_FLOAT, {batch_size, TF_MASKRCNN_IMAGE_METADATA_LENGTH});

    auto inputMetadataTensorMap=inputMetadataTensor.tensor<float,2>();
    for(int j=0;j<batch_size;j++)
    {
        for(int i=0;i<TF_MASKRCNN_IMAGE_METADATA_LENGTH;i++)
        {
            //std::cout<<"image_meta["<<i<<"] is "<<image_meta[i]<<std::endl;
            inputMetadataTensorMap(j,i)=image_meta[i];

        }
    }

}

void detectBatch::get_anchors()
{
    //tensorflow::Tensor flatTensor(tensorflow::DT_FLOAT,{1,2,4,5});
    //auto f=flatTensor.shape();

    //compute_backbone_shape
    //Cache anchors and reuse if image shape is the same
    //计算图像backbone_shapes
    if (inputImg_h!=_anchor_cache[0] || inputImg_w!=_anchor_cache[1] )//如果之前计算过就不用重新计算了,相当于cache
    {
        for(int i=0;i<backbone_strides_num;i++)
        {
            backbone_shape[i][0]=ceil(inputImg_h/BACKBONE_STRIDES[i]);
            backbone_shape[i][1]=ceil(inputImg_w/BACKBONE_STRIDES[i]);
        }
        //std::vector<tensorflow::Tensor> anchors;
        std::vector<Eigen::MatrixXf> boxesVec;//eigen矩阵类型容器,用于存储anchors
        //boxesVec作用类似于python代码中的
        /*
           for i in range(len(scales)):
                anchors.append(generate_anchors(scales[i], ratios, feature_shapes[i],
                                        feature_strides[i], anchor_stride))
            return np.concatenate(anchors, axis=0)



         */


        int finalBoxesRows=0;//用于统计五个RPN_ANCHOR_SCALES尺度对应的所有boxes的行数,可以先不看这个

        //generate_pyramid_anchors //生成不同尺度(配置参数中是5个)的anchor
        for(int j=0;j<rpn_anchor_scales_num;j++)
        {
            //generate_anchors

            //Get all combinations of scales and ratios
            Eigen::RowVectorXf scalesVec(1);//遍历并且临时存储RPN_ANCHOR_SCALES[5]={32, 64, 128, 256, 512}的每个元素,主要给scalesMat赋值用
            Eigen::VectorXf ratiosVec(rpn_anchor_ratios_num);
            Eigen::MatrixXf scalesMat=Eigen::MatrixXf(rpn_anchor_ratios_num, 1);//();
            Eigen::MatrixXf ratiosMat=Eigen::MatrixXf(rpn_anchor_ratios_num, 1);//();
            Eigen::MatrixXf heightsMat;//=Eigen::MatrixXf(rpn_anchor_ratios_num, 1);//();
            Eigen::MatrixXf widthsMat;//=Eigen::MatrixXf(rpn_anchor_ratios_num, 1);//();


            //以下步骤主要是实现python中的
            /*
             scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
             */

            scalesVec(0)=(RPN_ANCHOR_SCALES[j]);

            //构造np.array(ratios)
            for(int i=0;i<rpn_anchor_ratios_num;i++)
            {
                ratiosVec(i)=RPN_ANCHOR_RATIOS[i];
            }
            for(int i=0;i<ratiosMat.cols();i++)
            {
                ratiosMat.col(i)<<ratiosVec;
            }


            //构造np.array(scales)
            //std::cout<<"scalesMat is <<"<<scalesMat.cols()<<std::endl;
            for(int i=0;i<scalesMat.rows();i++)
            {
                scalesMat.row(i)<<scalesVec;

            }


            //构造heights,widths,这两个在python里面是长度为3的向量,但为了后面的点乘等操作换成了3*1的矩阵
            //python代码如下
            /*
                heights = scales / np.sqrt(ratios)
                widths = scales * np.sqrt(ratios)
             */

            //Enumerate heights and widths from scales and ratios
            heightsMat=scalesMat.cwiseQuotient(ratiosMat.cwiseSqrt());
            widthsMat=scalesMat.cwiseProduct(ratiosMat.cwiseSqrt());

            //构造shifts_x, shifts_y
            //python代码如下
            /*
            shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
            shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
            shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)
             */
            //Enumerate shifts in feature space
            //先进行   shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
            //        shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride

            int step=RPN_ANCHOR_STRIDE,low=0,hight_y=backbone_shape[j][0],hight_x=backbone_shape[j][1];//获取shape[0],shape[1],anchor_stride,
            Eigen::RowVectorXf shifts_y;//行向量
            Eigen::RowVectorXf shifts_x;
            int realsize_y=((hight_y-low)/step);
            int realsize_x=((hight_x-low)/step);
            shifts_y.setLinSpaced(realsize_y,low,low+step*(realsize_y-1));
            shifts_x.setLinSpaced(realsize_x,low,low+step*(realsize_x-1));
            shifts_y*=BACKBONE_STRIDES[j];//获取feature_stride,这里的feature_stride其实是python代码中外围循环送进的参数BACKBONE_STRIDES[j]
            shifts_x*=BACKBONE_STRIDES[j];//获取feature_stride,这里的feature_stride其实是python代码中外围循环送进的参数BACKBONE_STRIDES[j]

            /*再进行   shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y),
            构造出最终的shifts_x,shifts_y矩阵,注意经过np.meshgrid后shifts_x,shifts_y是二维的矩阵
            */
            //构造shifts_x,shifts_y矩阵
            Eigen::MatrixXf shifts_xMat(shifts_y.cols(),shifts_x.cols())
                    ,shifts_yMat(shifts_y.cols(),shifts_x.cols());
            for(int i=0;i<shifts_xMat.rows();i++)
            {
                shifts_xMat.row(i)=shifts_x;

            }
            for(int i=0;i<shifts_yMat.cols();i++)
            {
                shifts_yMat.col(i)=shifts_y;
            }





            //进行python代码
            /*
                box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
                box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

                # Reshape to get a list of (y, x) and a list of (h, w)
                box_centers = np.stack([box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
                box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

                # Convert to corner coordinates (y1, x1, y2, x2)
                boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)
                return boxes

             */
            //Enumerate combinations of shifts, widths, and heights
            //先进行 box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
            //      box_heights, box_centers_y = np.meshgrid(heights, shifts_y)
            //先把heightsMat,widthsMat换成行向量方便赋值,
            Eigen::RowVectorXf heightsMatFlat(Eigen::Map<Eigen::VectorXf>(heightsMat.data(),heightsMat.rows()*heightsMat.cols()));
            Eigen::RowVectorXf widthsMatFlat(Eigen::Map<Eigen::VectorXf>(widthsMat.data(),widthsMat.rows()*widthsMat.cols()));

            /*因为上面的np.meshgrid(widths, shifts_x)
            中widths是长度为3的向量,shifts_x是二维矩阵,所以np.meshgrid(widths, shifts_x)生成的矩阵列数是widths的长度
            生成的矩阵行数是--shifts_x按照行方向平铺后的长度,假如shifts_x是2*3矩阵,那么就是6.而后面
            box_widths, box_centers_x = np.meshgrid(widths, shifts_x)生成的box_centers_x的行数是shifts_x的行数*列数,box_centers_x每一列是shifts_x矩阵的元素按照行方向平铺后构成的,
            但是因为eigen里面的矩阵是列优先存储,所以要在c++代码中对shifts_xMat(shift_x)进行转置,这样通过Eigen::Map映射到shifts_yMatFlat就是相当于把shifts_x矩阵的元素按照行方向平铺后构成的向量
            同理对shifts_yMat进行同样的操作得到shifts_yMatFlat.
            而box_widths,box_heights可以通过widthsMatFlat,和heightsMatFlat赋值得到,因为heightsMatFlat和box_heights可以通过widthsMatFlat
            本身是一维的向量
            */
            shifts_xMat.transposeInPlace();
            shifts_yMat.transposeInPlace();
            Eigen::RowVectorXf shifts_yMatFlat(Eigen::Map<Eigen::VectorXf>(shifts_yMat.data(),shifts_yMat.rows()*shifts_yMat.cols()));
            //Eigen::RowVectorXf shifts_xMatFlat(Eigen::Map<Eigen::VectorXf>(shifts_xMat.data(),shifts_xMat.rows()*shifts_xMat.cols(),Eigen::ColMajor));
            Eigen::RowVectorXf shifts_xMatFlat(Eigen::Map<Eigen::VectorXf>(shifts_xMat.data(),shifts_xMat.rows()*shifts_xMat.cols()));
            Eigen::MatrixXf box_widthsMat=Eigen::MatrixXf(shifts_xMatFlat.cols(),widthsMatFlat.cols());//();
            Eigen::MatrixXf box_center_xMat=Eigen::MatrixXf(shifts_xMatFlat.cols(),widthsMatFlat.cols());//();
            Eigen::MatrixXf box_heightsMat=Eigen::MatrixXf(shifts_yMatFlat.cols(),heightsMatFlat.cols());//();
            Eigen::MatrixXf box_center_yMat=Eigen::MatrixXf(shifts_yMatFlat.cols(),heightsMatFlat.cols());//();
            for(int i=0;i<box_widthsMat.rows();i++)
            {
                box_widthsMat.row(i)=widthsMatFlat;
                box_heightsMat.row(i)=heightsMatFlat;
            }
            for(int i=0;i<box_heightsMat.cols();i++)
            {
                box_center_xMat.col(i)=shifts_xMatFlat;
                box_center_yMat.col(i)=shifts_yMatFlat;
            }


            //Convert to corner coordinates (y1, x1, y2, x2)
            // 'e for 's element abbreviation
            //note that ,in the bellow,matrix's element which to be add or substract, is In the corresponding position
            //python method: box_centers_y mat ,box_centers_x mat  stack to  mat A whose unit format is (box_center_y'e,box_center_x'e)
            //then reshape to [-1,2],so the result is mat whose  col format is (box_center_y'e,box_center_x'e),box_sizes mat B is the same,col format is (box_height'e,box_width'e)
            //then  A-B ,A+B get the mat C,D whose col format are  respectively  (box_center_y'e-box_height'e,box_center_x'e-box_width'e) and (box_center_y'e+box_height'e,box_center_x'e+box_width'e)
            //then concat C and D get mat E whose col format is (box_center_y'e-box_height'e,box_center_x'e-box_width'e ,box_center_y'e+box_height'e,box_center_x'e+box_width'e)
            //and that is (y1,x1,y2,x2)
            //in eigen3,different to python
            //first we have got the matrix box_center_yMat box_center_xMat box_heightsMat box_widthsMat
            //for abbreviation is center_yMat,center_xMat,heightMat,widthMat
            //center_yMat-0.5*heightMat=y1Mat
            //center_yMat+0.5*heightMat=y2Mat
            //center_xMat-0.5*widthMat=x1Mat
            //center_xMat+0.5*widthMat=x2Mat
            //then generate the matrix boxes whose col format is (y1Mat's e,x1Mat's e,y2Mat's e ,x2Mat's e),rows in the num

            //进行如下操作
            //boxes = np.concatenate([box_centers - 0.5 * box_sizes,
            //box_centers + 0.5 * box_sizes], axis=1)
            //boxes形式如[(y1, x1, y2, x2),...,...]
            Eigen::MatrixXf y1Mat=box_center_yMat-box_heightsMat*0.5;
            Eigen::MatrixXf x1Mat=box_center_xMat-box_widthsMat*0.5;
            Eigen::MatrixXf y2Mat=box_center_yMat+box_heightsMat*0.5;
            Eigen::MatrixXf x2Mat=box_center_xMat+box_widthsMat*0.5;
            y1Mat.transposeInPlace();
            x1Mat.transposeInPlace();
            y2Mat.transposeInPlace();
            x2Mat.transposeInPlace();
            Eigen::RowVectorXf y1MatFlat(Eigen::Map<Eigen::VectorXf>(y1Mat.data(),y1Mat.rows()*y1Mat.cols()));
            Eigen::RowVectorXf x1MatFlat(Eigen::Map<Eigen::VectorXf>(x1Mat.data(),x1Mat.rows()*x1Mat.cols()));
            Eigen::RowVectorXf y2MatFlat(Eigen::Map<Eigen::VectorXf>(y2Mat.data(),y2Mat.rows()*y2Mat.cols()));
            Eigen::RowVectorXf x2MatFlat(Eigen::Map<Eigen::VectorXf>(x2Mat.data(),x2Mat.rows()*x2Mat.cols()));
            Eigen::MatrixXf boxes(y1Mat.rows()*y1Mat.cols(),4);//注意这里的boxes不是python代码里面对应的boxes
            boxes.col(0)=y1MatFlat;
            boxes.col(1)=x1MatFlat;
            boxes.col(2)=y2MatFlat;
            boxes.col(3)=x2MatFlat;
            //到此已经完成单独一个RPN_ANCHOR_SCALES[i]尺度对应的boxes了
            //下一步把它放进容器里

            boxesVec.push_back(boxes);
            finalBoxesRows+=boxes.rows();//统计五个RPN_ANCHOR_SCALES尺度对应的所有boxes的行数
            //break;
        }
        //以上一步得到的boxes的finalBoxesRows为行数,4为列数创建二维矩阵finalBox(对应python代码的boxes),
        //其实就是用上面所有的boxes构建形式如[(y1, x1, y2, x2),...,...]的矩阵
        finalBox=Eigen::MatrixXf (finalBoxesRows,4);
        //Eigen::VectorXf a(3);
        //Eigen::VectorXf b(4);
        //Eigen::VectorXf c(7);
        //取出boxesVec容器里面每个boxes构建最终的finalBox矩阵(对应boxes)
        //至此完成了boxes的构建
        int beginX=0;
        for(int i=0;i<boxesVec.size();i++)
        {
             //mat1.block<rows,cols>(i,j)
            //矩阵块赋值
            finalBox.block(beginX,0,boxesVec[i].rows(),boxesVec[i].cols())=boxesVec[i];
            beginX+=boxesVec[i].rows();
            //tensorflow::Tensor matTensor(tensorflow::DT_FLOAT,{boxesVec[i].rows(),boxesVec[i].cols()});
        }


        /*get normalization finalbox
        归一化finalBox
        python代码如下:
        scale = np.array([h - 1, w - 1, h - 1, w - 1])
        shift = np.array([0, 0, 1, 1])
        return np.divide((boxes - shift), scale).astype(np.float32)
        */

        //先创建scale,shift两个向量
        Eigen::MatrixXf scaleMat_1r(1,finalBox.cols());
        Eigen::MatrixXf shiftMat_1r(1,finalBox.cols());
        scaleMat_1r<<float(inputImg_h-1),float(inputImg_w-1),float(inputImg_h-1),float(inputImg_w-1);
        shiftMat_1r<<0.f,0.f,1.f,1.f;
        //因为上一步得到是scaleMat_1r,shiftMat_1r是向量,接下来创建对应的矩阵,该矩阵与finalBox有相同的
        //形状
        Eigen::MatrixXf scaleMat=scaleMat_1r.colwise().replicate(finalBox.rows());//通过重复与finalBox同样的行数构建scaleMat
        Eigen::MatrixXf shiftMat=shiftMat_1r.colwise().replicate(finalBox.rows());//同上
        Eigen::MatrixXf tmpMat=finalBox-shiftMat;//finalBox对应位置元素减去偏移量
        finalBox_norm=tmpMat.cwiseQuotient(scaleMat);//finalBox对应位置元素处以scale
        //至此完成了python代码中的boxes(finalBox_norm),下一步把finalBox_norm矩阵弄成Eigen::tensor类型的inputAnchorsTensor_temp
        //再通过inputAnchorsTensor_temp填充到tensorflow::tensor类型的inputAnchorsTensor构建最后送入模型的anchor boxes


        inputAnchorsTensor=tensorflow::Tensor(tensorflow::DT_FLOAT,{batch_size,finalBox_norm.rows(),finalBox_norm.cols()});//初始化inputAnchorsTensor
        //float *p=inputAnchorsTensor.flat<float>().data();
        //通finalBox_norm矩阵构建Eigen::tensor类型的inputAnchorsTensor_temp
        Eigen::Tensor<float,3>inputAnchorsTensor_temp(1,finalBox_norm.rows(),finalBox_norm.cols());
        for(int i=0;i<finalBox_norm.rows();i++){

            Eigen::Tensor<float,1>eachrow(finalBox_norm.cols());//用于临时存储finalBox_norm矩阵的的每一行
            //把finalBox_norm矩阵的一行放进eachrow
            eachrow.setValues({finalBox_norm.row(i)[0],finalBox_norm.row(i)[1],finalBox_norm.row(i)[2],finalBox_norm.row(i)[3]});
            //把eachrow放进inputAnchorsTensor_temp的每一行
            inputAnchorsTensor_temp.chip(i,1)=eachrow;
        }
        //把inputAnchorsTensor_temp赋值给inputAnchorsTensor,注意它们两个的类型是不同的
        auto showMap=inputAnchorsTensor.tensor<float,3>();
        for(int b=0;b<showMap.dimension(0);b++)
        {
            for(int r=0;r<showMap.dimension(1);r++)
            {
                for(int c=0;c<showMap.dimension(2);c++)
                {

                    showMap(b,r,c)=inputAnchorsTensor_temp(0,r,c);//这里为0是因为
                    //我的batch里面的图片都是同样尺寸的,所以它们最终的anchor boxes都是一样,
                    //只要赋值一个就行了,建议batch里面图片尺寸都是一样的,这样好处理
                }
            }
        }


    }

}


void detectBatch::unmold_detections(std::vector<tensorflow::Tensor>&output_tensors,std::vector<imageDetectInfo> &output_vec)
{

    tensorflow::Tensor &detections_tensor=output_tensors[0];
    auto  boxes_tensor=detections_tensor.tensor<float,3>();
    //Extract boxes, class_ids, scores, and class-specific masks
    //whose classId in not 0 ,because 0 is background
    //std::cout<<"resized_tensor is "<<resized_tensor.shape()<<std::endl;
    //std::cout<<"inputAnchorsTensor is "<<inputAnchorsTensor.shape()<<std::endl;
    //std::cout<<"inputMetadataTensor is "<<inputMetadataTensor.shape()<<std::endl;
    //std::cout<<"detections_tensor is "<<detections_tensor.shape()<<std::endl;
    for(int imgNum=0;imgNum<boxes_tensor.dimension(0);imgNum++)
    {
        std::vector<Eigen::RowVectorXf> noZeroRow;

        //struct imageDetectInfo imageDetectInfotmp ;
        for(int boxNum=0;boxNum<boxes_tensor.dimension(1);boxNum++)
        {
           if (boxes_tensor(imgNum,boxNum,4)>0)
           {
               Eigen::RowVectorXf eachrow(boxes_tensor.dimension(2));
               eachrow<<boxes_tensor(imgNum,boxNum,0),
                       boxes_tensor(imgNum,boxNum,1),
                       boxes_tensor(imgNum,boxNum,2),
                       boxes_tensor(imgNum,boxNum,3),
                       boxes_tensor(imgNum,boxNum,4),
                       boxes_tensor(imgNum,boxNum,5);
                noZeroRow.push_back(eachrow);

           }

        }

        Eigen::MatrixXf noZeroMat (noZeroRow.size(),6);
        for(int r=0;r<noZeroRow.size();r++)
        {
            noZeroMat.row(r)=noZeroRow[r];

        }
        Eigen::MatrixXf boxMat(noZeroMat.rows(),4);
        Eigen::MatrixXf classSoresMat(noZeroMat.rows(),2);
        boxMat.block(0,0,boxMat.rows(),4)=noZeroMat.block(0,0,noZeroMat.rows(),4);
        classSoresMat.block(0,0,classSoresMat.rows(),2)=noZeroMat.block(0,4,classSoresMat.rows(),2);
        //std::cout<<"noZeroMat "<<noZeroMat<<std::endl;
        //std::cout<<"boxMat "<<boxMat<<std::endl;

        //get the window in image meta
        auto metaTensor=inputMetadataTensor.tensor<float,2>();
        Eigen::MatrixXf windowMat(1,4);
        Eigen::MatrixXf scale_rMat(1,4);
        windowMat<<metaTensor(0,7),metaTensor(0,8),
                metaTensor(0,7),metaTensor(0,8);
        scale_rMat<<metaTensor(0,9)-metaTensor(0,7),
                    metaTensor(0,10)-metaTensor(0,8),
                    metaTensor(0,9)-metaTensor(0,7),
                    metaTensor(0,10)-metaTensor(0,8);
        //get shiftmat
        //boxMat=tmpMat.cwiseQuotient(scaleMat);//that is unnecessary
        //because in my case ,shiftmat is [0,0,0,0],scale is [1,1,1,1]


        //denorm_boxes
        Eigen::MatrixXf shiftNorm_rMat(1,4);
        Eigen::MatrixXf scaleNorm_rMat(1,4);
        shiftNorm_rMat<<0,0,1,1;
        scaleNorm_rMat<<metaTensor(0,1)-1,
                    metaTensor(0,2)-1,
                    metaTensor(0,1)-1,
                    metaTensor(0,2)-1;
        Eigen::MatrixXf shiftNormMat=shiftNorm_rMat.colwise().replicate(boxMat.rows());
        Eigen::MatrixXf scaleNormMat=scaleNorm_rMat.colwise().replicate(boxMat.rows());
        boxMat=boxMat.cwiseProduct(scaleNormMat);
        boxMat=boxMat+shiftNormMat;
        finalboxMat=boxMat;
        //std::cout<<"final box mat is "<<finalboxMat<<std::endl;
        struct imageDetectInfo imageDetectInfoTmp;
        for(int i=0;i<finalboxMat.rows();i++)
        {
            struct boxInfo boxInfoTmp;
            boxInfoTmp.y1=(int)(finalboxMat(i,0));
            boxInfoTmp.x1=(int)(finalboxMat(i,1));
            boxInfoTmp.y2=(int)(finalboxMat(i,2));
            boxInfoTmp.x2=(int)(finalboxMat(i,3));
            boxInfoTmp.classId=(int)(classSoresMat(i,0));
            boxInfoTmp.scores=classSoresMat(i,1);
            boxInfoTmp.boxNum=i;
            imageDetectInfoTmp.detectInfo.push_back(boxInfoTmp);
        }
        imageDetectInfoTmp.imageNum=imgNum;
        output_vec[imgNum]=imageDetectInfoTmp;

        //outputsInfo.push_back(imageDetectInfoTmp);

    }
}
