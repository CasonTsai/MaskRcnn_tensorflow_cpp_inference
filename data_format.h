#ifndef DATA_FORMAT_H
#define DATA_FORMAT_H
#include <vector>
#include <map>
#include <string>
#include "tensorflow/core/framework/tensor.h"
const size_t ColorTable[50]={0xBBFFFF,0x00868B,0x00FF00,0x008B45,
                  0xBCEE68,0xFFF68F,0x8B864E,0xFFFFE0,
                  0x8B8B7A,0xFFFF00,0x8B8B00,0x8B658B,
                  0xFFC1C1,0x8B6969,0xFF6A6A,0x8B3A3A,
                  0xFF8247,0xFF3030,0xFF69B4,0x0000FF,
                  0x00688B

                  };//颜色对照表 color table


struct boxForXml{
    int y1,x1,y2,x2;
    float scores=0.f;


};//boxes info for creating  xml file 
struct boxInfo{
    int y1,x1,y2,x2;
    int classId=0;
    float scores=0.f;
    int boxNum=-1;
};//boxes info ,the 'boxNum' 

struct imageDetectInfo{
    int imageWidth=0;//not yet
    int imageHeight=0;//not yet
    int imageNum=-1;
    std::vector<boxInfo> detectInfo;//save the detected boxes info

};//save detect result




#endif // DATA_FORMAT_H
