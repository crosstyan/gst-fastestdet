#ifndef NCNN_H_
#define NCNN_H_

#include "net.h"

#include <opencv2/opencv.hpp>
#include <vector>

class TargetBox {
private:
  float getWidth() { return (x2 - x1); };
  float getHeight() { return (y2 - y1); };

public:
  int x1;
  int y1;
  int x2;
  int y2;

  int cate;
  float score;

  float area() { return getWidth() * getHeight(); };
};

class yoloFastestv2 {
private:
  ncnn::Net net;
  std::vector<float> anchor;

  std::string inputName;
  std::string outputName1;
  std::string outputName2;

  int numAnchor;
  int numOutput;
  int numThreads;
  int numCategory;
  int inputWidth, inputHeight;

  float nmsThresh;

  int nmsHandle(std::vector<TargetBox> &tmpBoxes,
                std::vector<TargetBox> &dstBoxes);
  int getCategory(const float *values, int index, int &category, float &score);
  int predHandle(const ncnn::Mat *out, std::vector<TargetBox> &dstBoxes,
                 const float scaleW, const float scaleH, const float thresh);

public:
  yoloFastestv2();
  ~yoloFastestv2();

  int loadModel(const char *paramPath, const char *binPath);
  int detection(const cv::Mat srcImg, std::vector<TargetBox> &dstBoxes,
                      const float thresh = 0.3);
};
#endif