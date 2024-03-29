#include "yolo-fastestv2.h"
#include "fmt/core.h"
#include <algorithm>
#include <math.h>
#include <numeric>
#include <opencv2/imgproc.hpp>

//模型的参数配置
yoloFastestv2::yoloFastestv2() {
  printf("Creat yoloFastestv2 Detector...\n");
  //输出节点数
  numOutput = 2;
  //推理线程数
  numThreads = 4;
  // anchor num
  numAnchor = 3;
  //类别数目
  numCategory = 80;
  // NMS阈值
  nmsThresh = 0.25;

  //模型输入尺寸大小
  inputWidth = 352;
  inputHeight = 352;

  //模型输入输出节点名称
  inputName = "input.1";
  outputName1 = "794"; // 22x22
  outputName2 = "796"; // 11x11

  //打印初始化相关信息
  printf("numThreads:%d\n", numThreads);
  printf("inputWidth:%d inputHeight:%d\n", inputWidth, inputHeight);

  // anchor box w h
  std::vector<float> bias{12.64,  19.39, 37.88,  51.48,  55.71,  138.31,
                          126.91, 78.23, 131.57, 214.55, 279.92, 258.87};

  anchor.assign(bias.begin(), bias.end());
}

yoloFastestv2::~yoloFastestv2() {
  printf("Destroy yoloFastestv2 Detector...\n");
}

// ncnn 模型加载
int yoloFastestv2::loadModel(const char *paramPath, const char *binPath) {
  printf("Ncnn mode init:\n%s\n%s\n", paramPath, binPath);

  net.load_param(paramPath);
  net.load_model(binPath);

  printf("Ncnn model init sucess...\n");

  return 0;
}

float intersection_area(const TargetBox &a, const TargetBox &b) {
  if (a.x1 > b.x2 || a.x2 < b.x1 || a.y1 > b.y2 || a.y2 < b.y1) {
    // no intersection
    return 0.f;
  }

  float inter_width = std::min(a.x2, b.x2) - std::max(a.x1, b.x1);
  float inter_height = std::min(a.y2, b.y2) - std::max(a.y1, b.y1);

  return inter_width * inter_height;
}

bool scoreSort(TargetBox a, TargetBox b) { return (a.score > b.score); }

// NMS处理
int yoloFastestv2::nmsHandle(std::vector<TargetBox> &tmpBoxes,
                             std::vector<TargetBox> &dstBoxes) {
  std::vector<int> picked;

  sort(tmpBoxes.begin(), tmpBoxes.end(), scoreSort);

  for (int i = 0; i < tmpBoxes.size(); i++) {
    int keep = 1;
    for (int j = 0; j < picked.size(); j++) {
      //交集
      float inter_area = intersection_area(tmpBoxes[i], tmpBoxes[picked[j]]);
      //并集
      float union_area =
          tmpBoxes[i].area() + tmpBoxes[picked[j]].area() - inter_area;
      float IoU = inter_area / union_area;

      if (IoU > nmsThresh && tmpBoxes[i].cate == tmpBoxes[picked[j]].cate) {
        keep = 0;
        break;
      }
    }

    if (keep) {
      picked.push_back(i);
    }
  }

  for (int i = 0; i < picked.size(); i++) {
    dstBoxes.push_back(tmpBoxes[picked[i]]);
  }

  return 0;
}

//检测类别分数处理
int yoloFastestv2::getCategory(const float *values, int index, int &category,
                               float &score) {
  float tmp = 0;
  float objScore = values[4 * numAnchor + index];

  for (int i = 0; i < numCategory; i++) {
    float clsScore = values[4 * numAnchor + numAnchor + i];
    clsScore *= objScore;

    if (clsScore > tmp) {
      score = clsScore;
      category = i;

      tmp = clsScore;
    }
  }

  return 0;
}

double mean(const float *begin, const float *end) {
  float sum = std::accumulate(begin, end, static_cast<double>(0.0f));
  return sum / (end - begin);
}

double varience(const float *begin, const float *end) {
  double m = mean(begin, end);
  double accum = 0.0f;
  std::for_each(begin, end, [&](const float d) { accum += (d - m) * (d - m); });
  return accum / (end - begin);
}

//特征图后处理
int yoloFastestv2::predHandle(const ncnn::Mat *out,
                              std::vector<TargetBox> &dstBoxes,
                              const float scaleW, const float scaleH,
                              const float thresh) { // do result
  for (int i = 0; i < numOutput; i++) {
    int stride;
    int outW, outH, outC;

    outH = out[i].c;
    outW = out[i].h;
    outC = out[i].w;

    assert(inputHeight / outH == inputWidth / outW);
    stride = inputHeight / outH;
    fmt::println("outH:{}, outW:{}, outC:{}\n stride:{}, scaleW:{}, scaleH:{}",
                 outH, outW, outC, stride, scaleW, scaleH);

    for (int h = 0; h < outH; h++) {
      auto output = out[i];
      auto begin = static_cast<float *>(output.channel(h).data);
      auto end = begin + output.channel(h).total();
      auto values = begin;
      if (h == 0) {
        fmt::println("total:{}\nmean:{}\nvariance:{}",
                     output.channel(h).total(), mean(begin, end),
                     varience(begin, end));
      }
      for (int w = 0; w < outW; w++) {
        for (int b = 0; b < numAnchor; b++) {
          // float objScore = values[4 * numAnchor + b];
          TargetBox tmpBox;
          int category = -1;
          float score = -1;

          getCategory(values, b, category, score);

          if (score > thresh) {
            float bcx, bcy, bw, bh;

            bcx = ((values[b * 4 + 0] * 2. - 0.5) + w) * stride;
            bcy = ((values[b * 4 + 1] * 2. - 0.5) + h) * stride;
            bw = pow((values[b * 4 + 2] * 2.), 2) *
                 anchor[(i * numAnchor * 2) + b * 2 + 0];
            bh = pow((values[b * 4 + 3] * 2.), 2) *
                 anchor[(i * numAnchor * 2) + b * 2 + 1];
            fmt::println("bcx:{}, bcy:{}, bw:{}, bh:{}, score:{}", bcx, bcy, bw, bh, score);
            tmpBox.x1 = (bcx - 0.5 * bw) * scaleW;
            tmpBox.x2 = (bcx + 0.5 * bw) * scaleW;
            tmpBox.y1 = (bcy - 0.5 * bh) * scaleH;
            tmpBox.y2 = (bcy + 0.5 * bh) * scaleH;
            tmpBox.score = score;
            tmpBox.cate = category;

            dstBoxes.push_back(tmpBox);
          }
        }
        values = values + outC;
      }
    }
  }
  return 0;
}

/// return the input mat for debugging
int yoloFastestv2::detection(const cv::Mat srcImg,
                             std::vector<TargetBox> &dstBoxes,
                             const float thresh) {
  dstBoxes.clear();

  float scaleW = (float)srcImg.cols / (float)inputWidth;
  float scaleH = (float)srcImg.rows / (float)inputHeight;
  auto stride = srcImg.step1();
  fmt::println("stride:{}", stride);
  auto rgbImg = cv::Mat();
  cv::cvtColor(srcImg, rgbImg, cv::COLOR_BGR2RGB);
  // resize of input image data
  fmt::println("Color Code {}", ncnn::Mat::PIXEL_RGB2BGR);
  ncnn::Mat inputImg = ncnn::Mat::from_pixels_resize(
      rgbImg.data, ncnn::Mat::PIXEL_RGB2BGR, srcImg.cols, srcImg.rows, stride,
      inputWidth, inputHeight);

  // Normalization of input image data
  const float mean_vals[3] = {0.f, 0.f, 0.f};
  const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
  inputImg.substract_mean_normalize(mean_vals, norm_vals);

  // creat extractor
  ncnn::Extractor ex = net.create_extractor();
  ex.set_num_threads(numThreads);
  // set input tensor
  ex.input(inputName.data(), inputImg);
  auto s = inputWidth * inputHeight * 3;
  const auto p = static_cast<float *>(inputImg.data);
  auto v = std::vector<float>();
  for (auto it = p; it != p + s; it++) {
    v.emplace_back(*p);
  }
  auto count = static_cast<float>(v.size());
  auto average = std::reduce(v.begin(), v.end()) / count;
  fmt::println("first {} el: {}", v.size(), average);

  // forward
  ncnn::Mat out[2];
  auto err = ex.extract(outputName1.data(), out[0]); // 22x22
  if (err != 0) {
    fmt::print("extract error:{}");
    return 1;
  }
  err = ex.extract(outputName2.data(), out[1]); // 11x11
  if (err != 0) {
    fmt::println("extract error:{}", err);
    return 1;
  }

  std::vector<TargetBox> tmpBoxes;
  //特征图后处理
  predHandle(out, tmpBoxes, scaleW, scaleH, thresh);

  // NMS
  nmsHandle(tmpBoxes, dstBoxes);
  for (auto box : dstBoxes) {
    fmt::println("\n\tx1:{},\n\ty1:{},\n\tx2:{},\n\ty2:{}, \n\tscore:{}, \n\tcate:{}\n",
                 box.x1, box.y1, box.x2, box.y2, box.score, box.cate);
  }

  return 0;
}