#include "CLI/App.hpp"
#include "CLI/Config.hpp"
#include "CLI/Formatter.hpp"
#include "fmt/core.h"
#include "mat.h"
#include "matrix.pb.h"
#include "yolo-fastestv2.h"
#include <fstream>
#include <opencv2/core/hal/interface.h>

void resize(const cv::Mat &mat) {
  const int inputWidth = 352;
  const int inputHeight = 352;
  auto input = ncnn::Mat::from_pixels_resize(mat.data, ncnn::Mat::PIXEL_BGR2RGB,
                                             mat.cols, mat.rows, inputWidth,
                                             inputHeight);
  assert(input.c == 3);
  // so the type of ncnn input would be float (32FC3)?
  const float mean_vals[3] = {0.f, 0.f, 0.f};
  const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
  // input.substract_mean_normalize(mean_vals, norm_vals);
  fmt::println("input.w:{}, h:{}, c:{}, cstep:{}, elsize:{}", input.w, input.h,
               input.c, input.cstep, input.elemsize);
  auto chn = input.channel(0);
  fmt::println("chn0 len:{}, w:{} h:{} c:{} cstep:{}", chn.total(), chn.w,
               chn.h, chn.c, chn.cstep);
  auto out = std::ofstream("chn0.bin", std::ios::binary);
  Mat out_mat;
  out_mat.set_height(chn.h);
  out_mat.set_width(chn.w);
  for (auto it = static_cast<float *>(input.data);
       it != static_cast<float *>(input.data) + input.total(); ++it) {
    out_mat.add_data(*it);
  }
  out_mat.SerializeToOstream(&out);
  auto cv_mat = cv::Mat(input.h, input.w, CV_32FC3, input.data);
  cv::imwrite("resized.png", cv_mat);
}

void dump_ncnn_mat(const ncnn::Mat &mat) {}

int main() {
  static const char *class_names[] = {
      "person",        "bicycle",      "car",
      "motorcycle",    "airplane",     "bus",
      "train",         "truck",        "boat",
      "traffic light", "fire hydrant", "stop sign",
      "parking meter", "bench",        "bird",
      "cat",           "dog",          "horse",
      "sheep",         "cow",          "elephant",
      "bear",          "zebra",        "giraffe",
      "backpack",      "umbrella",     "handbag",
      "tie",           "suitcase",     "frisbee",
      "skis",          "snowboard",    "sports ball",
      "kite",          "baseball bat", "baseball glove",
      "skateboard",    "surfboard",    "tennis racket",
      "bottle",        "wine glass",   "cup",
      "fork",          "knife",        "spoon",
      "bowl",          "banana",       "apple",
      "sandwich",      "orange",       "broccoli",
      "carrot",        "hot dog",      "pizza",
      "donut",         "cake",         "chair",
      "couch",         "potted plant", "bed",
      "dining table",  "toilet",       "tv",
      "laptop",        "mouse",        "remote",
      "keyboard",      "cell phone",   "microwave",
      "oven",          "toaster",      "sink",
      "refrigerator",  "book",         "clock",
      "vase",          "scissors",     "teddy bear",
      "hair drier",    "toothbrush"};
  CLI::App app{"YoloFastest"};
  std::string param_path = "./model/yolo-fastestv2-opt.param";
  std::string bin_path = "./model/yolo-fastestv2-opt.bin";
  std::string input_path = "";
  std::string output_path = "output.png";
  app.add_option("-p, --param", param_path)->check(CLI::ExistingFile);
  app.add_option("-b, --bin", bin_path)->check(CLI::ExistingFile);
  app.add_option("-i, --input", input_path)
      ->required()
      ->check(CLI::ExistingFile);
  app.add_option("-o, --output", output_path);
  try {
    app.parse();
  } catch (const CLI::ParseError &e) {
    return app.exit(e);
  }

  yoloFastestv2 api;

  api.loadModel(param_path.c_str(), bin_path.c_str());

  cv::Mat cvImg = cv::imread(input_path);
  resize(cvImg);

  // std::vector<TargetBox> boxes;
  // api.detection(cvImg, boxes);

  // for (int i = 0; i < boxes.size(); i++) {
  //     std::cout<<boxes[i].x1<<" "<<boxes[i].y1<<" "<<boxes[i].x2<<"
  //     "<<boxes[i].y2
  //              <<" "<<boxes[i].score<<" "<<boxes[i].cate<<std::endl;

  //     char text[256];
  //     sprintf(text, "%s %.1f%%", class_names[boxes[i].cate], boxes[i].score *
  //     100);

  //     int baseLine = 0;
  //     cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX,
  //     0.5, 1, &baseLine);

  //     int x = boxes[i].x1;
  //     int y = boxes[i].y1 - label_size.height - baseLine;
  //     if (y < 0)
  //         y = 0;
  //     if (x + label_size.width > cvImg.cols)
  //         x = cvImg.cols - label_size.width;

  //     cv::rectangle(cvImg, cv::Rect(cv::Point(x, y),
  //     cv::Size(label_size.width, label_size.height + baseLine)),
  //                   cv::Scalar(255, 255, 255), -1);

  //     cv::putText(cvImg, text, cv::Point(x, y + label_size.height),
  //                 cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

  //     cv::rectangle (cvImg, cv::Point(boxes[i].x1, boxes[i].y1),
  //                    cv::Point(boxes[i].x2, boxes[i].y2), cv::Scalar(255,
  //                    255, 0), 2, 2, 0);
  // }

  // cv::imwrite(output_path, cvImg);

  return 0;
}
