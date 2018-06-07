#include <opencv2/opencv.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>

#include <random>

#include "sampler.h"

class Image : public Sampler<VectorXd> {
 public:
  Image(const cv::Mat& img) : image(img), generator_() {
    scale = (double) std::min(img.rows, img.cols);

    build_density();
  }

  VectorXd operator()() {
    return sample();
  }

  VectorXd sample() {
    int id = dist_(generator_);
    double i = (id / image.cols) / scale;
    double j = (id % image.cols) / scale;

    VectorXd val(2); val << i, j;

    return val;
  }

  VectorXd mean() const {
    return VectorXd::Zero(2); // not implemented
  }

  int dimension() const {
    return 2;
  }

  double volume() const {
    return (image.rows * image.cols);
  }

 private:
  void build_density() {
    int N = image.rows * image.cols;
    int r = image.rows;
    int c = image.cols;

    std::vector<double> intensity(N, 0);
    for (int i = 0; i < r; ++i)
      for (int j = 0; j < c; ++j) {
        int id = i * c + j;
        intensity[id] = image.at<uchar>(i, j);
      }

    dist_ = std::discrete_distribution<int>(intensity.begin(), intensity.end());
  }

  cv::Mat image;
  double scale;

  std::default_random_engine generator_;
  std::discrete_distribution<int> dist_;
};
