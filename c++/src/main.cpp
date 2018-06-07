#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <vector>

#include "boost/program_options.hpp"

#include "image.h"
#include "optimizer.h"
#include "powercell.h"
#include "sampler.h"
#include "scene.h"
#include "site.h"

using namespace Eigen;
namespace po = boost::program_options;

const int DROPOFF = 200;
std::string folder;

void test_line_2d(int iters=100, int num=3) {
  VectorXd ag(2); VectorXd bg(2);
  ag << -3, -3; bg << 3, 3;

  std::srand((unsigned int) 10);
  std::vector<VectorXd> p, q;
  for (int i = 0; i < num; ++i) {
    p.push_back(3 * VectorXd::Random(2));
    q.push_back(3 * VectorXd::Random(2));
  }

  Sampler<VectorXd>* gamma = new Uniform<VectorXd>(ag, bg);
  auto scene_ptr = std::make_shared<Scene<Sampler<VectorXd>, VectorXd>>(gamma, 8 * 8);

  std::vector<std::shared_ptr<Site<Sampler<VectorXd>, VectorXd>>> sites;
  auto len = std::min(p.size(), q.size());
  for (int i = 0; i < len; ++i) {
    Sampler<VectorXd>* mu = new UniformLine<VectorXd>(p[i], q[i]);
    auto pwc = std::make_shared<Powercell<VectorXd>>();
    auto site = std::make_shared<Site<Sampler<VectorXd>, VectorXd>>(mu, pwc);

    sites.push_back(site);
  }

  std::ofstream fcomps("../results/line2d/components.txt", std::ofstream::trunc);
  for (int i = 0; i < num; ++i) {
    fcomps << p[i].transpose() << std::endl;
    fcomps << q[i].transpose() << std::endl;
  }

  Optimizer<Sampler<VectorXd>, VectorXd, MatrixXd> opt(scene_ptr);
  opt.set_parameters(0.0001, 0.999);
  for (int i = 0; i < iters; ++i) {
    std::cout << "Iteration " << i << std::endl << "========================="
              << std::endl << std::endl;
    auto pts = opt.run(sites, 10000);
    std::cout << std::endl;

    std::string filename = folder + std::to_string(i + 1) + ".txt";
    std::ofstream fout(filename, std::ofstream::trunc);
    for (auto pt : pts) {
      fout << pt.transpose() << std::endl;
    }
    fout << std::endl;
  }
}

void test_uniform(const std::vector<VectorXd>& a,
                  const std::vector<VectorXd>& b,
                  int iters=100) {
  std::vector<std::shared_ptr<Site<Sampler<VectorXd>, VectorXd>>> sites;

  double vol = 1.0;
  for (int i = 0; i < a[0].size(); ++i) {
    vol *= (b[0](i) - a[0](i));
  }
  Sampler<VectorXd>* gamma = new Uniform<VectorXd>(a[0], b[0]);
  auto scene_ptr = std::make_shared<Scene<Sampler<VectorXd>, VectorXd>>(gamma, vol);

  auto len = std::min(a.size(), b.size());
  for (std::size_t i = 1; i < len; ++i) {
    Sampler<VectorXd>* mu = new Uniform<VectorXd>(a[i], b[i]);
    auto pwc = std::make_shared<Powercell<VectorXd>>(DROPOFF);
    auto site = std::make_shared<Site<Sampler<VectorXd>, VectorXd>>(mu, pwc);

    sites.push_back(site);
  }

  Optimizer<Sampler<VectorXd>, VectorXd, MatrixXd> opt(scene_ptr);
  opt.set_parameters(0.001, 0.99);

  for (int i = 0; i < iters; ++i) {
    std::cout << "Iteration " << i << std::endl << "========================="
              << std::endl << std::endl;
    auto pts = opt.run(sites, 4000);
    std::cout << std::endl;

    std::string filename = folder + std::to_string(i + 1) + ".txt";
    std::ofstream fout(filename, std::ofstream::trunc);
    for (auto pt : pts) {
      fout << pt.transpose() << std::endl;
    }
    fout << std::endl;
  }
}

void test_uniform_2d(int iters=100) {
  std::cout << "Testing with two uniform distributions (2D)." << std::endl;
  std::cout << "=======================================" << std::endl;

  VectorXd ag(2); VectorXd bg(2);
  ag << -4, -4; bg << 4, 4;
  VectorXd a1(2); VectorXd b1(2);
  a1 << 2, 2; b1 << 3, 3;
  VectorXd a2(2); VectorXd b2(2);
  a2 << -3, -3; b2 << -2, -2;

  std::vector<VectorXd> a{ag, a1, a2};
  std::vector<VectorXd> b{bg, b1, b2};

  test_uniform(a, b, iters);
}

void test_uniform_3d(int iters=100, int num=2) {
  std::cout << "Testing with three uniform distributions (3D)." << std::endl;
  std::cout << "==============================================" << std::endl;

  VectorXd ag(3); VectorXd bg(3);
  ag << -3, -3, -3; bg << 3, 3, 3;

  std::vector<VectorXd> as {ag};
  std::vector<VectorXd> bs {bg};
  VectorXd a1(3), b1(3);
  VectorXd a2(3), b2(3);
  a1 << 1, 1, 1; b1 << 2, 2, 2;
  a2 << -2, -2, -2; b2 << -1, -1, -1;
  // for (int i = 0; i < num; ++i) {
  //   VectorXd a = 3 * VectorXd::Random(3);
  //   VectorXd b = 3 * VectorXd::Random(3);
  //   for (int j = 0; j < 3; ++j) {
  //     if (a(j) > b(j))
  //       std::swap(a(j), b(j));

  //     auto diff = 1 - (b(j) - a(j));
  //     b(j) += diff;
  //   }
  //   as.push_back(a);
  //   bs.push_back(b);
  // }
  as.push_back(a1); as.push_back(a2);
  bs.push_back(b1); bs.push_back(b2);

  std::ofstream fcomps("../results/uniform3d/components.txt", std::ofstream::trunc);
  for (int i = 1; i <= num; ++i) {
    fcomps << as[i].transpose() << std::endl;
    fcomps << bs[i].transpose() << std::endl;
  }

  test_uniform(as, bs, iters);
}

void test_gaussian_mixture(int iters=100, int num=10) {
  std::cout << "Testing with a mixture of Gaussians (2D)." << std::endl;
  std::cout << "=========================================" << std::endl;

  std::vector<VectorXd> mus;
  std::vector<MatrixXd> sigmas;
  std::vector<double> ws;

  std::srand((unsigned int) 10);
  for (int i = 0; i < num; ++i) {
    VectorXd mu = 4 * VectorXd::Random(2);
    MatrixXd sigma = MatrixXd::Random(2, 2);
    sigma = sigma.adjoint() * sigma;

    double d1 = sigma(0, 0) / sigma(1, 1);
    double d2 = sigma(1, 1) / sigma(0, 0);
    if (d1 > 10)
      sigma(1, 1) += (sigma(0, 0) + sigma(1, 1)) / 2.0;
    else if (d2 > 10)
      sigma(0, 0) += (sigma(0, 0) + sigma(1, 1)) / 2.0;

    mus.push_back(mu);
    sigmas.push_back(sigma);
    ws.push_back(1.0);
  }

  std::ofstream fcomps("../results/mixture/components.txt", std::ofstream::trunc);
  for (int i = 0; i < num; ++i) {
    fcomps << mus[i].transpose() << std::endl;
    fcomps << sigmas[i].transpose() << std::endl;
  }

  Sampler<VectorXd>* mixture = new GaussianMixture(mus, sigmas, ws);

  auto scene_ptr = std::make_shared<Scene<Sampler<VectorXd>, VectorXd>>(mixture, 10 * 10);
  auto pwc = std::make_shared<Powercell<VectorXd>>();
  auto site = std::make_shared<Site<Sampler<VectorXd>, VectorXd>>(mixture, pwc);

  std::vector<std::shared_ptr<Site<Sampler<VectorXd>, VectorXd>>> sites {site};
  Optimizer<Sampler<VectorXd>, VectorXd, MatrixXd> opt(scene_ptr);
  opt.set_parameters();
  opt.GLOBAL_OPT = true;

  for (int i = 0; i < iters; ++i) {
    std::cout << "Iteration " << i + 1 << std::endl << "========================="
              << std::endl << std::endl;
    auto pts = opt.run(sites, 5000);
    std::cout << std::endl;

    std::string uniform = folder + std::to_string(i + 1) + "-uniform.txt";
    std::ofstream funi(uniform, std::ofstream::trunc);
    for (int j = 0; j <= i; ++j)
      funi << scene_ptr->sample() << std::endl;
    funi.close();

    std::string result = folder + std::to_string(i + 1) + "-res.txt";
    std::ofstream fres(result, std::ofstream::trunc);
    for (auto pt : pts)
      fres << pt.transpose() << std::endl;
    fres.close();
  }
}

void test_gaussians(int iters=100, int num=10) {
  std::cout << "Testing with several Gaussians (2D)." << std::endl;
  std::cout << "=========================================" << std::endl;

  std::vector<VectorXd> mus;
  std::vector<MatrixXd> sigmas;
  std::vector<double> ws;

  std::srand((unsigned int) 31);
  for (int i = 0; i < num; ++i) {
    VectorXd mu = VectorXd::Zero(2);
    MatrixXd sigma = MatrixXd::Random(2, 2);
    sigma = sigma.adjoint() * sigma;

    double d1 = sigma(0, 0) / sigma(1, 1);
    double d2 = sigma(1, 1) / sigma(0, 0);
    if (d1 > 10)
      sigma(1, 1) += (sigma(0, 0) + sigma(1, 1)) / 2.0;
    else if (d2 > 10)
      sigma(0, 0) += (sigma(0, 0) + sigma(1, 1)) / 2.0;

    mus.push_back(mu);
    sigmas.push_back(sigma);
    ws.push_back(1.0);
  }

  std::ofstream fcomps("../results/gaussian/components.txt", std::ofstream::trunc);
  for (int i = 0; i < num; ++i) {
    fcomps << mus[i].transpose() << std::endl;
    fcomps << sigmas[i].transpose() << std::endl;
  }

  VectorXd a(2); VectorXd b(2);
  a << -3, -3; b << 3, 3;
  double vol = 1.0;
  for (int i = 0; i < a.size(); ++i) {
    vol *= (b(i) - a(i));
  }
  Sampler<VectorXd>* gamma = new Uniform<VectorXd>(a, b);
  auto scene_ptr = std::make_shared<Scene<Sampler<VectorXd>, VectorXd>>(gamma, vol);

  std::vector<std::shared_ptr<Site<Sampler<VectorXd>, VectorXd>>> sites;
  for (int i = 0; i < num; ++i) {
    Sampler<VectorXd>* gaussian = new Gaussian(mus[i], sigmas[i]);
    auto pwc = std::make_shared<Powercell<VectorXd>>();
    auto site = std::make_shared<Site<Sampler<VectorXd>, VectorXd>>(gaussian, pwc);
    sites.push_back(site);
  }

  Optimizer<Sampler<VectorXd>, VectorXd, MatrixXd> opt(scene_ptr);
  opt.set_parameters(0.01, 0.99);
  opt.GLOBAL_OPT = true;

  for (int i = 0; i < iters; ++i) {
    std::cout << "Iteration " << i + 1 << std::endl << "========================="
              << std::endl << std::endl;
    auto pts = opt.run(sites, 1000);
    std::cout << std::endl;

    std::string uniform = folder + std::to_string(i + 1) + "-uniform.txt";
    std::ofstream funi(uniform, std::ofstream::trunc);
    for (int j = 0; j <= i; ++j)
      funi << scene_ptr->sample() << std::endl;
    funi.close();

    std::string result = folder + std::to_string(i + 1) + "-res.txt";
    std::ofstream fres(result, std::ofstream::trunc);
    for (auto pt : pts)
      fres << pt.transpose() << std::endl;
    fres.close();
  }
}

void test_images(const std::string& filename, int npts=5000) {
  cv::Mat img = cv::imread(filename, cv::IMREAD_GRAYSCALE);
  img = 255 - img;
  Sampler<VectorXd>* image = new Image(img);

  auto scene_ptr = std::make_shared<Scene<Sampler<VectorXd>, VectorXd>>(image);
  auto pwc_ptr = std::make_shared<Powercell<VectorXd>>();
  auto site_ptr = std::make_shared<Site<Sampler<VectorXd>, VectorXd>>(image, pwc_ptr);

  std::vector<std::shared_ptr<Site<Sampler<VectorXd>, VectorXd>>> sites{ site_ptr };
  Optimizer<Sampler<VectorXd>, VectorXd, MatrixXd> opt(scene_ptr);

  opt.set_parameters(0.001, 0.99);
  auto pts = opt.run_all(sites, 5000, npts);

  std::string result = folder + std::to_string(npts) + "-res.txt";
  std::ofstream fres(result, std::ofstream::trunc);
  for (auto pt : pts)
    fres << pt.transpose() << std::endl;
  fres.close();
}

int main(int argc, char *argv[])
{
  // Declare the supported options.
  po::options_description desc("Allowed options");
  desc.add_options()
    ("help", "produce help message")
    ("folder", po::value<std::string>(), "folder to save results in")
    ("uniform2d", "two 2D uniform distributions")
    ("uniform3d", po::value<int>(), "3D uniform distributions")
    ("mixture", po::value<int>(), "Gaussian mixture model")
    ("gaussian", po::value<int>(), "Gaussian distributions")
    ("line2d", po::value<int>(), "lines in 2D")
    ("image", po::value<std::string>(), "images")
    ;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << desc << "\n";
    return 1;
  }

  folder = vm["folder"].as<std::string>();
  if (vm.count("uniform2d")) {
    std::cout << "Testing uniform distribution in 2D." << std::endl;
    test_uniform_2d(100);
  }
  if (vm.count("uniform3d")) {
    std::cout << "Testing uniform distribution in 3D." << std::endl;
    int num = vm["uniform3d"].as<int>();
    test_uniform_3d(200, num);
  }
  if (vm.count("mixture")) {
    std::cout << "Testing mixture of Gaussians." << std::endl;
    int num = vm["mixture"].as<int>();
    test_gaussian_mixture(200, num);
  }
  if (vm.count("gaussian")) {
    std::cout << "Testing several Gaussians." << std::endl;
    int num = vm["gaussian"].as<int>();
    test_gaussians(200, num);
  }
  if (vm.count("line2d")) {
    std::cout << "Testing uniform over lines in 2D." << std::endl;
    int num = vm["line2d"].as<int>();
    test_line_2d(1000, num);
  }
  if (vm.count("image")) {
    std::cout << "Testing blue noise generation." << std::endl;
    std::string file = vm["image"].as<std::string>();
    test_images(file, 30000);
  }

  return 0;
}
