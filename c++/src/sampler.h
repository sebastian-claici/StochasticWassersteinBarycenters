#ifndef SAMPLER_H
#define SAMPLER_H

#include <random>

#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "Eigen/Cholesky"
#include "Eigen/Eigenvalues"

using namespace Eigen;

template<typename V>
class Sampler {
 public:
  virtual V operator()() = 0;
  virtual V sample() = 0;
  virtual V mean() const = 0;
  virtual int dimension() const = 0;
};


template<typename V>
class Uniform : public Sampler<V> {
 public:
  Uniform(const V &a, const V &b):
    a_(a), b_(b) {
    if (a_.size() < b_.size())
      b_.resize(a_.size());
    else
      a_.resize(b_.size());

    for (int i = 0; i < a.size(); ++i) {
      distributions_.push_back(std::uniform_real_distribution<double>(a(i), b(i)));
    }
  }

  V operator()() {
    return sample();
  }

  V sample() {
    V samples(a_.size());
    for (int i = 0; i < samples.size(); ++i) {
      samples(i) = distributions_[i](generator_);
    }

    return samples;
  }

  V mean() const {
    return (a_ + b_) / 2;
  }

  int dimension() const {
    return a_.size();
  }

 private:
  V a_;
  V b_;
  std::vector<std::uniform_real_distribution<double>> distributions_;

  std::mt19937 generator_{ std::random_device{}() };
};


class Gaussian : public Sampler<VectorXd> {
 public:
  Gaussian(const VectorXd &mu, const MatrixXd &sigma):
    mu_(mu), sigma_(sigma) {
    SelfAdjointEigenSolver<MatrixXd> eigenSolver(sigma);
    transform_ = eigenSolver.eigenvectors() * eigenSolver.eigenvalues().cwiseSqrt().asDiagonal();
  }

  VectorXd operator()() {
    return sample();
  }

  VectorXd sample() {
    return mu_ + transform_ * VectorXd{ mu_.size() }.unaryExpr([&](auto x) {
        return dist(generator_);
      });
  }

  VectorXd mean() const {
    return mu_;
  }

  MatrixXd covariance() const {
    return sigma_;
  }

  int dimension() const {
    return mu_.size();
  }

 private:
  VectorXd mu_;
  MatrixXd sigma_;
  MatrixXd transform_;

  std::mt19937 generator_{ std::random_device{}() };
  std::normal_distribution<> dist;
};


class GaussianMixture : public Sampler<VectorXd> {
 public:
  GaussianMixture(const std::vector<VectorXd>& mus,
                  const std::vector<MatrixXd>& sigmas,
                  const std::vector<double>& ws) : mixture_(ws.begin(), ws.end()) {
    for (std::size_t i = 0; i < mus.size(); ++i)
      gaussians_.push_back(Gaussian(mus[i], sigmas[i]));
  }

  GaussianMixture(const std::vector<Gaussian>& gaussians):
    gaussians_(gaussians) {}

  VectorXd operator()() {
    return sample();
  }

  VectorXd sample() {
    auto x = gaussians_.back().sample();
    x.setZero();

    int i = mixture_(generator_);
    x = gaussians_[i].sample();

    return x;
  }

  VectorXd mean() const {
    auto x = gaussians_.back().mean();
    x.setZero();

    int i = 0;
    double w_total = 0.0;
    for (auto g : gaussians_) {
      w_total += ws_[i];
      x += ws_[i++] * g.mean();
    }

    return x / w_total;
  }

  int dimension() const {
    return gaussians_.back().dimension();
  }

 private:
  std::vector<double> ws_;
  std::vector<Gaussian> gaussians_;

  std::discrete_distribution<int> mixture_;
  std::mt19937 generator_{ std::random_device{}() };
};


template<typename V>
class UniformLine : public Sampler<V> {
 public:
  UniformLine(const V &p, const V &q):
    p_(p), q_(q) {
  }

  V operator()() {
    return sample();
  }

  V sample() {
    double t = dist_(generator_);

    return p_ + t * (q_ - p_);
  }

  V mean() const {
    return (p_ + q_) / 2;
  }

  int dimension() const {
    return p_.size();
  }

 private:
  V p_;
  V q_;
  std::uniform_real_distribution<double> dist_ {};

  std::mt19937 generator_{ std::random_device{}() };
};

template<typename V>
class UniformEllipse : public Sampler<V> {
 public:
 UniformEllipse(const V &p, const V &q, double e):
  q_(q), p_(p), e_(e) {
    c_ = (p_ + q_) / 2;
    precompute();
  }

  V operator()() {
    return sample();
  }

  V sample() {
    double t = dist_(generator_);
    int i = choose_(generator_);
    int j = (i + 1) % static_cast<int>(points_.size());

    auto P = points_[i];
    auto Q = points_[j];

    return P + t * (Q - P);
  }

  V mean() const {
    return c_;
  }

  int dimension() const {
    return c_.size();
  }

 private:
  void precompute() {
    int N = 10000;
    double t = 0.0;
    double h = 2 * M_PI / N;

    double axis_norm = (p_ - q_).squaredNorm();
    double a = 0.5 * std::sqrt(axis_norm);
    double b = a * std::sqrt(1 - e_ * e_);

    for (int i = 0; i <= N; ++i) {
      double alpha = t + h * i;
      double x = a * std::cos(alpha);
      double y = b * std::sin(alpha);
      double w = std::atan2(q_(2) - p_(2), q_(1) - p_(1));

      x = (p_(1) + q_(1)) / 2 + x * std::cos(w) - y * std::sin(w);
      y = (p_(2) + q_(2)) / 2 + x * std::sin(w) + y * std::cos(w);

      VectorXd P(2); P << x, y;
      points_.push_back(P);
    }

    std::vector<double> weights;
    for (int i = 0; i < N; ++i) {
      double norm = std::sqrt((points_[i + 1] - points_[i]).squaredNorm());
      weights.push_back(norm);
    }

    choose_ = std::discrete_distribution<int>(weights.begin(), weights.end());
  }

  V c_;
  V p_, q_;
  double e_;

  std::discrete_distribution<int> choose_;
  std::vector<VectorXd> points_;

  std::uniform_real_distribution<double> dist_ {};

  std::mt19937 generator_{ std::random_device{}() };
};

#endif
