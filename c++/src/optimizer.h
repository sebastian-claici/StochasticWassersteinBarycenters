#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <cmath>

#include "powercell.h"
#include "scene.h"
#include "site.h"

template<typename S, typename V, typename M>
class Optimizer {
 public:
  Optimizer(std::shared_ptr<Scene<S, V>> scene) :
    scene_(scene) {}

  /**
   *
   * TODO: refactor this into constructor
   */
  void set_parameters(double alpha=0.02, double beta=0.99) {
    alpha_ = alpha;
    beta_ = beta;
  }

  V weight_gradient(std::shared_ptr<Site<S, V>> site) {
    auto n = site->pwc_sites();
    auto densities = site->pwc_densities();

    return -densities.array() + 1. / n;
  }

  V weight_fem_newton(std::shared_ptr<Site<S, V>> site, double h = 0.0001) {
    auto w = site->pwc_weights();
    auto grad = weight_gradient(site); // \nabla f(w)

    M H(w.size(), w.size());
    for (int i = 0; i < w.size(); ++i) {
      // change weights and compute new gradient
      auto w_copy = w;
      w_copy(i) += h;
      site->pwc_set_weights(w_copy);

      auto grad_h = weight_gradient(site);
      H.col(i) = (grad_h - grad) / h;
    }

    H = 0.5 * (H + H.adjoint());

    double beta = 1;
    double tau = H(0, 0);
    for (int i = 1; i < w.size(); ++i)
      tau = std::min(tau, H(i, i));
    tau = tau > 0 ? 0 : (-tau + beta);

    auto eye(H);
    eye.setIdentity();

    while (true) {
      Eigen::LLT<M> lltOfH(H);
      if (lltOfH.info() == Eigen::NumericalIssue) {
        H = H + tau * eye;
        tau = std::max(tau, 2 * beta);
      } else {
        return lltOfH.solve(grad);
      }
    }
  }

  bool weight_step(std::shared_ptr<Site<S, V>> site) {
    auto w_k = site->pwc_weights();
    auto grad = weight_gradient(site);
    if (grad.squaredNorm() < 1e-6)
      return false;

    auto d = weight_fem_newton(site);
    w_k = w_k + alpha_ * d;
    site->pwc_set_weights(w_k);

    return true;
  }

  bool weight_step_accelerated(std::shared_ptr<Site<S, V>> site, V& z_k) {
    auto w_k = site->pwc_weights();
    auto grad = weight_gradient(site);
    if (grad.squaredNorm() < 1e-6)
      return false;

    z_k = beta_ * z_k + grad;
    w_k = w_k + alpha_ * z_k;
    site->pwc_set_weights(w_k);

    return true;
  }

  double hot_start_weights(std::shared_ptr<Site<S, V>> site, int id) {
    // figure out if we need to increase or decrease the weight
    double w = 1e3;
    int n = site->pwc_sites();
    double wstart, wend;
    double sgn = 1;

    site->pwc_set_weight(w, id);
    auto density = site->pwc_densities();
    if (density(id) >= 1 - 1e-6) {
      wstart = -1e3;
      wend = 1e3;
    } else {
      wstart = 1e3;
      wend = -1e3;
      sgn = -1;
    }

    const double eps = 1e-4;
    while (sgn * wstart < sgn * wend) {
      auto wmid = (wstart + wend) / 2.0;

      site->pwc_set_weight(wmid, id);
      density = site->pwc_densities();

      if (density(id) < 0.1 - eps && density(id) > eps) {
        return wmid;
      } else if (density(id) >= 0.1 - eps) {
        wend = wmid;
      } else if (density(id) <= eps) {
        wstart = wmid;
      }
    }
  }

  void solve_weights(const std::vector<std::shared_ptr<Site<S, V>>>& sites, int max_iters) {
    for (auto site : sites) {
      auto z_k = V(site->pwc_weights());
      z_k.setZero();
      auto w_k = site->pwc_weights();
      auto dens = site->pwc_densities();

      if (DEBUG) {
        std::cout << "Optimizing site" << std::endl << "---------------" << std::endl;
        std::cout << "Starting weights: " << w_k.transpose() << std::endl;
        std::cout << "Starting densities: " << dens.transpose() << std::endl;
        std::cout << "alpha, beta = " << alpha_ << ", " << beta_ << std::endl << std::endl;
      }

      for (int i = 0; i < max_iters; ++i) {
        if (!weight_step_accelerated(site, z_k)) {
          std::cout << "CONVERGED AFTER " << i << " ITERATIONS!" << std::endl;
          break;
        }
      }
      if (DEBUG) {
        dens = site->pwc_densities();
        std::cout << "Weights after convergence: " << (site->pwc_weights()).transpose() << std::endl;
        std::cout << "Densities after convergence: " << dens.transpose() << std::endl;
        std::cout << std::endl;
      }
    }
  }

  void point_step_last(const std::vector<std::shared_ptr<Site<S, V>>>& sites) {
    auto total_density = sites.back()->pwc_weights();
    total_density.setZero();

    auto point = scene_->sample();
    point.setZero();

    int n = sites.back()->pwc_sites();
    for (auto site : sites) {
      auto means = site->pwc_means();
      auto densities = site->pwc_densities();
      total_density += densities;

      auto mean = means.back() * densities(n - 1);
      point += mean;
    }

    point /= total_density(n - 1);

    if (DEBUG) {
      std::cout << std::endl << "New point: " << std::endl;
      std::cout << point.transpose() << std::endl;
      std::cout << std::endl << std::endl;
    }

    for (auto site : sites)
      site->pwc_set_position(point, n - 1); // set new positions
  }

  void point_step(const std::vector<std::shared_ptr<Site<S, V>>>& sites) {
    auto total_density = sites.back()->pwc_weights();
    total_density.setZero();

    std::vector<V> points = sites.back()->pwc_positions();
    for (std::size_t i = 0; i < points.size(); ++i) {
      points[i].setZero();
    }

    for (auto site : sites) {
      auto means = site->pwc_means();
      auto densities = site->pwc_densities();
      total_density += densities;

      for (std::size_t i = 0; i < means.size(); ++i) {
        means[i] *= densities(i);
        points[i] += means[i];
      }
    }
    for (std::size_t i = 0; i < points.size(); ++i)
      points[i] /= total_density(i);

    if (DEBUG) {
      std::cout << std::endl << "Points: " << std::endl;
      for (auto p : points)
        std::cout << p.transpose() << std::endl;
      std::cout << std::endl << std::endl;
    }

    for (auto site : sites) {
      site->pwc_set_positions(points); // set new positions

      // subtract mean from weights
      auto weights = site->pwc_weights();
      weights = (weights.array() - weights.sum() / weights.size());
      site->pwc_set_weights(weights);
    }
  }

  std::vector<V> run(const std::vector<std::shared_ptr<Site<S, V>>>& sites, int max_iters) {
    auto x = scene_->sample();
    std::cout << x.transpose() << std::endl;
    for (auto site : sites) {
      site->pwc_push_back(x, 0.0);
      int n = site->pwc_sites();
      if (n == 1)
        continue;

      auto w = hot_start_weights(site, n - 1);
      site->pwc_set_weight(w, n - 1);
    }

    solve_weights(sites, max_iters);
    for (int i = 0; i < LOCAL_ITERS; ++i) {
      if (GLOBAL_OPT) {
        point_step(sites);
        solve_weights(sites, max_iters);
      } else {
        point_step_last(sites);
      }
    }

    return sites.back()->pwc_positions();
  }

  std::vector<V> run_all(const std::vector<std::shared_ptr<Site<S, V>>>& sites, int max_iters, int pts) {
    for (int i = 0; i < pts; ++i) {
      auto x = scene_->sample();
      for (auto site : sites)
        site->pwc_push_back(x, 0.0);
    }
    // for (auto site : sites) {
    //   auto densities = site->pwc_densities();
    //   for (int i = 0; i < densities.size(); ++i)
    //     if (densities(i) == 0) {
    //       auto w = hot_start_weights(site, i);
    //       site->pwc_set_weight(w, i);
    //     }
    // }

    solve_weights(sites, max_iters);

    for (int i = 0; i < LOCAL_ITERS; ++i) {
      point_step(sites);
      solve_weights(sites, max_iters);
    }

    return sites.back()->pwc_positions();
  }

  bool GLOBAL_OPT = true;
  int LOCAL_ITERS = 1;

 private:

  std::shared_ptr<Scene<S, V>> scene_;

  // parameters for gradient descent
  double alpha_;   // gradient descent step size
  double beta_;    // Nesterov acceleration constant

  const int DEBUG = 1;
};

#endif
