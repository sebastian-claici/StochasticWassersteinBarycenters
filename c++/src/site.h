#ifndef SITE_H
#define SITE_H

#include <memory>

const int ITERS=256000;

/** A SITE is an extension of a distribution.
 *
 * A SITE can compute densities and means of a given Laguerre decomposition
 * with respect to its own density.
 *
 * Each SITE has an associated Laguerre decomposition of the SCENE. That means that
 * even if the positions and weights are shared across all SITES, there will be as
 * many copies of those positions and weights as there are SITES. This is done for
 * extensibility mostly.
 */
template<typename S, typename V>
class Site {
 public:
  Site(Sampler<V>* sampler, std::shared_ptr<Powercell<V>> pwc) : sampler_(sampler), pwc_(pwc) {}

  V sample() const {
    return sampler_->sample();
  }

  // Powercell functions
  VectorXd pwc_weights() const {
    return pwc_->w_;
  }

  std::vector<V> pwc_positions() const {
    return pwc_->X_;
  }

  int pwc_sites() const {
    return pwc_->sites();
  }

  template<typename U>
  void pwc_push_back(U&& x, double w) const {
    pwc_->push_back(x, w);
  }

  template<typename U>
  void pwc_set_position(U&& x, int i) const {
    pwc_->set_position(x, i);
  }

  void pwc_set_positions(const std::vector<V>& X) const {
    pwc_->set_positions(X);
  }

  void pwc_set_weight(double w, int i) const {
    pwc_->set_weight(w, i);
  }

  template<typename U>
  void pwc_set_weights(U&& w) const {
    pwc_->set_weights(w);
  }

  /** Compute the powercell densities wrt the distribution given by SAMPLER.
   *
   * Currently computed using a simple counting scheme and Monte-Carlo integration.
   */
  VectorXd pwc_densities(int iters=ITERS) {
    if (pwc_->sites() == 0)
      return V(1);

    VectorXd density(pwc_->sites());
    density.setZero();

    sample_points(iters);
    int ctotal = iters;
    int taken = 0;
    for (auto p : cached_) {
      if (taken++ >= iters)
        break;
      density(pwc_->in(p)) += 1;
    }

    return density / ctotal;
  }

  VectorXd pwc_parallel_densities(int iters=ITERS) {
    if (pwc_->sites() == 0)
      return V(1);

    int n = pwc_->sites();
    VectorXd density(n);
    density.setZero();

    #pragma omp parallel
    {
      const int nthreads = omp_get_num_threads();
      const int liters = iters / nthreads;

      VectorXd dens_local(n);
      #pragma omp for nowait
      for (int i = 0; i < liters; ++i) {
        auto p = sample();
        dens_local(pwc_->in(p))++;
      }
      #pragma omp critical
      {
        density += dens_local;
      }
    }

    return density / density.sum();
  }

  /** Compute the powercell means wrt the distribution given by SAMPLER.
   *
   * Currently computed using a Monte-Carlo integrator as an expectation of a random variable.
   */
  std::vector<V> pwc_means(int iters=ITERS) {
    V zero = sample();
    zero.setZero();

    std::vector<V> means(pwc_->sites(), zero);
    std::vector<int> cnt(pwc_->sites(), 0);

    sample_points(iters);
    int taken = 0;
    for (auto p : cached_) {
      if (taken++ >= iters)
        break;

      int id = pwc_->in(p);
      cnt[id]++;
      means[id] += p;
    }

    for (std::size_t i = 0; i < means.size(); ++i)
      means[i] /= cnt[i];

    return means;
  }

 private:
  void sample_points(int iters) {
    auto need = iters - cached_.size();

    if (need > 0) {
      for (int i = 0; i < need; ++i) {
        auto p = sample();
        cached_.push_back(p);
      }
    }
  }

  std::vector<V> cached_;
  std::shared_ptr<Powercell<V>> pwc_;
  std::unique_ptr<Sampler<V>> sampler_;
};

#endif
