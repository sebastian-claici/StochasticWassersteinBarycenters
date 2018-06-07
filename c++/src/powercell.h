#ifndef POWERCELL_H
#define POWERCELL_H

#include <functional>
#include <limits>
#include <memory>

template<typename V>
struct Boundary {
  std::vector<V> basis;
  std::function<bool(V)> in;
};

/** A POWERCELL is a set of points associated with a set of weights that
 * generalizes the Voronoi diagram.
 *
 * A site i in a powercell (X, w) is the set of all points x such that
 *        ||x - X_i||_2^2 - w_i <= ||x - X_j||_2^2 - w_j    for all j
 */
template<typename V>
class Powercell {
 public:
  Powercell():
    dropoff_(std::numeric_limits<int>::max()) {}

  Powercell(int dropoff):
    dropoff_(dropoff) {}

  int sites() const {
    return X_.size();
  }

  /** Add a site at position X with weight W.
   *
   * This function overwrites early points if we're only keeping
   * the most recent K sites
   */
  template<typename U>
  void push_back(U&& x, double w) {
    total_++;
    if (total_ >= dropoff_) {
      w_.conservativeResize(dropoff_ - 1);
      X_.erase(X_.begin());
    }

    X_.push_back(std::forward<U>(x));
    w_.conservativeResize(w_.size() + 1);
    w_(w_.size() - 1) = w;
  }

  /** Change the position of site X_i to X
   */
  template<typename U>
  void set_position(U&& x, int i) {
    X_[i] = std::forward<U>(x);
  }

  void set_positions(const std::vector<V>& X) {
    X_ = X;
  }

  /** Change the weight of site X_i to W
   */
  void set_weight(double w, int i) {
    w_(i) = w;
  }

  template<typename U>
  void set_weights(U&& w) {
    w_ = std::forward<U>(w);
  }

  /** Return the powercell site that contains point P.
   */
  template<typename U>
  int in(U&& p) {
    double dist = (X_[0] - p).squaredNorm() - w_(0);
    int id = 0;

    for (int i = 1; i < X_.size(); ++i) {
      double new_dist = (X_[i] - p).squaredNorm() - w_(i);
      if (new_dist < dist) {
        dist = new_dist;
        id = i;
      }
    }

    return id;
  }

  /** Return the BOUNDARY between sites i and j
   */
  Boundary<V> boundary(int i, int j) {
    V xi = X_[i];
    V xj = X_[j];

    // find center point and normal to hyperplane
    double e_ij = (xi - xj).norm();
    double d_ij = (e_ij * e_ij + w_(i) - w_(j)) / (2 * e_ij);

    V c_ij = xi + d_ij / e_ij * (xj - xi);

    auto in = [&](V&& p) -> bool {
     double d_i = (xi - p).squaredNorm() - w_(i);
     double d_j = (xj - p).squaredNorm() - w_(j);
     if (std::fabs(d_i - d_j) > 1e-6)
       return false;
     for (int k = 0; k < sites(); ++k) {
       if (i == k || j == k)
         continue;
       double d_k = (X_[k] - p).squaredNorm() - w_(k);
       if (d_k < d_i)
         return false;
     }
     return true;
    };

    return Boundary<V>(std::vector<V>(), in);
  }

 public:
  std::vector<V> X_;   // site centers
  V w_;                // site weights

 private:
  int dropoff_;         // number of most recent sites to keep
  int total_{0};           // total number of sites added
};

#endif
