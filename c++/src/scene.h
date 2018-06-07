#ifndef SCENE_H
#define SCENE_H

#include <memory>

#include "powercell.h"
#include "sampler.h"

/** The SCENE represents the universe of possible points.
 *
 * There should only be one SCENE instantiated at any given instant.
 * The SCENE is meant to represent a bounding box on all given
 * input distributions, and is mostly used for volume computation
 * of given powercells and powercell boundaries.
 */
template<typename S, typename V>
class Scene {
 public:
  Scene(S* sampler) : sampler_(sampler), vol_(0.0) {}
  Scene(S* sampler, double vol) : sampler_(sampler), vol_(vol) {}

  /** Return the volume of the SCENE.
   *
   * The VOLUME is a user input to the constructor, and thus this output is
   * only correct if the constructor argument is correct.
   */
  double vol() const {
    return vol_;
  }

  /** Return a uniform sample from within the SCENE.
   *
   * The SAMPLER is an user input to the constructor, and thus this sample is
   * only uniform if the input sampler is uniform over the SCENE geometry.
   */
  V sample() const {
    return sampler_->sample();
  }

  /** Monte-Carlo integrator for volume of a boolean function within the scene.
   */
  double mc_volume(const std::function<bool(V)>& in, int iters) const {
    double cin = 0.0;
    int ctotal = iters;

    for (int i = 0; i < iters; ++i)
      if (in(sample()))
        cin += 1.0;

    return vol_ * (cin / ctotal);
  }

  /** Monte-Carlo integrator for the volumes of each piece of a partition of a scene.
   */
  V mc_volumes(const std::function<int(V)>& in, int n, int iters) const {
    V areas(n);
    areas.setZero();

    int ctotal = iters;
    for (int i = 0; i < iters; ++i) {
      int id = in(sample());
      areas(id) += 1.0;
    }

    return vol_ * (areas / ctotal);
  }

  /** Volume of a powercell index by ID.
   */
  double powercell_volume(std::shared_ptr<Powercell<V>> pwc, int id, int iters) const {
    auto in = [&](V p) { return (pwc->in(p) == id); };

    return mc_volume(in, iters);
  }

  /** Volumes of all powercells in a given Laguerre decomposition.
   */
  V powercell_volumes(std::shared_ptr<Powercell<V>> pwc, int iters) const {
    auto in = [&](V p) { return pwc->in(p); };

    return mc_volumes(in, iters);
  }

 private:
  double vol_;
  std::unique_ptr<S> sampler_;
};

#endif
