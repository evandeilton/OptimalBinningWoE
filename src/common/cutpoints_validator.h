#ifndef CUTPOINTS_VALIDATOR_H
#define CUTPOINTS_VALIDATOR_H

#include <vector>
#include <algorithm>
#include <cmath>

namespace OptimalBinning {

/**
 * @brief Validate and clean cutpoints for binning operations
 *
 * Ensures cutpoints are unique and sorted. This prevents errors when
 * cutpoints are passed to R's cut() function, which requires unique breaks.
 *
 * The function:
 * 1. Sorts cutpoints in ascending order
 * 2. Removes duplicates using floating-point tolerance (1e-10)
 * 3. Returns cleaned vector
 *
 * @param cutpoints Vector of cutpoints to validate
 * @return Vector of unique, sorted cutpoints
 *
 * @note Uses floating-point comparison with tolerance to handle
 *       numerical precision issues from bin merging operations
 *
 * @example
 * std::vector<double> cp = {10.0, 20.0, 20.0, 30.0};
 * cp = validate_cutpoints(cp);  // Returns {10.0, 20.0, 30.0}
 */
inline std::vector<double> validate_cutpoints(std::vector<double> cutpoints) {
  if (cutpoints.empty()) {
    return cutpoints;
  }

  // Sort cutpoints in ascending order
  std::sort(cutpoints.begin(), cutpoints.end());

  // Remove duplicates within tolerance (1e-10 for floating-point safety)
  auto it = std::unique(cutpoints.begin(), cutpoints.end(),
    [](double a, double b) {
      return std::abs(a - b) < 1e-10;
    });

  cutpoints.erase(it, cutpoints.end());

  return cutpoints;
}

} // namespace OptimalBinning

#endif // CUTPOINTS_VALIDATOR_H
