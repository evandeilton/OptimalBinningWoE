// Maximum-likelihood logistic regression for the "obwoe" scorecard engine.
//
// Plain C++ Newton-Raphson (IRLS) with step halving: no linear-algebra
// dependency. The design has at most a few dozen columns, so the p x p
// systems are solved by a hand-written Cholesky factorisation, and the only
// O(n) work per iteration is X * beta, X' (p - y) and X' W X. Newton converges
// quadratically to the MLE (typically 4-8 iterations, the same fixed point as
// stats::glm), where the former L-BFGS solver stopped on a gradient tolerance
// short of it. Dropping RcppEigen / RcppNumerical also removes several hundred
// compiler warnings their headers emit (-Wignored-attributes) on every build.

#include <Rcpp.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

using namespace Rcpp;

namespace {

// log(1 + exp(z)) without overflow or cancellation.
inline double softplus(double z) {
  return (z > 0.0) ? z + std::log1p(std::exp(-z)) : std::log1p(std::exp(z));
}

// Logistic function without overflow.
inline double inv_logit(double z) {
  if (z >= 0.0) {
    return 1.0 / (1.0 + std::exp(-z));
  }
  const double e = std::exp(z);
  return e / (1.0 + e);
}

// Dense column-major design matrix, n x p.
struct Design {
  std::size_t n = 0, p = 0;
  std::vector<double> x;
  double at(std::size_t i, std::size_t j) const { return x[j * n + i]; }
  const double* col(std::size_t j) const { return x.data() + j * n; }
};

Design read_design(SEXP X_r) {
  Design d;
  if (Rf_isMatrix(X_r) && TYPEOF(X_r) == REALSXP) {
    NumericMatrix X(X_r);
    d.n = static_cast<std::size_t>(X.nrow());
    d.p = static_cast<std::size_t>(X.ncol());
    d.x.assign(X.begin(), X.end());
    return d;
  }
  if (Rf_isS4(X_r) && Rf_inherits(X_r, "dgCMatrix")) {
    IntegerVector dim(R_do_slot(X_r, Rf_install("Dim")));
    IntegerVector ii(R_do_slot(X_r, Rf_install("i")));
    IntegerVector pp(R_do_slot(X_r, Rf_install("p")));
    NumericVector xx(R_do_slot(X_r, Rf_install("x")));
    d.n = static_cast<std::size_t>(dim[0]);
    d.p = static_cast<std::size_t>(dim[1]);
    d.x.assign(d.n * d.p, 0.0);
    for (std::size_t j = 0; j < d.p; ++j) {
      for (int k = pp[static_cast<R_xlen_t>(j)]; k < pp[static_cast<R_xlen_t>(j) + 1]; ++k) {
        d.x[j * d.n + static_cast<std::size_t>(ii[k])] = xx[k];
      }
    }
    return d;
  }
  stop("X_r must be a double matrix or a dgCMatrix.");
}

// eta = X beta
void linear_predictor(const Design& X, const std::vector<double>& beta,
                      std::vector<double>& eta) {
  eta.assign(X.n, 0.0);
  for (std::size_t j = 0; j < X.p; ++j) {
    const double b = beta[j];
    if (b == 0.0) continue;
    const double* c = X.col(j);
    for (std::size_t i = 0; i < X.n; ++i) eta[i] += c[i] * b;
  }
}

// Log-likelihood sum_i y_i eta_i - log(1 + exp(eta_i)).
double loglik(const std::vector<double>& eta, const std::vector<double>& y) {
  double ll = 0.0;
  for (std::size_t i = 0; i < eta.size(); ++i) ll += y[i] * eta[i] - softplus(eta[i]);
  return ll;
}

// Gradient of the NEGATIVE log-likelihood, X'(mu - y), and its Hessian
// X' W X with W = diag(mu (1 - mu)). Four columns of the Hessian are
// accumulated per pass for instruction-level parallelism.
void grad_hess(const Design& X, const std::vector<double>& eta,
               const std::vector<double>& y, std::vector<double>& g,
               std::vector<double>& H) {
  const std::size_t n = X.n, p = X.p;
  std::vector<double> r(n), w(n), v(n);
  for (std::size_t i = 0; i < n; ++i) {
    const double mu = inv_logit(eta[i]);
    r[i] = mu - y[i];
    w[i] = mu * (1.0 - mu);
  }
  g.assign(p, 0.0);
  H.assign(p * p, 0.0);
  for (std::size_t j = 0; j < p; ++j) {
    const double* cj = X.col(j);
    double gj = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      gj += cj[i] * r[i];
      v[i] = cj[i] * w[i];
    }
    g[j] = gj;
    std::size_t k = 0;
    for (; k + 4 <= j + 1; k += 4) {
      const double* c0 = X.col(k);
      const double* c1 = X.col(k + 1);
      const double* c2 = X.col(k + 2);
      const double* c3 = X.col(k + 3);
      double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
      for (std::size_t i = 0; i < n; ++i) {
        const double vi = v[i];
        s0 += vi * c0[i];
        s1 += vi * c1[i];
        s2 += vi * c2[i];
        s3 += vi * c3[i];
      }
      H[j * p + k] = s0;
      H[j * p + k + 1] = s1;
      H[j * p + k + 2] = s2;
      H[j * p + k + 3] = s3;
    }
    for (; k <= j; ++k) {
      const double* ck = X.col(k);
      double s = 0.0;
      for (std::size_t i = 0; i < n; ++i) s += v[i] * ck[i];
      H[j * p + k] = s;
    }
  }
  for (std::size_t j = 0; j < p; ++j) {
    for (std::size_t k = j + 1; k < p; ++k) H[j * p + k] = H[k * p + j];
  }
}

// In-place Cholesky factorisation A = L L' of a symmetric p x p matrix
// (row-major; only the lower triangle is read and L overwrites it).
// Returns false when A is not numerically positive definite.
bool cholesky(std::vector<double>& A, std::size_t p) {
  for (std::size_t j = 0; j < p; ++j) {
    double d = A[j * p + j];
    for (std::size_t k = 0; k < j; ++k) d -= A[j * p + k] * A[j * p + k];
    if (!(d > 0.0) || !std::isfinite(d)) return false;
    const double ljj = std::sqrt(d);
    A[j * p + j] = ljj;
    for (std::size_t i = j + 1; i < p; ++i) {
      double s = A[i * p + j];
      for (std::size_t k = 0; k < j; ++k) s -= A[i * p + k] * A[j * p + k];
      A[i * p + j] = s / ljj;
    }
  }
  return true;
}

// Solves L L' x = b given the Cholesky factor (lower triangle of L).
void chol_solve(const std::vector<double>& L, std::size_t p, std::vector<double>& b) {
  for (std::size_t i = 0; i < p; ++i) {
    double s = b[i];
    for (std::size_t k = 0; k < i; ++k) s -= L[i * p + k] * b[k];
    b[i] = s / L[i * p + i];
  }
  for (std::size_t ii = p; ii-- > 0;) {
    double s = b[ii];
    for (std::size_t k = ii + 1; k < p; ++k) s -= L[k * p + ii] * b[k];
    b[ii] = s / L[ii * p + ii];
  }
}

// 1-norm of a symmetric p x p matrix stored row-major.
double norm1(const std::vector<double>& A, std::size_t p) {
  double best = 0.0;
  for (std::size_t j = 0; j < p; ++j) {
    double s = 0.0;
    for (std::size_t i = 0; i < p; ++i) s += std::fabs(A[i * p + j]);
    best = std::max(best, s);
  }
  return best;
}

}  // namespace

// [[Rcpp::export]]
List fit_logistic_regression(SEXP X_r, const NumericVector& y_r,
                             int maxit = 300, double eps_f = 1e-8, double eps_g = 1e-5) {
  const Design X = read_design(X_r);
  const std::size_t n = X.n, p = X.p;
  if (static_cast<std::size_t>(y_r.size()) != n) {
    stop("Number of rows in X_r must match the length of y_r.");
  }
  if (maxit < 1) stop("maxit must be at least 1.");
  const std::vector<double> y(y_r.begin(), y_r.end());

  std::vector<double> beta(p, 0.0), eta, g, H, L, step(p), trial(p), eta_trial;
  linear_predictor(X, beta, eta);
  double ll = loglik(eta, y);

  bool converged = false;
  std::string failure;
  int iter = 0;

  while (iter < maxit) {
    ++iter;
    grad_hess(X, eta, y, g, H);

    // Newton direction H^{-1} g. A (near-)singular Hessian -- collinear
    // columns or fitted probabilities pinned at 0/1 -- gets the smallest
    // ridge that makes it positive definite, so the iteration still moves
    // uphill instead of stopping.
    L = H;
    double ridge = 0.0;
    double diag_max = 0.0;
    for (std::size_t j = 0; j < p; ++j) diag_max = std::max(diag_max, std::fabs(H[j * p + j]));
    if (diag_max <= 0.0 || !std::isfinite(diag_max)) diag_max = 1.0;
    while (!cholesky(L, p)) {
      ridge = (ridge == 0.0) ? 1e-10 * diag_max : ridge * 10.0;
      if (ridge > 1e10 * diag_max) break;
      L = H;
      for (std::size_t j = 0; j < p; ++j) L[j * p + j] += ridge;
    }
    if (ridge > 1e10 * diag_max) {
      failure = "Hessian could not be factorised";
      break;
    }
    step = g;
    chol_solve(L, p, step);

    // Step halving (as stats::glm.fit): accept the first step that does not
    // decrease the log-likelihood.
    double t = 1.0, ll_new = -std::numeric_limits<double>::infinity();
    bool accepted = false;
    for (int half = 0; half < 60; ++half) {
      for (std::size_t j = 0; j < p; ++j) trial[j] = beta[j] - t * step[j];
      linear_predictor(X, trial, eta_trial);
      ll_new = loglik(eta_trial, y);
      if (std::isfinite(ll_new) && ll_new >= ll - 1e-12 * std::fabs(ll)) {
        accepted = true;
        break;
      }
      t *= 0.5;
    }
    if (!accepted) {
      failure = "step halving failed to increase the log-likelihood";
      break;
    }
    beta.swap(trial);
    eta.swap(eta_trial);

    // glm.fit's deviance criterion, deviance = -2 log-likelihood.
    const double dev_old = -2.0 * ll, dev_new = -2.0 * ll_new;
    ll = ll_new;
    if (std::fabs(dev_new - dev_old) / (std::fabs(dev_new) + 0.1) < eps_f) {
      converged = true;
      break;
    }
  }

  // Final gradient and Hessian at the returned estimate.
  grad_hess(X, eta, y, g, H);
  double gnorm = 0.0, bnorm = 0.0;
  for (std::size_t j = 0; j < p; ++j) {
    gnorm += g[j] * g[j];
    bnorm += beta[j] * beta[j];
  }
  gnorm = std::sqrt(gnorm);
  bnorm = std::sqrt(bnorm);
  // The iteration cap counts as convergence only if the gradient test holds.
  if (!converged && failure.empty() && std::isfinite(gnorm) &&
      (gnorm <= eps_g || gnorm <= eps_g * bnorm)) {
    converged = true;
  }

  // Standard errors from the inverse Hessian, provided the Hessian is well
  // conditioned: exact reciprocal 1-norm condition number rcond(H) > 1e-12.
  bool hessian_ok = p > 0;
  for (std::size_t k = 0; k < p * p && hessian_ok; ++k) hessian_ok = std::isfinite(H[k]);
  std::vector<double> Hinv(p * p, 0.0);
  if (hessian_ok) {
    L = H;
    hessian_ok = cholesky(L, p);
  }
  if (hessian_ok) {
    std::vector<double> e(p);
    for (std::size_t j = 0; j < p; ++j) {
      std::fill(e.begin(), e.end(), 0.0);
      e[j] = 1.0;
      chol_solve(L, p, e);
      for (std::size_t i = 0; i < p; ++i) Hinv[i * p + j] = e[i];
    }
    const double rc = 1.0 / (norm1(H, p) * norm1(Hinv, p));
    hessian_ok = std::isfinite(rc) && rc > 1e-12;
  }

  NumericVector coefficients(beta.begin(), beta.end());
  NumericVector gradient(g.begin(), g.end());
  NumericMatrix hessian(static_cast<int>(p), static_cast<int>(p));
  for (std::size_t i = 0; i < p; ++i) {
    for (std::size_t j = 0; j < p; ++j) {
      hessian(static_cast<int>(i), static_cast<int>(j)) = H[i * p + j];
    }
  }

  std::string message = converged ? "converged" : "not converged";
  if (!converged && !failure.empty()) message += ": " + failure;

  if (hessian_ok) {
    NumericVector se(p), z_scores(p), p_values(p);
    for (std::size_t j = 0; j < p; ++j) {
      const R_xlen_t jj = static_cast<R_xlen_t>(j);
      se[jj] = std::sqrt(std::max(0.0, Hinv[j * p + j]));
      z_scores[jj] = beta[j] / se[jj];
      p_values[jj] = 2.0 * R::pnorm(std::fabs(z_scores[jj]), 0.0, 1.0, false, false);
    }
    return List::create(
      Named("coefficients") = coefficients,
      Named("se") = se,
      Named("z_scores") = z_scores,
      Named("p_values") = p_values,
      Named("loglikelihood") = ll,
      Named("gradient") = gradient,
      Named("hessian") = hessian,
      Named("convergence") = converged,
      Named("iterations") = iter,
      Named("message") = message
    );
  }

  return List::create(
    Named("coefficients") = coefficients,
    Named("se") = NA_REAL,
    Named("z_scores") = NA_REAL,
    Named("p_values") = NA_REAL,
    Named("loglikelihood") = ll,
    Named("gradient") = gradient,
    Named("hessian") = hessian,
    Named("convergence") = converged,
    Named("iterations") = iter,
    Named("message") = converged ? std::string("converged (singular hessian)") : message
  );
}
