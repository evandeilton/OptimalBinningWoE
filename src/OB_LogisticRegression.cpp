// [[Rcpp::depends(RcppEigen, RcppNumerical)]]
#include <RcppEigen.h>
#include <RcppNumerical.h>

#include <algorithm>
#include <cmath>
#include <string>

using namespace Rcpp;
using namespace Eigen;

namespace {

// log(1 + exp(z)) for a whole vector. The direct formula is kept wherever it
// is finite (so fits that never reach extreme linear predictors are unchanged
// bit for bit); above z = 700 exp(z) is about to overflow to Inf, which used to
// turn the objective into Inf and derail the line search on (quasi-)separated
// data, so the algebraically equal z + log1p(exp(-z)) is used there.
inline double softplus(double z) {
  return (z <= 700.0) ? std::log(1.0 + std::exp(z)) : z + std::log1p(std::exp(-z));
}

// Negative log-likelihood of a logistic regression without regularisation.
template <typename MatrixType>
class LogisticRegression : public Numer::MFuncGrad
{
private:
  const MatrixType& X;
  const Map<VectorXd> y;

public:
  LogisticRegression(const MatrixType& X_, const Map<VectorXd>& y_) : X(X_), y(y_) {}

  double f_grad(Numer::Constvec& beta, Numer::Refvec grad) override
  {
    VectorXd Xbeta = X * beta;
    VectorXd p = 1.0 / (1.0 + (-Xbeta.array()).exp());
    VectorXd diff = p - y;
    grad = X.transpose() * diff;
    // Same (vectorised) expression as before, patched only where it overflows.
    VectorXd lp = (1.0 + Xbeta.array().exp()).log();
    for (Index i = 0; i < lp.size(); ++i) {
      if (!(Xbeta[i] <= 700.0) || !std::isfinite(lp[i])) lp[i] = softplus(Xbeta[i]);
    }
    double loglik = -(y.array() * Xbeta.array() - lp.array()).sum();
    return loglik;
  }

  // Hessian of the negative log-likelihood: X' W X, W = diag(p (1 - p)).
  MatrixXd hessian(const VectorXd& beta) const
  {
    VectorXd Xbeta = X * beta;
    VectorXd p = 1.0 / (1.0 + (-Xbeta.array()).exp());
    VectorXd w = p.array() * (1 - p.array());
    return X.transpose() * w.asDiagonal() * X;
  }
};

bool is_sparse(SEXP x) {
  return Rf_inherits(x, "dgCMatrix");
}

// Restores Eigen's thread count on scope exit.
//
// This translation unit is compiled with the package's OpenMP flags, and Eigen
// then parallelises its large matrix products over omp_get_max_threads()
// threads -- every core of the machine by default, against the CRAN policy
// of at most two cores unless the user asks for more. The fit is capped at two
// threads (never more than OpenMP already allows) and the previous setting is
// put back afterwards.
struct EigenThreadCap {
  int saved;
  EigenThreadCap() : saved(Eigen::nbThreads()) {
    Eigen::setNbThreads(std::max(1, std::min(2, saved)));
  }
  ~EigenThreadCap() { Eigen::setNbThreads(saved); }
};

template <typename MatrixType>
List fit_logistic_regression_template(const MatrixType& X, const Map<VectorXd>& y,
                                      int maxit = 300, double eps_f = 1e-8, double eps_g = 1e-5)
{
  EigenThreadCap thread_cap;
  LogisticRegression<MatrixType> f(X, y);
  VectorXd beta = VectorXd::Zero(X.cols());
  double fopt = 0.0;

  // Same solver and parameters as Numer::optim_lbfgs(), called directly so the
  // number of iterations is known. optim_lbfgs() hides it: `iterations` was
  // always reported as `maxit`, and `convergence` was TRUE even when the
  // iteration limit had been exhausted.
  Numer::LBFGSFun fun(f);
  LBFGSpp::LBFGSParam<double> param;
  param.epsilon        = eps_g;
  param.epsilon_rel    = eps_g;
  param.past           = 1;
  param.delta          = eps_f;
  param.max_iterations = maxit;
  param.max_linesearch = 100;
  param.linesearch     = LBFGSpp::LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE;

  int status = 0;
  int iter_count = NA_INTEGER;   // unknown if the solver throws
  std::string failure;
  {
    VectorXd xx = beta;
    try {
      LBFGSpp::LBFGSSolver<double> solver(param);
      iter_count = solver.minimize(fun, xx, fopt);
    } catch (const std::exception& e) {
      // optim_lbfgs() turned this into an R warning; the failure is reported
      // through `convergence` and `message` instead.
      status = -1;
      failure = e.what();
    }
    beta = xx;
  }

  VectorXd final_grad(X.cols());
  const double fbeta = f.f_grad(beta, final_grad);
  if (status < 0) fopt = fbeta;   // the solver did not report a value

  // The iteration cap counts as convergence only if the gradient test holds.
  const double gnorm = final_grad.norm();
  const bool grad_ok = std::isfinite(gnorm) &&
    (gnorm <= eps_g || gnorm <= eps_g * beta.norm());
  const bool converged = (status >= 0) && (iter_count < maxit || grad_ok);

  MatrixXd hessian = f.hessian(beta);

  // Singularity is judged from the conditioning of the Hessian. The old test,
  // |det(H)| > 1e-10 * ||H||, compares a quantity that scales like lambda^p
  // with one that scales like lambda: a perfectly well-conditioned Hessian
  // with small eigenvalues (predictors on a small scale, or simply p >= 10)
  // was declared singular and every standard error came back NA, while a
  // near-singular Hessian with large eigenvalues passed.
  bool hessian_ok = hessian.allFinite() && hessian.rows() > 0;
  Eigen::LDLT<MatrixXd> ldlt;
  if (hessian_ok) {
    ldlt.compute(hessian);
    hessian_ok = (ldlt.info() == Eigen::Success) && ldlt.isPositive() &&
      (ldlt.vectorD().array() > 0.0).all() &&
      ldlt.rcond() > 1e-12;
  }

  const char* msg_ok = converged ? "converged" : "not converged";
  std::string message = msg_ok;
  if (!converged && !failure.empty()) message += std::string(": ") + failure;

  if (hessian_ok) {
    MatrixXd H_inv = ldlt.solve(MatrixXd::Identity(hessian.rows(), hessian.cols()));
    // cwiseMax(0) prevents sqrt(negative) NaN from tiny numerical errors.
    VectorXd se = H_inv.diagonal().cwiseMax(0.0).array().sqrt();

    VectorXd z_scores = beta.array() / se.array();
    VectorXd p_values(z_scores.size());
    for (Index i = 0; i < z_scores.size(); ++i) {
      p_values(i) = 2.0 * (1.0 - R::pnorm(std::abs(z_scores(i)), 0.0, 1.0, true, false));
    }

    return List::create(
      Named("coefficients") = beta,
      Named("se") = se,
      Named("z_scores") = z_scores,
      Named("p_values") = p_values,
      Named("loglikelihood") = -fopt,
      Named("gradient") = final_grad,
      Named("hessian") = hessian,
      Named("convergence") = converged,
      Named("iterations") = iter_count,
      Named("message") = message
    );
  }

  return List::create(
    Named("coefficients") = beta,
    Named("se") = NA_REAL,
    Named("z_scores") = NA_REAL,
    Named("p_values") = NA_REAL,
    Named("loglikelihood") = -fopt,
    Named("gradient") = final_grad,
    Named("hessian") = hessian,
    Named("convergence") = converged,
    Named("iterations") = iter_count,
    Named("message") = converged ? std::string("converged (singular hessian)") : message
  );
}

} // namespace

// [[Rcpp::export]]
List fit_logistic_regression(SEXP X_r, const NumericVector& y_r,
                             int maxit = 300, double eps_f = 1e-8, double eps_g = 1e-5)
{
  const Map<VectorXd> y(as<Map<VectorXd>>(y_r));

  if (is_sparse(X_r)) {
    const MappedSparseMatrix<double> X(as<MappedSparseMatrix<double>>(X_r));
    return fit_logistic_regression_template(X, y, maxit, eps_f, eps_g);
  } else {
    const Map<MatrixXd> X(as<Map<MatrixXd>>(X_r));
    return fit_logistic_regression_template(X, y, maxit, eps_f, eps_g);
  }
}
