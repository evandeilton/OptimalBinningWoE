// [[Rcpp::depends(RcppEigen, RcppNumerical)]]
// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::plugins(openmp)]]
#include <RcppEigen.h>
#include <RcppNumerical.h>

// #pragma GCC diagnostic push
// #pragma GCC diagnostic ignored "-Wignored-attributes"

using namespace Numer;
using namespace Rcpp;

// Include shared headers
#include "common/optimal_binning_common.h"
#include "common/bin_structures.h"

using namespace Rcpp;
using namespace OptimalBinning;

using namespace Eigen;

// Template class for Logistic Regression without Regularization
template <typename MatrixType>
class LogisticRegression : public MFuncGrad
{
private:
  const MatrixType& X;
  const Map<VectorXd> y;

public:
  LogisticRegression(const MatrixType& X_, const Map<VectorXd>& y_) : X(X_), y(y_) {}

  double f_grad(Constvec& beta, Refvec grad) override
  {
    VectorXd Xbeta = X * beta;
    VectorXd p = 1.0 / (1.0 + (-Xbeta.array()).exp());
    VectorXd diff = p - y;
    grad = X.transpose() * diff;
    double loglik = -(y.array() * Xbeta.array() - (1.0 + Xbeta.array().exp()).log()).sum();
    return loglik;
  }

  // Method to compute Hessian
  MatrixXd hessian(const VectorXd& beta) const
  {
    VectorXd Xbeta = X * beta;
    VectorXd p = 1.0 / (1.0 + (-Xbeta.array()).exp());
    VectorXd w = p.array() * (1 - p.array());
    return X.transpose() * w.asDiagonal() * X;
  }
};

// Helper function to determine if input is sparse
bool is_sparse(SEXP x) {
  return Rf_inherits(x, "dgCMatrix");
}

// Template function for fitting logistic regression
template <typename MatrixType>
List fit_logistic_regression_template(const MatrixType& X, const Map<VectorXd>& y,
                                      int maxit = 300, double eps_f = 1e-8, double eps_g = 1e-5)
{
  LogisticRegression<MatrixType> f(X, y);
  VectorXd beta = VectorXd::Zero(X.cols());
  double fopt;

  int iter_count = 0;

  int status = optim_lbfgs(f, beta, fopt, maxit, eps_f, eps_g);

  iter_count = maxit; // Replace with actual iteration count if available

  VectorXd final_grad(X.cols());
  f.f_grad(beta, final_grad);
  MatrixXd hessian = f.hessian(beta);

  double det = hessian.determinant();
  if (det != 0) {
    VectorXd se = hessian.inverse().diagonal().array().sqrt();

    VectorXd z_scores = beta.array() / se.array();
    VectorXd p_values(z_scores.size());
    for (int i = 0; i < z_scores.size(); ++i) {
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
      Named("convergence") = (status >= 0),
      Named("iterations") = iter_count,
      Named("message") = (status >= 0 ? "converged" : "not converged")
    );
  } else {
    return List::create(
      Named("coefficients") = beta,
      Named("se") = NA_REAL,
      Named("z_scores") = NA_REAL,
      Named("p_values") = NA_REAL,
      Named("loglikelihood") = -fopt,
      Named("gradient") = final_grad,
      Named("hessian") = hessian,
      Named("convergence") = (status >= 0),
      Named("iterations") = iter_count,
      Named("message") = (status >= 0 ? "converged" : "not converged")
    );
  }
}

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