// [[Rcpp::depends(RcppEigen, RcppNumerical)]]
// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::plugins(openmp)]]
#include <RcppEigen.h>
#include <RcppNumerical.h>

// #pragma GCC diagnostic push
// #pragma GCC diagnostic ignored "-Wignored-attributes"

using namespace Numer;
using namespace Rcpp;
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

//' @title Logistic Regression with Optional Hessian Calculation
//'
//' @description
//' This function performs logistic regression using a gradient-based optimization algorithm (L-BFGS)
//' and provides the option to compute the Hessian matrix for variance estimation. It supports both
//' dense and sparse matrices as input.
//'
//' @param X_r A matrix of predictor variables. This can be a dense matrix (`MatrixXd`) or a sparse matrix (`dgCMatrix`).
//' @param y_r A numeric vector of binary target values (0 or 1).
//' @param maxit Maximum number of iterations for the L-BFGS optimization algorithm (default: 300).
//' @param eps_f Convergence tolerance for the function value (default: 1e-8).
//' @param eps_g Convergence tolerance for the gradient (default: 1e-5).
//'
//' @return A list containing the following elements:
//' \item{coefficients}{A numeric vector of the estimated coefficients for each predictor variable.}
//' \item{se}{A numeric vector of the standard errors of the coefficients, computed from the inverse Hessian (if applicable).}
//' \item{z_scores}{Z-scores for each coefficient, calculated as the ratio between the coefficient and its standard error.}
//' \item{p_values}{P-values corresponding to the Z-scores for each coefficient.}
//' \item{loglikelihood}{The negative log-likelihood of the final model.}
//' \item{gradient}{The gradient of the log-likelihood function at the final estimate.}
//' \item{hessian}{The Hessian matrix of the log-likelihood function, used to compute standard errors.}
//' \item{convergence}{A boolean indicating whether the optimization algorithm converged successfully.}
//' \item{iterations}{The number of iterations performed by the optimization algorithm.}
//' \item{message}{A message indicating whether the model converged or not.}
//'
//' @details
//' The logistic regression model is fitted using the L-BFGS optimization algorithm. For sparse matrices, the algorithm
//' automatically detects and handles the matrix efficiently.
//'
//' The log-likelihood function for logistic regression is maximized:
//' \deqn{\log(L(\beta)) = \sum_{i=1}^{n} \left( y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right)}
//' where \eqn{p_i} is the predicted probability for observation \eqn{i}.
//'
//' The Hessian matrix is computed to estimate the variance of the coefficients, which is necessary for calculating
//' the standard errors, Z-scores, and p-values.
//'
//' @references
//' \itemize{
//'   \item Nocedal, J., & Wright, S. J. (2006). Numerical Optimization. Springer Science & Business Media.
//'   \item Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer.
//' }
//'
//' @author
//' José E. Lopes
//'
//' @examples
//' \dontrun{
//' # Create sample data
//' set.seed(123)
//' X <- matrix(rnorm(1000), ncol = 10)
//' y <- rbinom(100, 1, 0.5)
//'
//' # Run logistic regression
//' result <- fit_logistic_regression(X, y)
//'
//' # View results
//' print(result$coefficients)
//' print(result$p_values)
//' }
//' @import Rcpp
//' @import RcppNumerical
//' @import RcppEigen
//' @export
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
