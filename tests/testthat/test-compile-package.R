## Preparativos do Pacote
rm(list = ls())
library(data.table);library(Rcpp);library(testthat);

# sourceCpp("src/algo.cpp")
# sourceCpp("src/dataprep.cpp")
# sourceCpp("src/utils.cpp")

## ---------------------------------------------------------------------------------------------- ##

# devtools::load_all()
# devtools::check()
# devtools::document()
# 
# Rcpp::compileAttributes()           # this updates the Rcpp layer from C++ to R
# roxygen2::roxygenize(roclets="rd")  # this updates the documentation based on roxygen comments
# pkgdown::init_site()
# pkgdown::build_site()
# pkgdown::build_site_github_pages()
# devtools::build_readme()
# devtools::build_manual()
# devtools::build()
# devtools::install()
# usethis::use_version()


# usethis::use_github_action()
# usethis::use_pkgdown()
# usethis::use_pkgdown_github_pages()
# usethis::use_tidy_style()
# usethis::use_pipe()
# usethis::use_citation()

# usethis::use_rcpp()
# usethis::use_mit_license()
# usethis::use_pkgdown()
# usethis::use_readme_rmd()
