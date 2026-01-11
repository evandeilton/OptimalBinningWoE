# Contributing to OptimalBinningWoE

Thank you for your interest in contributing to OptimalBinningWoE! We
welcome contributions from the community to help make this package
better for everyone.

By participating in this project, you agree to abide by the [Code of
Conduct](https://evandeilton.github.io/OptimalBinningWoE/CODE_OF_CONDUCT.md).

## How You Can Contribute

### Reporting Bugs

If you find a bug, please check the [Issue
Tracker](https://github.com/evandeilton/OptimalBinningWoE/issues) to see
if it has already been reported. If not, please open a new issue and
include:

1.  A clear and descriptive title.
2.  A reproducible example (reprex) demonstrating the issue.
3.  Your session info
    ([`sessionInfo()`](https://rdrr.io/r/utils/sessionInfo.html)).
4.  Any relevant screenshots or error messages.

### Suggesting Enhancements

We welcome suggestions for new features or improvements. Please open an
issue and clearly describe:

1.  The problem you are trying to solve.
2.  The proposed solution or feature.
3.  Examples of how it would work.

### Pull Requests

1.  **Fork** the repository and clone it locally.
2.  Create a **new branch** for your feature or bug fix:
    `git checkout -b feature/my-new-feature`.
3.  Make your changes. Ensure you follow the coding style and add
    comments where necessary.
4.  **Run tests** and ensure everything passes: `devtools::test()`.
5.  **Commit** your changes with clear and descriptive commit messages.
6.  **Push** to your fork and submit a **Pull Request**.

## Development Workflow

### Prerequisites

- R (\>= 4.1.0)
- C++ compiler (supporting C++17)
- `devtools`, `testthat`, `knitr`, `rmarkdown`

### Setup

``` r
# Clone the repository
git clone https://github.com/evandeilton/OptimalBinningWoE.git
cd OptimalBinningWoE

# Open in R or RStudio
# Install dependencies
devtools::install_deps(dependencies = TRUE)
```

### Style Guide

- We use the [tidyverse style guide](https://style.tidyverse.org/).
- Use meaningful variable and function names.
- Document functions using Roxygen2.

### Testing

Please add tests for any new functionality. We use `testthat` for
testing.

``` r
devtools::test()
```

## Questions?

If you have questions, feel free to open a discussion or contact the
maintainer.

Thank you for contributing!
