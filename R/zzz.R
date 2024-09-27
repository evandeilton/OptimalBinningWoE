#' Configure Parallel Processing for Package Installation
#'
#' This function detects the operating system and sets up the environment
#' for parallel processing during package installation. It determines the
#' number of cores to use based on the system's capabilities and sets
#' the appropriate compiler flags for OpenMP support.
#'
#' @return A list with the following components:
#'   \item{os}{Detected operating system (Windows, macOS, or Linux)}
#'   \item{cores}{Number of cores to use for parallel processing}
#'   \item{openmp_flags}{Compiler flags for OpenMP support}
#'
#' @details
#' The function performs the following tasks:
#' 1. Detects the operating system.
#' 2. Determines the number of available cores, using a conservative approach.
#' 3. Sets appropriate compiler flags for OpenMP based on the OS.
#' 4. For macOS, checks if OpenMP is available and provides alternative flags.
#'
#' This function is designed to be called silently during package installation,
#' typically within the .onLoad() function of the package.
#'
#' @note
#' The function is conservative in core allocation to avoid system overload.
#' It uses 50% of available cores on systems with more than 2 cores.
#'
#' @examples
#' \dontrun{
#' parallel_setup <- configure_parallel_setup()
#' print(parallel_setup)
#' }
#'
#' @keywords internal
configure_parallel_setup <- function() {
  # Detect operating system
  os <- .Platform$OS.type
  if (os == "windows") {
    os <- "Windows"
  } else if (Sys.info()["sysname"] == "Darwin") {
    os <- "macOS"
  } else {
    os <- "Linux"
  }
  
  # Determine number of cores to use
  total_cores <- parallel::detectCores()
  cores <- if (total_cores > 2) max(1, floor(total_cores * 0.5)) else 1
  
  # Set OpenMP flags based on OS
  openmp_flags <- switch(os,
                         "Windows" = c("-fopenmp", "-fopenmp"),
                         "Linux" = c("-fopenmp", "-fopenmp"),
                         "macOS" = {
                           # Check if OpenMP is available on macOS
                           if (system("command -v brew >/dev/null 2>&1 && brew --prefix libomp >/dev/null 2>&1", ignore.stdout = TRUE, ignore.stderr = TRUE) == 0) {
                             libomp_path <- system("brew --prefix libomp", intern = TRUE)
                             c(sprintf("-Xclang -fopenmp -I%s/include", libomp_path),
                               sprintf("-lomp -L%s/lib", libomp_path))
                           } else {
                             c("", "")  # No OpenMP support
                           }
                         }
  )
  
  # Return configuration silently
  list(
    os = os,
    cores = cores,
    openmp_flags = openmp_flags
  )
}

.onLoad <- function(libname, pkgname) {
  parallel_setup <- configure_parallel_setup()
  
  # Set the number of threads for OpenMP
  Sys.setenv("OMP_NUM_THREADS" = parallel_setup$cores)
  
  # Set compiler flags for OpenMP
  if (parallel_setup$os == "macOS" && all(parallel_setup$openmp_flags == "")) {
    warning("OpenMP is not available on this macOS system. Parallel processing may be limited.", call. = FALSE)
  } else {
    Sys.setenv("PKG_CXXFLAGS" = parallel_setup$openmp_flags[1])
    Sys.setenv("PKG_LIBS" = parallel_setup$openmp_flags[2])
  }
}
