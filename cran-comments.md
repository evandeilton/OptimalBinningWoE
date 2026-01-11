## R CMD check results

0 errors | 0 warnings | 1 note

* This is a new release.

## Test environments
* local Zorin OS 17 (Linux), R 4.5.2
* win-builder (devel and release)

## Reverse dependencies

This is a new package, so there are no reverse dependencies.

## Explanation of NOTEs

* "New submission" - expected for initial release.
* "Compilation used the following non-portable flag(s): '-mno-omit-leaf-frame-pointer'" - This flag originates from the R configuration on the check system and not from the package's Makevars. The package uses standard C++17 configuration.
