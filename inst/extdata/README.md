# Bundled example data

## `germancredit.csv.gz`

Statlog (German Credit Data), 1000 loan applications described by 20
attributes (7 numerical, 13 categorical) and a binary credit-risk label.

* **Source:** UCI Machine Learning Repository, dataset 144 —
  <https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data>
* **Donor:** Hans Hofmann, Universität Hamburg (1994).
* **Licence:** Creative Commons Attribution 4.0 International (CC BY 4.0),
  the licence UCI applies to its repository datasets.
* **Encoding of the label:** `credit_risk` is `1` for a good customer and `0`
  for a bad one, so the event (default) indicator used throughout the package
  examples and tests is `1 - credit_risk`, giving a 30% event rate.

The file is the categorical-label version of the dataset: the coded values of
`german.data` (`A11`, `A34`, ...) have been replaced by their meanings from
`german.doc`, which is why category names contain spaces, slashes and colons.
That makes it a realistic test bed for SQL string escaping.

The dataset is used by the package tests and examples to validate Information
Value, KS, Gini and the generated SQL against published reference figures for
this well-known benchmark.
