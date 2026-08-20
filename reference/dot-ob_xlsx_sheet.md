# Internal: Write One Styled Sheet

Internal: Write One Styled Sheet

## Usage

``` r
.ob_xlsx_sheet(wb, name, df, digits = 6L, widths = "auto")
```

## Arguments

- wb:

  An openxlsx workbook.

- name:

  Sheet name; Excel caps these at 31 characters.

- df:

  The table to write.

- digits:

  Rounding applied to double columns.

- widths:

  Column widths, or `"auto"`.
