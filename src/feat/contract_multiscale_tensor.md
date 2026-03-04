# MULTISCALE_TENSOR_CONTRACT_V1

This module builds fixed-shape multi-scale feature tensors for `(code, asof_date=T)`.

## Scales
- **micro**: `5m`, 1 day, length `48`
- **mezzo**: `30m`, 5 days, length `40`
- **macro**: `1d`, 30 days, length `30`

## Channels (C1..C7)
1. `C1 = ln(Close_t / Close_{t-1})`
2. `C2 = ln(Open_t / Close_{t-1})`
3. `C3 = ln(High_t / Low_t)`
4. `C4 = (Close_t - Low_t)/(High_t - Low_t)` and `0.5` when `High_t == Low_t`
5. `C5 = Volume_t / mean(Volume_{t-20:t-1})`
6. `C6 = std(Return_{t-5:t-1}, ddof=0)` where `Return_k = ln(Close_k/Close_{k-1})`
7. `C7 = VWAP_t / Close_t`

## Strict `dp_ok` policy
`dp_ok=True` only when all three scales are valid and every bar has all seven channels computable.
Any missing bar/field/history causes full datapoint drop.

When invalid:
- all tensors are zero-padded with shape `[48,7]`, `[40,7]`, `[30,7]`
- all masks are zeros
- `dp_ok=False`

## Calendar
Trading calendar is generated only from filenames in the daily data directory using regex:
`(20\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])`.

The calendar is the sole source for window alignment.
