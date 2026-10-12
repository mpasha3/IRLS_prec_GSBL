"""Generalized cross-validation with a search interval of [1e-9, 1e18]."""

from scipy.optimize import fminbound
from trips.utilities.reg_param.gcv import (
    gcv_numerator,
    gcv_denominator,
    generalized_crossvalidation as _trips_gcv,
)


def generalized_crossvalidation(Q_A, R_A, R_L, b, **kwargs):
    if kwargs.get("gcvtype", "tikhonov") != "tikhonov":
        return _trips_gcv(Q_A, R_A, R_L, b, **kwargs)

    objective = lambda reg_param: (
        gcv_numerator(reg_param, Q_A, R_A, R_L, b)
        / gcv_denominator(reg_param, R_A, R_L, b, **kwargs)
    )
    return fminbound(objective, 1e-9, 1e18, xtol=1e-12,
                     maxfun=1000, full_output=0, disp=0)
