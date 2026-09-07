from __future__ import annotations

"""Repeated-root Z_348 [[696,188,*]] exact/exact campaign wrapper."""

import gb_m350_k188_repeated_campaign as campaign
from gf2_factor import factor_xm_plus_one


campaign.M = 348
campaign.N = 696
campaign.Q = 94
campaign.K = 188
campaign.CHECK_CAP = 32
campaign.MODULUS = (1 << campaign.M) | 1
campaign.ROOT_MULTIPLICITY = 4
campaign.BASE_FACTORS = tuple(factor_xm_plus_one(87))
campaign.PAIR_EXACT_ONLY = True
campaign._gcd_exponents.cache_clear()
campaign._configure_core()
campaign._install_overrides()


if __name__ == "__main__":
    campaign.main()
