import numpy as np
import xarray as xr
from numpy import newaxis as na
from py_wake.deficit_models.deficit_model import WakeDeficitModel, BlockageDeficitModel
from py_wake.turbulence_models.turbulence_model import TurbulenceModel
from py_wake.superposition_models import LinearSum #as turbSuperposition
from py_wake.utils.grid_interpolator import GridInterpolator


class Rans_Lut_Deficit(WakeDeficitModel, BlockageDeficitModel):
    args4deficit = ['WS_ilk', 'WS_eff_ilk', 'TI_eff_ilk',  'D_dst_ijl', 'dw_ijlk', 'hcw_ijlk', 'dh_ijlk', 'h_ilk', 'ct_ilk', 'yaw_ilk']

    def __init__(self, path):
        BlockageDeficitModel.__init__(self, upstream_only=True)
        lut = xr.open_dataset(path)
        self.lut_interpolator = GridInterpolator(
            [lut.ti.values, lut.ct.values, lut.x.values, lut.y.values, lut.z.values], lut.deficits.values,
            bounds='limit') 
        lut.close()
        
    def calc_deficit(self, WS_eff_ilk, TI_eff_ilk, dw_ijlk, hcw_ijlk, dh_ijlk, h_ilk, ct_ilk, D_src_il, D_dst_ijl, yaw_ilk, **_):
        IJLKX = list(hcw_ijlk.shape)
        IJLKX[3] = ct_ilk.shape[2]
        IJLKX =tuple(IJLKX)

        def lim(x, i):
            c = self.lut_interpolator.x[i]
            return np.minimum(np.maximum(x, c[0]), c[-1])
        
        xp = np.array([np.broadcast_to(v, IJLKX).flatten()
                        for v in [lim(TI_eff_ilk[:, na], 0),
                                  lim(ct_ilk[:, na], 1),
                                  lim(dw_ijlk, 2),
                                  lim(hcw_ijlk, 3),
                                  (h_ilk[:, na] + dh_ijlk)]]).T
        
        du_ijlk = WS_eff_ilk[:, na] * self.lut_interpolator(xp).reshape(IJLKX) * \
        ~((dw_ijlk == 0) & (hcw_ijlk <= D_src_il[:, na, :, na]/2)
          )
        return du_ijlk

class Rans_Lut_Turbulence(TurbulenceModel):
    args4addturb = ['dw_ijlk', 'cw_ijlk', 'D_src_il',  'D_dst_ijl', 'ct_ilk', 'TI_eff_ilk', 'dh_ijlk', 'h_ilk']

    def __init__(self, path, addedTurbulenceSuperpositionModel=LinearSum(), use_effective_ti=True, **kwargs):
        TurbulenceModel.__init__(self, addedTurbulenceSuperpositionModel, **kwargs)
        lut = xr.open_dataset(path)
        self.lut_interpolator = GridInterpolator( 
            [lut.ti.values, lut.ct.values, lut.x.values, lut.y.values, lut.z.values], lut.added_ti.values,
            bounds='limit')
        lut.close()
        self.use_effective_ti = use_effective_ti
        
    def calc_added_turbulence(self, ct_ilk, dw_ijlk, TI_eff_ilk, hcw_ijlk, h_ilk, dh_ijlk, D_src_il, D_dst_ijl, yaw_ilk, **_):
        IJLKX = list(hcw_ijlk.shape)
        IJLKX[3] = ct_ilk.shape[2]
        IJLKX =tuple(IJLKX)

        def lim(x, i):
            c = self.lut_interpolator.x[i]
            return np.minimum(np.maximum(x, c[0]), c[-1])
        
        xp = np.array([np.broadcast_to(v, IJLKX).flatten()
                        for v in [lim(TI_eff_ilk[:, na],0),
                                  lim(ct_ilk[:, na],1),
                                  lim(dw_ijlk, 2),
                                  lim(hcw_ijlk, 3),
                                  (h_ilk[:, na] + dh_ijlk)]]).T
        
        TI_add_ijlk = self.lut_interpolator(xp).reshape(IJLKX) *\
            ~((np.abs(dw_ijlk) < 1e-10) & (np.abs(hcw_ijlk) <= D_src_il[:, na, :, na]/2)
            )
        if self.use_effective_ti == False:
            TI_add_ijlk = 0 * TI_add_ijlk
        return TI_add_ijlk
