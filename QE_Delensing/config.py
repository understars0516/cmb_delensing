import numpy as np
import os
from os.path import join as opj

import delensalot
import delensalot.core.power.pospace as pospace
from delensalot.utility.utils_hp import gauss_beam
from delensalot.config.config_helper import LEREPI_Constants as lc
from delensalot.config.metamodel.dlensalot_mm import *

dlensalot_model = DLENSALOT_Model(
    job = DLENSALOT_Job(
        jobs = ["generate_sim", "QE_lensrec"]
    ),
    analysis = DLENSALOT_Analysis(
        key = 'p',
        simidxs = np.arange(0,3),
        TEMP_suffix = 'my_first_dlensalot_analysis_fullsky',
        beam = 1.0,
        lm_max_ivf = (4000, 4000),
    ),
    simulationdata = DLENSALOT_Simulation(
        space = 'cl',
        flavour = 'unl',
        lmax = 4096,
        phi_lmax = 5120,
        transfunction = gauss_beam(7.9/180/60 * np.pi, lmax=4096),
        nlev = {'P': np.sqrt(2), 'T': np.sqrt(1)},
        geominfo = ('healpix', {'nside': 2048}),
        lenjob_geominfo = ('thingauss', {'lmax': 4200 + 300, 'smax': 3}),
        CMB_fn = opj(os.path.dirname(delensalot.__file__), 'data', 'cls', 'FFP10_wdipole_lenspotentialCls.dat'),
    ),
    noisemodel = DLENSALOT_Noisemodel(
        nlev = {'P': 2.1, 'T': 1.5},
        geominfo = ('healpix', {'nside': 2048}),
    ),
    qerec = DLENSALOT_Qerec(
        tasks = ["calc_phi", "calc_blt"],
        lm_max_qlm = (3000, 3000),
        cg_tol = 1e-7
    ),
    itrec = DLENSALOT_Itrec(
        tasks = ["calc_phi", "calc_blt"],
        itmax = 5,
        lm_max_unl = (4200, 4200),
        lm_max_qlm = (3000, 3000),
        lenjob_geominfo = ('thingauss', {'lmax': 4200 + 300, 'smax': 3}),
        cg_tol = 1e-7
    ),
    madel = DLENSALOT_Mapdelensing(
        data_from_CFS = False,
        edges = lc.cmbs4_edges,
        iterations = [2],
        masks_fn = [opj(os.environ['SCRATCH'], 'analysis/my_first_dlensalot_analysis_fullsky_lminB200/mask.fits')],
        lmax = 2048,
        Cl_fid = 'ffp10',
        libdir_it = None,
        binning = 'binned',
        spectrum_calculator = pospace,
    )
)
