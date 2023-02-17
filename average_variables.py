import numpy as np
from py_wake.rotor_avg_models import CGIRotorAvg

# Rotor model to average deficit and turbulence fields
n_points = 21
cgi = CGIRotorAvg(n_points)


def average_variables(wfm, wt_x, wt_y, x, wd, ws):
    windTurbines = wfm.windTurbines
    D = windTurbines.diameter()
    zRef = windTurbines.hub_height()
    x_j = np.array([x for _ in cgi.nodes_x]).flatten()
    y_j = np.array([x * 0 + y * D/2 for y in cgi.nodes_x]).flatten()
    h_j = np.array([x * 0 + zRef + z * D/2 for z in cgi.nodes_y]).flatten()

    sim_res = wfm(wt_x, wt_y, ws=ws, wd=270)
    lw_j, WS_eff_jlk, TI_eff_jlk = wfm._flow_map(x_j, y_j, h_j, sim_res)
    # TI and WS effective variables
    TI_eff = TI_eff_jlk.reshape((len(cgi.nodes_x), len(x))).T
    WS_eff = WS_eff_jlk.reshape((len(cgi.nodes_x), len(x))).T
    # Averaged over the rotor with the rotormodel
    cgi.nodes_weight = 1/n_points
    ws_avg = (WS_eff * cgi.nodes_weight).sum(1)
    ti_avg = (TI_eff * cgi.nodes_weight).sum(1)
    return ws_avg, ti_avg

def get_UAD(wfm, wt_x, wt_y, wd, ws):
    wt_x = np.asarray(wt_x)
    wt_y = np.asarray(wt_y)
    windTurbines = wfm.windTurbines
    D = windTurbines.diameter()
    zRef = windTurbines.hub_height()
    sim_res = wfm(wt_x, wt_y, ws=ws, wd=270)
    
    UAD = []
    x_j = np.array([wt_x for _ in cgi.nodes_x]).flatten()
    for xx, yy in zip(wt_x, wt_y):
        y_j = np.array([wt_x * 0 + yy + y * D/2 for y in cgi.nodes_x]).flatten()
        h_j = np.array([wt_x * 0 + zRef + z * D/2 for z in cgi.nodes_y]).flatten()
        lw_j, WS_eff_jlk, TI_eff_jlk = wfm._flow_map(x_j, y_j, h_j, sim_res)
        WS_eff = WS_eff_jlk.reshape((len(cgi.nodes_x), len(wt_x))).T
        cgi.nodes_weight = 1/n_points
        ws_avg = (WS_eff * cgi.nodes_weight).sum(1)
        UAD.append(np.interp(xx, wt_x, ws_avg))
    return np.asarray(UAD)

def average_rans_data(wfm, wt_x, wt_y, x, ws, ti, rans_deficits, rans_added_ti):
    rans_ws = np.zeros((len(x), len(cgi.nodes_x)))
    rans_ti = np.zeros((len(x), len(cgi.nodes_x)))
    windTurbines = wfm.windTurbines
    D = windTurbines.diameter()
    zRef = windTurbines.hub_height()
    x_j = np.array([x for _ in cgi.nodes_x]).flatten()
    y_j = np.array([x * 0 + y * D/2 for y in cgi.nodes_x]).flatten()
    h_j = np.array([x * 0 + zRef + z * D/2 for z in cgi.nodes_y]).flatten()
    points_per_node = len(y_j)/len(cgi.nodes_x)
    xii = x - wt_x[-1]/2
    for node in range(n_points):
        xj = int(node * points_per_node)
        rans_ws[:, node] = np.asarray(rans_deficits.U.interp(x=xii, y=y_j[xj], z=h_j[xj])).ravel() * ws/8
        rans_ti[:, node] = np.asarray(rans_added_ti.I.interp(x=xii, y=y_j[xj], z=h_j[xj])).ravel() + ti
    cgi.nodes_weight = 1/n_points
    rans_ws_avg = (rans_ws * cgi.nodes_weight).sum(1)
    rans_ti_avg = (rans_ti * cgi.nodes_weight).sum(1)
    return rans_ws_avg, rans_ti_avg

def average_rans_from_nc(windTurbines, wt_x, wt_y, x, ws, ti, dataset):
    rans_ws = np.zeros((len(x), len(cgi.nodes_x)))
    rans_ti = np.zeros((len(x), len(cgi.nodes_x)))
    D = windTurbines.diameter()
    zRef = windTurbines.hub_height()
    x_j = np.array([x for _ in cgi.nodes_x]).flatten()
    y_j = np.array([x * 0 + y * D/2 for y in cgi.nodes_x]).flatten()
    h_j = np.array([x * 0 + zRef + z * D/2 for z in cgi.nodes_y]).flatten()
    points_per_node = len(y_j)/len(cgi.nodes_x)
    # xii = x - wt_x[-1]/2
    for node in range(n_points):
        xj = int(node * points_per_node)
        rans_ws[:, node] = np.asarray(dataset.U.interp(x=x, y=y_j[xj], z=h_j[xj])).ravel()
        rans_ti[:, node] = ((2/3 * np.asarray(dataset.tke.interp(x=x, y=y_j[xj], z=h_j[xj]))) ** 0.5).ravel()/ws
    cgi.nodes_weight = 1/n_points
    rans_ws_avg = (rans_ws * cgi.nodes_weight).sum(1)
    rans_ti_avg = ti + (rans_ti * cgi.nodes_weight).sum(1)
    
    return rans_ws_avg, rans_ti_avg


def setup_layout(n_wt, windTurbines, spacing, staggering):
    D = windTurbines.diameter()
    wt_x = [spacing * D * i for i in range(n_wt)]
    wt_y = [staggering * D * j for j in range(n_wt)]
    return wt_x, wt_y

