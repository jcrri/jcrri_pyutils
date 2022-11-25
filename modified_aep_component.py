from py_wake.examples.data.iea37._iea37 import IEA37_WindTurbines, IEA37Site
from topfarm._topfarm import TopFarmProblem
from topfarm.constraint_components.boundary import CircleBoundaryConstraint
from topfarm.plotting import XYPlotComp
from topfarm.cost_models.cost_model_wrappers import AEPCostModelComponent
from topfarm.easy_drivers import EasyRandomSearchDriver
from topfarm.drivers.random_search_driver import RandomizeTurbinePosition_Circle
from py_wake.deficit_models.gaussian import IEA37SimpleBastankhahGaussian
import topfarm
import numpy as np
from py_wake.flow_map import Points
from py_wake.utils.gradients import autograd
import time

class modified_aep_component(AEPCostModelComponent):
    '''
    This cost model component is based on the PyWakeAEPCostComponent but it
    adds an option that records the AEP value and the number of iterations
    in a text file. In addition, the optimization is not interrupted when
    two wind turbines are in the same position. Instead, a penalty is applied
    that makes the AEP = 0. 
    '''
    def __init__(self, windFarmModel, n_wt, wd=None, ws=None, max_eval=None, grad_method=autograd, n_cpu=1, wd_chunks=None, text_file=None, **kwargs):
        self.windFarmModel = windFarmModel
        self.n_cpu = n_cpu
        self.wd_chunks = wd_chunks
        self.text_file = text_file

        def aep(**kwargs):
            try: 
                aep = self.windFarmModel.aep(x=kwargs[topfarm.x_key],
                                             y=kwargs[topfarm.y_key],
                                             h=kwargs.get(topfarm.z_key, None),
                                             type=kwargs.get(topfarm.type_key, 0),
                                             wd=wd, ws=ws,
                                             n_cpu=n_cpu,
                                             wd_chunks=wd_chunks)
            except:
                aep = 0
                print('WT are at same positions!')
                
           # my text recorder
            if text_file != None:
                with open(text_file, 'a') as filehandle:
                    filehandle.write('%s, %s, %s \n' % (self.n_grad_eval, time.time(), aep))
                    filehandle.close()
            return aep

        if grad_method:
            if hasattr(self.windFarmModel, 'dAEPdxy'):
                # for backward compatibility
                dAEPdxy = self.windFarmModel.dAEPdxy(grad_method)
            else:
                def dAEPdxy(**kwargs):
                    return self.windFarmModel.aep_gradients(
                        gradient_method=grad_method, wrt_arg=['x', 'y'], n_cpu=n_cpu, **kwargs)

            def daep(**kwargs):
                return dAEPdxy(x=kwargs[topfarm.x_key],
                               y=kwargs[topfarm.y_key],
                               h=kwargs.get(topfarm.z_key, None),
                               type=kwargs.get(topfarm.type_key, 0),
                               wd=wd, ws=ws)
        else:
            daep = None
        AEPCostModelComponent.__init__(self,
                                       input_keys=[topfarm.x_key, topfarm.y_key],
                                       n_wt=n_wt,
                                       cost_function=aep,
                                       cost_gradient_function=daep,
                                       output_unit='GWh',
                                       max_eval=max_eval, **kwargs)

    def get_aep4smart_start(self, ws=[6, 8, 10, 12, 14], wd=np.arange(360)):
        def aep4smart_start(X, Y, wt_x, wt_y, type=0):
            sim_res = self.windFarmModel(wt_x, wt_y, type=type, wd=wd, ws=ws, n_cpu=self.n_cpu)
            H = np.full(X.shape, self.windFarmModel.windTurbines.hub_height())
            return sim_res.aep_map(Points(X, Y, H), n_cpu=self.n_cpu).values
        return aep4smart_start


def main():
    if __name__ == '__main__':
        n_wt = 16
        site = IEA37Site(n_wt)
        windTurbines = IEA37_WindTurbines()
        windFarmModel = IEA37SimpleBastankhahGaussian(site, windTurbines)
        tf = TopFarmProblem(
            design_vars=dict(zip('xy', site.initial_position.T)),
            cost_comp=modified_aep_component(windFarmModel, n_wt, text_file=None),
            driver=EasyRandomSearchDriver(randomize_func=RandomizeTurbinePosition_Circle(), max_iter=5),
            constraints=[CircleBoundaryConstraint([0, 0], 1300.1)],
            plot_comp=XYPlotComp())
        tf.optimize()
        tf.plot_comp.show()


main()
