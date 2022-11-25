import numpy as np
import geotable
import utm


def kml2list(files, booleans):
    '''
    This function returns a multipolygon boundary list in the required format
    from a list of kml files.

    Inputs
    ------
    files: list of .kml files imported from google earth.
    booleans: list of boolean variables assigned to each file, 0 = exclusion
    and 1 = inclusion.

    Outputs
    -------
    list of tupples (first component of the tupple are the polygon vertices,
                     second component is the boolean).
    '''

    def load_files(file, boolean):
        t = geotable.load(file)
        polygons = t['geometry_object'].tolist()
        return polygons

    def polygon_list_to_topfarm_bounds(obj_list, boolean):
        return [(np.asarray(utm.from_latlon(np.asarray(x.exterior.xy[1]), np.asarray(x.exterior.xy[0]))[0:2]).T, boolean) for x in obj_list]

    bound_lists = []
    topfarm_lists = []

    for file, boolean in zip(files, booleans):
        bound_list = load_files(file, boolean)
        topfarm_list = polygon_list_to_topfarm_bounds(bound_list, boolean)

        bound_lists.extend(bound_list)
        topfarm_lists.extend(topfarm_list)

    return topfarm_lists


def main():
    if __name__ == '__main__':
        '''Example to instantiate a multi polygon boundary constraint in 
        Topfarm. 
        
        Inputs: kml file 
        
        Outputs: list of tupples (polygon vertices, boolean) 
        '''
        import matplotlib.pyplot as plt
        from topfarm import XYBoundaryConstraint
        
        files = ['polygon_example.kml']
        booleans = [1]
        boundaries = kml2list(files, booleans)
        xybound = XYBoundaryConstraint(boundaries, 
                                       boundary_type = 'multi_polygon')
        
        for n, bound in enumerate(xybound.get_comp(n_wt=1).boundaries):
            x_bound, y_bound = bound[0].T
            x_bound = np.append(x_bound, x_bound[0])
            y_bound = np.append(y_bound, y_bound[0])
            plt.plot(x_bound, y_bound, color='k', linewidth=0.5)
        plt.axis('equal')

main()
