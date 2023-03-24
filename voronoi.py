from numba import njit
import noise

def gen_voronoi_map(self,x,y n_points):
    """
    Generate a Voronoi (Worley) noise map
    n_points is the number of randomly distributed points on the map that the voronoi cells are defined by
    """

    loc_coords = []
    
    #Create n number of random point coordinates
    for n in range(n_points):
        x_loc = np.random.randint(0,x)
        y_loc = np.random.randint(0,y)
        pnt = np.array((x_loc,y_loc))
        loc_coords.append(pnt)

    #Calculate distances
    voronoi_map = voronoi_distances(loc_coords, n_points, x, y)

    #Normalize map to a 0-1 range
    voronoi_map = ((voronoi_map - voronoi_map.min()) / (voronoi_map.max() - voronoi_map.min))

    return voronoi_map

@njit
def voronoi_distances(loc_coords, n_points, x, y):

    point_distances = np.zeros((n_points))
    dist_arr = np.zeros((x,y))
    for i in range(len(dist_arr)):
        for j in range(len(dist_arr[0])):
            for k in range(len(loc_coords)):
                point = loc_coords[k]
                dist = np.sqrt((point[0] - i)**2 + (point[1] - j)**2)
                point_distances[k] = dist

            min_dist_ind = np.argmin(point_distances)
            dist_arr[i][j] = point_distances[min_dist_ind]

    return dist_arr
