
# This library contains functions for extracting object level features from the visual search in VR experiment.
#
# Authors: Angela Radulescu

import numpy as np
from scipy.ndimage.filters import gaussian_filter
import itertools
import trimesh
import plotly.graph_objects as go
from joblib import Parallel, delayed
import multiprocessing
from time import perf_counter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import re
import pandas as pd
from scipy.spatial import distance

def kl(p, q):
    """ Computes KL distance between distributions p and q"""

    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)

    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def smoothed_hist_kl_distance(a, b, nbins=20, sigma=1):
    """ Computes histogram and applies a Gaussian filter sets of points a and b for computing
    KL distance. """

    ahist, bhist = (np.histogram(a, bins=nbins)[0],
                    np.histogram(b, bins=nbins)[0])

    asmooth, bsmooth = (gaussian_filter(ahist, sigma),
                        gaussian_filter(bhist, sigma))

    return kl(asmooth, bsmooth)

def d2(vertices, triangles, T=100000):
    """ Loops through a list of 3D meshes defined by vertices and triangles and computes the
    D2 distribution for all objects by sampling from the mesh T times."""

    n_objects = len(vertices)
    D2_distribution = []

    for o in np.arange(n_objects): 

        verts = vertices[o]
        triangs = triangles[o]

        mesh = trimesh.Trimesh(vertices=verts,faces=triangs)

        print("Starting sampling for object " + str(o+1) + " out of " + str(n_objects)) 
        t_start = perf_counter()        
        inputs = np.arange(T) 
        def sample_mesh(i):
            points_on_mesh, d = trimesh.sample.sample_surface(mesh, 2)
            D2 = np.linalg.norm(points_on_mesh[0,:]-points_on_mesh[1,:])
            return D2

        num_cores = multiprocessing.cpu_count()
        results = Parallel(n_jobs=num_cores)(delayed(sample_mesh)(i) for i in inputs)

        D2 = np.vstack(results)
        D2_distribution.append(D2)

        t_stop = perf_counter() 
        print("Elapsed time for D2 sampling: ", t_stop-t_start) 

    return D2_distribution

def d2_similarity(vertices, triangles):
    """ Loops through a list of 3D meshes defined by vertices and triangles and computes the
    pairwise similarity matrix. Deprecated, since the sampling step must be repeated for each object
    every time the object appears in a pair. See shape extraction notebook for most recent version. """

    n_objects = len(vertices)
    object_pairs = list(itertools.combinations(np.arange(n_objects), 2))
    
    similarity_matrix = np.zeros((n_objects, n_objects))
    D2_distribution = []

    p_count = 1

    for p in object_pairs:

        print("Computing pair: " + str(p_count) + " out of " + str(len(object_pairs)))

        ## Define meshes.
        vertices_1 = vertices[p[0]]
        vertices_2 = vertices[p[1]]
        
        triangles_1 = triangles[p[0]]
        triangles_2 = triangles[p[1]]

        mesh_1 = trimesh.Trimesh(vertices=vertices_1,faces=triangles_1)
        mesh_2 = trimesh.Trimesh(vertices=vertices_2,faces=triangles_2)
        
        t1_start = perf_counter()
        T = 200000
        inputs = np.arange(T) 
        def sampleMeshes(i):
            points_on_mesh_1, d = trimesh.sample.sample_surface(mesh_1, 2)
            points_on_mesh_2, d = trimesh.sample.sample_surface(mesh_2, 2)
            D2_1 = np.linalg.norm(points_on_mesh_1[0,:]-points_on_mesh_1[1,:])
            D2_2 = np.linalg.norm(points_on_mesh_2[0,:]-points_on_mesh_2[1,:])
            return D2_1, D2_2
        num_cores = multiprocessing.cpu_count()
        results = Parallel(n_jobs=num_cores)(delayed(sampleMeshes)(i) for i in inputs)

        D2 = np.vstack(results)
        D2_distribution.append(D2)

        D2_1 = D2[:,0]
        D2_2 = D2[:,1]
        
        kl_divergence = smoothed_hist_kl_distance(D2_1, D2_2, nbins=150, sigma=1)
        t1_stop = perf_counter() 
        print("Distance: " + str(kl_divergence))
        print("Elapsed time for KL computation: ", t1_stop-t1_start) 

        p_count = p_count + 1
        
        similarity_matrix[p[0],p[1]] = kl_divergence

    return object_pairs, D2_distribution, similarity_matrix

def plot_mesh(vertices, triangles):

    """ Mesh plotting function."""

    x, y, z = vertices.T
    i, j, k = triangles.T
   
    fig = go.Figure(data=[go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color='black', opacity=0.50)])

    return fig

def pca_wrapper(data, var_explained=0.95):

    """ Wrapper around PCA. Input is a n_datapoints x n_features array."""

    if var_explained == None: var_explained = 0.95

    ## Scale. 
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    ## Fit.
    pca = PCA(var_explained)
    pca.fit(data_scaled)

    return pca

### Occlusion tagging functions.

def parse_auto_occlusion_tags(fp):
    """
    Returns the automated occlusion tags and locations of all the objects 
    in the scene found at fp.
    
    Parameters
    ----------
    
    fp: string
    File path for the .txt file containing the occlusion annotations.
    
    """
    
    with open(fp + '.txt','r') as f:
        num_header_lines = 0 # Number of lines in the text files wich contain metadata but no objects. Fixed when generating these files in Unity
        lines=f.readlines()

        object_labels = []
        occluded_labels = []
        object_locations = []
        for l in lines[num_header_lines:]:
           
            ## Black regex magic.
            object_labels.append(re.findall('"(.*?)"', l)[0])
            occluded_labels.append(re.findall("(?:^|(?<=,))[^,]*", l)[1][1:])
            object_locations.append(re.findall('"(.*?)"', l)[1])

    # objects, occluded, locations_str = visual_search_objects.parse_auto_occlusion_tags(auto_occlusion_direc + 'results_' + scene)
    objects = [o.replace("(Clone)", "") for o in object_labels]
    objects = [o.replace("_LOD0", "") for o in objects]
    objects = [o.replace("_LOD1", "") for o in objects]
    # In auto annotator, True means visible and False means occluded.
    occluded_labels = [(s != 'True')*1 for s in occluded_labels]
    locations_arr = parse_locations(object_locations)
    
    ## Make dataframe. 
    d = {'object': objects, 
        'occluded': occluded_labels, 
        'location': locations_arr}
    
    auto_occlusion_labels = pd.DataFrame(d)

    return auto_occlusion_labels
    
def match_occlusion_labels(auto_occlusion_labels, hand_occlusion_labels):

    """ This function uses location to match the objects in a hand annotation dataframe with those in an auto annotation dataframe. 
    This is necessary because there may be several material objects belonging to the same prefab, and raycasting marks only the top one as visible. 
    There are some fail cases:  
    
    * For some prefabs (e.g. laptop), not all materials inherit the same location.
    * Some objects may have duplicate materials.

    These fail cases can be handled by an enhancement of the tool which generates the output grouped by prefab. 
    Scott MacDonald wrote a piece of code that does this, which I (Angela) placed in the Unity folder, but did not have time to plug it into the project. 
    This should not affect more than 2-3 tags per scene. """

    ## Get locations of objects in the auto annotation. 
    locs_arr = np.vstack(auto_occlusion_labels.location.values)

    ## Mark unique objects. This assumes materials for one object are bundled together. 
    new_object_idx = np.where(np.round(np.sum(np.diff(locs_arr, axis=0), axis=1), 3) != 0)[0]+1 
    new_object = np.zeros((locs_arr.shape[0], 1))
    new_object[0] = 1
    new_object[new_object_idx] = 1

    obj_idx = 0
    unique_obj = []
    for i in np.arange(new_object.shape[0]):
        if new_object[i] == 1:
            obj_idx = obj_idx + 1
        unique_obj.append(obj_idx)
    unique_obj = np.vstack(unique_obj)
    auto_occlusion_labels['unique'] = unique_obj

    ## Remove redundant materials and remake dataframe. 
    auto_new = []
    for u in np.unique(auto_occlusion_labels['unique'].values): 

        u_df = auto_occlusion_labels[auto_occlusion_labels['unique'].values == u].reset_index(drop=True)

        if len(u_df) == 1: auto_new.append(u_df)

        else: 
            # More than one of the materials was visible. 
            if len(np.where(u_df['occluded'] == 0)[0]) > 1: 
                auto_new.append(u_df.loc[np.where(u_df['occluded'] == 0)[0][0]])
            # One of the materials was visible. 
            elif len(np.where(u_df['occluded'] == 0)[0]) == 1:  
                auto_new.append(u_df.loc[np.where(u_df['occluded'] == 0)[0]])
            # No materials were visible.
            else: 
                auto_new.append(u_df.loc[0])

    auto_occlusion_labels_new = pd.DataFrame(np.vstack(auto_new), columns=auto_occlusion_labels.columns)
    auto_occlusion_labels_new.head(10)

    ## Match hand and auto locations and make combined dataframe. 
    h_locs = np.vstack(hand_occlusion_labels.location.values)
    a_locs = np.vstack(auto_occlusion_labels_new.location.values)
    nearest = np.argmin(distance.cdist(h_locs, a_locs),1)
    df = auto_occlusion_labels_new.loc[nearest].reset_index()

    combined_labels = hand_occlusion_labels.copy()
    combined_labels['nearest_auto_annot_obj'] = df['object'].values
    combined_labels['nearest_occluded'] = df['occluded'].values
    combined_labels['nearest_location'] = df['location'].values
       
    return combined_labels

def parse_locations(locations_str):
    
    """ Parses a list of string vector locations obtained from Unity and converts to list of arrays. 
    """
    
    locations_arr = [np.asarray(locations_str[o][1:-1].split(',')).astype(float) for o in np.arange(len(locations_str))]
    
    return locations_arr


