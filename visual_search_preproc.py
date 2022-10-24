# This library contains functions for preprocessing eyetracking data
# from the visual search in VR experiment.
#
# Authors: Bas van Opheusden, Angela Radulescu

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.interpolate
import quaternion as qt
from sklearn.linear_model import LinearRegression
from itertools import groupby
from operator import itemgetter

#################################################################
## Projection to trial specific camera-centric coordinate system.
#################################################################

def get_gaze_direction_camera_centric(d, camera_locations):
    """
    This function takes one participant's data and returns an array with coordinates of gaze for each
    eye tracker sample, in a static coordinate frame.

    For each trial, the coordinate frame is centered at the main camera position of that trial
    (the location participants are teleported to at the start of the trial), and the orientation aligned
    so that the z-axis is the forward-looking cirection of the camera.

    For eye tracker samples taken outside the trials (in the ITI), there will be rows in the returned gaze matrix,
    but they should never be used
    """

    # Get the camera location and orientation at each trial, taking into account the random ordering of trials
    camera_locations_at_trial_start, camera_rotation_at_trial_start = zip(*[camera_locations[trial_name.split(' Trial ')[0]]
                                                                           for _,trial_name in d['Behavior']['TrialOrder']])

    # For each time point in d['Eye']['TimeStamp'], we use scipy.interpolate to find the last trial which was
    # started before that time,
    # which allows us to identify the correct reference frame to use for that time point.
    # Note that time points outside trials will therefore use the reference frame of the last preceding trial,
    # which is not useful.
    # For time points before any trial has started, the interpolation function returns NaNs
    f = scipy.interpolate.interp1d(d['Behavior']['TrialStartTimes'],camera_locations_at_trial_start,
                                   kind='previous',axis=0,fill_value=(np.nan, camera_locations_at_trial_start[-1]),bounds_error=False)
    cam_loc = f(d['Eye']['TimeStamp'])

    #same for camera orientation
    f = scipy.interpolate.interp1d(d['Behavior']['TrialStartTimes'],camera_rotation_at_trial_start,
                                   kind='previous',axis=0,fill_value=(np.nan, camera_rotation_at_trial_start[-1]),bounds_error=False)
    cam_dir = f(d['Eye']['TimeStamp'])

    #Compute the point where the person's gaze hit a collider (the mesh which defines an object in Unity)
    #This we take as the target of their gaze
    #We could also try to infer their gaze target from vergence data, but since we know the scene graph
    #we can assume that people always fixate on the surface of an object, never in mid-air
    g = d['Eye']['GazeLocation']+d['Eye']['GazeDirection']*d['Eye']['HitDistance'][:,None]

    #subtract the camera location to get relative position in the camera-centric frame
    gaze_from_camera = rotate_gaze(cam_dir,g-cam_loc)
    #Rotate the reference frame and convert to polar coordinates
    gaze_from_camera_polar = convert_to_polar(normalize(gaze_from_camera))

    return gaze_from_camera_polar

##############################################################
## Projection to frame by frame egocentric coordinate system.
##############################################################

def get_last_known_input(d,t):
    """
    Helper function for the gaze-to-pixel coordinate transformation below. Figures out for each provided time stamp
    which data stream in d contains the last preceding information
    """
    f = scipy.interpolate.interp1d(np.hstack([d['Eye']['TimeStamp'],
                                              d['Controller']['TimeStamp'],
                                              d['Behavior']['TimeStamp']]),
                                   np.hstack([np.ones_like(d['Eye']['TimeStamp']),
                                              2*np.ones_like(d['Controller']['TimeStamp']),
                                              3*np.ones_like(d['Behavior']['TimeStamp'])]),kind='previous',axis=0)

    return f(t)

def get_gaze_to_pixel_regression(gaze,screen_coords,last_known_input,i):
    """
    This function uses a linear regression to figure out the transformation from gaze cordinates in Unity's global frame
    To the pixel coordinates in VisualSearch Videos.

    Input:
    gaze: matrix with normalized gaze directions in egocentric space, one for each sample of the VisualSearchVideo movie
    screen_coords: matrix with pixel coordinates of gaze saved by Unity during the construction of the movie
    last_known_input: array with the output of get_last_known_input for the correct participant and screenshot_times
    i: index of the coordiante which we want to regress (0 for horizontal/x, 1 for vertical/y)

    Output:
    regression coefficients which relate world to pixel coordinate
    """
    #select only those time points for which the last preceding input before a screenshot is an eye tracking sample.
    #this eliminates time points where the screen_coords file and the gaze input are not lined up
    x = gaze[last_known_input==1,i]
    y = screen_coords[last_known_input==1,i]

    #eliminate nans
    inds = np.logical_and(~np.isnan(x),~np.isnan(y))

    #regress gaze and screen_coords to find scaling.
    regression_output = LinearRegression(fit_intercept=True).fit(x[inds,None],y[inds])
    return [regression_output.intercept_,regression_output.coef_[0]]

def gaze_to_pixel_coords(g,beta):
    """
    Converts gaze in world coordinates to the pixel representation of VisualSearchVideos, using the regression coefficients
    Computed by get_gaze_to_pixel_regression()
    """
    return beta[:,0]+beta[:,1]*g[:,:2]

def get_times_before_trials(d,screen_times):
    """
    Function which split up the screenshot times into trials, and keeps the part in the pre-trial target viewing period
    Input:
    d: one participant's data
    screen_times: the screenshot times for the same participant

    Output:
    List of 300 lists (I like lists), with the timestamps of all the screenshots in the pre-trial target viewing period,
    as well as the indices of the corresponding frames of the VisualSearchVideo
    """
    #note: this code is not particalurly efficient and can probably be sped up
    return [[(i,ts) for i,ts in enumerate(screen_times) if ts>s and ts<t]
                       for s,t in pair_times(d['Behavior']['PlayerOutsideTimes'],
                                                                   d['Behavior']['TrialStartTimes'])]

def get_times_in_trials(d,screen_times):
    """
    Same as above, but for times during trials
    """
    return [[(i,ts) for i,ts in enumerate(screen_times) if ts>s and ts<t]
                       for s,t in pair_times(d['Behavior']['TrialStartTimes'],
                                                                   d['Behavior']['SearchOverTimes'])]

###########
## Helpers.
###########

def rotate_gaze(head,gaze):
    """
    Computes egocentric gaze coordinates for a matrix of head orientations and
    gaze directions (assumed to be same size)
    """
    return np.array([qt.rotate_vectors(q.inverse(),v)
                      for q,v in zip(convert_to_quaternion(head),gaze)])

def convert_to_quaternion(A):
    """
    Converts an N-by-4 array to an N-by-1 array of quaternions

    Note the cycling of indices which is necessary since Unity encodes the
    quaternion a + b*i + c*j + d*k as (b,c,d,a) whereas Python encodes it as
    (a,b,c,d)
    """
    return qt.as_quat_array(np.hstack([A[:,3:],A[:,:3]]))

def convert_to_polar(A):
    """
    Converts an array of normalized vectors in a user-centric cartesian
    coordinates to a polar representation. In cartesian coordinates (x,y,z),
    the z-coordinate is the user's forward, the y-direction is upwards and x
    is lateral. In polar coordinates, the first angle is the radial coordinate,
    the second the azimuth.
    """
    return np.hstack([np.arctan2(A[:,0],A[:,2])[:,None],np.arcsin(A[:,1])[:,None]])

def convert_polar_to_geographic(A):
    """ 
    Converts a 2D array of vectors in a user-centric polar representation to
    a geographic coordinate representation.
    """

    latitude = A[:,0]-90
    longitude = A[:,1]

def normalize(A):
    """
    Normalizes each column in a 2-D numpy array, or normalizes a 1-D array
    Normalization performed with regards to L2 norm
    """
    if len(A.shape)>1:
        return A/np.sqrt(np.sum(A**2,axis=1))[:,None]
    else:
        return A/np.sqrt(np.sum(A**2))

def get_ambient_luminance(im):
    """
    Returns the average luminance across the entire visual field (all pixels in the screenshot)
    The magic numbers [0.2126,0.7152,0.0722] are taken from https://en.wikipedia.org/wiki/Relative_luminance
    """
    return np.mean(np.sum(im*(np.array([0.2126,0.7152,0.0722])[None,None,:]),axis=2))/255

def get_focal_luminance(im,coords,radius):
    """
    Returns the average luminance across a circle of given radius around the provided coordinate
    (can be the user's gaze coordinates or a relevant control)
    """

    X,Y = np.meshgrid(np.arange(0,1080,1),np.arange(0,1000,1)[::-1])

    ind = (X-coords[0])**2 + (Y-coords[1])**2 < radius**2
    return np.mean(np.sum(im[ind]*(np.array([0.2126,0.7152,0.0722])[None,None,:]),axis=2))/255

def get_color_at_gaze(im,coords,radius):
    """
    Returns the average color (in RGB space) of an image at the provided coordinates in a circle for the provided radius
    Any part of the circle that falls outside the image size is ignored. if the entire circle falls outside
    Input:
    im: N x K x 3 image
    coords: binary tuple
    radius: float

    Output: 1x3 vector of RGB values
    """
    X,Y = np.meshgrid(np.arange(0,1080,1),np.arange(0,1000,1)[::-1])

    #Here it is crucially important that Y is reversed, to flip the vertical axis of each image
    ind = (X-coords[0])**2 + (Y-coords[1])**2 < radius**2
    return np.nanmean(im[ind],axis=0)

def show_color(c):
    #plotting function which displays a rectangle of a given color
    #Input: a vector of RGB values (in 0-255 range) for a given color
    #output: none, but plots a rectangle of the input color
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.add_patch(patches.Rectangle(xy = [0,0], width =1, height =1, color = c/255))
    plt.show()

def get_nonextreme_ind(A,cutoff=99):
    """
    Helper function which finds all indices in an array A for which the absolute difference between
    one row of A and the next is in the 99th percentile. Used to filter extreme events in eye tracking data,
    which are likely tracking errors

    Input: A, numpy array of any size
    Output: mask with the same size of A
    """
    if len(A.shape)==1:
        A=A[:,None]
    #take the absolute difference of A along the first dimension. Almost aalways, the first dimension will be time
    A = np.abs(np.diff(A,axis=0))
    #add abs(A_t - A_t-1) and abs(A_t+1 - A_t)
    A = np.vstack([A,np.zeros([1,A.shape[1]])]) + np.vstack([np.zeros([1,A.shape[1]]),A])
    #mask out nans
    mask = np.all(~np.isnan(A),axis=1)
    #update the mask to also mask out all values for which the absolute difference exceeds the 99th percentile
    mask[mask==True] = np.all(A[mask,:]<np.nanpercentile(A[mask,:],99),axis=1)
    return mask

def pair_times(t1,t2):
    """
    Loops through a pair of lists of event times, and for each time t in the
    first list it finds the first time s in the second list that's later than t
    Used primarily to separate events by the trial in which they occurred
    """
    return [(t,np.min([s for s in t2 if s>t])) for t in t1]

def epoch_frames(inds_in_trial):
    """
    Epochs frames. Returns a list of frame indices for which we were in
    a trial, as returned by get_times_in_trials. Length of list should be n_trials.
    """

    frames_in_trial = []
    for k, g in groupby(enumerate(inds_in_trial), lambda ix : ix[0] - ix[1]):
        frames_in_trial.append(list(map(itemgetter(1), g)))

    return frames_in_trial

def epoch_gaze(data):
    """
    Epochs gaze data. Returns a list of gaze indices for which we were in
    a trial. Length of list should be n_trials. Assumes data is from one participant.

    # TO-DO: to make consistent with times_in_trials, pass d as variable (data from one participant)
    """

    # Number of trials fixed.
    n_trials = 300

    # Initialize list for eyetracking samples during trials.
    gaze_in_trial = []

    # Loop and get trial onset and offset.
    for t in np.arange(n_trials):

        trial_onset = data[0]["Behavior"]["TrialStartTimes"][t]
        trial_offset = data[0]["Behavior"]["SearchOverTimes"][t]

        gaze_in_trial.append(np.where(np.logical_and(np.array(data[0]["Eye"]["TimeStamp"]) > trial_onset, np.array(data[0]["Eye"]["TimeStamp"]) < trial_offset)))

    return gaze_in_trial

def map_gaze_to_objects(data,gaze_in_trial):
    """
    Maps sequence of gaze samples to object labels. Assumes data is from one participant.
    # TO-DO: to make consistent with times_in_trials, pass d as variable (data from one participant)
    """

    # Clean object labels.
    n_samples = len(data[0]["Eye"]["HitObject"])
    gaze_object_labels = [data[0]["Eye"]["HitObject"][s].strip().replace('(Clone)','') for s in np.arange(n_samples)]

    # Map to trial.
    n_trials = 300
    objects_in_trial = [np.array(gaze_object_labels)[gaze_in_trial[t]] for t in np.arange(n_trials)]

    return objects_in_trial

