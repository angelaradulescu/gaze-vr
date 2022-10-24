# This library contains helper functions for loading eyetracking data
# from the visual search in VR experiment.
#
# Authors: Bas van Opheusden, Angela Radulescu

import pickle
import numpy as np
import ast
import datetime as dt
import xml.etree.ElementTree as et
from xdf import load_xdf
import scipy.interpolate
import cv2
from ast import literal_eval
import pandas as pd
import quaternion as qt

def get_direc_participants(dataset, base):
    if dataset == 'Expt1':
        data_direc = base
        participant_list = ['BC_07192018','CG_07232018','ER_07232018','JKP_07202018','JY_07192018',
                            'KC_07242018','LL_07242018','LV_07212018','MLW_07202018','SO_07202018',
                            'SP_07192018','SR_07232018','SSS_07202018','YD_07202018','AL_07242018',
                            'RD_07242018','LK_07242018']

    elif dataset == 'Expt2':
        data_direc = base + 'Expt2/'
        participant_list = ['P2_09262018','P3_09272018','P5_09272018','P6_10022018',
                            'P7_10022018','P8_10032018','P9_10032018','P10_10042018',
                            'P11_10052018','P12_10052018','P13_10082018','P14_10082018',
                            'P15_10082018','P16_10082018','P17_10082018','P18_10092018',
                            'P20_10092018','P22_10102018','P23_10102018','P24_10102018',
                            'P26_10112018']

        ## Exclusion notes: 
        # p1 adaptive visual search -- no notes, check with BVO
        # P4_09272018 -- crashes in the video rendering 
        # P19_10092018 -- crashes in the video rendering 
        # P21_10102018 -- crashes in visual_search_preproc.get_gaze_to_pixel_regression
        # P25_10112018 --  only has partial data
        # P27_10122018 --  crashes in visual_search_preproc.get_gaze_to_pixel_regression
        
        # Check with BVO regarding crashes
		# Compare ExperimentFinishedTime(?) to last time screenshot times, should be 4 seconds after 
		# Making video function assumes a 2s buffer at the of the experiment 

    elif dataset == 'Pittsburgh':
        data_direc = base + 'Pittsburgh/'
        participant_list = ['20180924']
    return data_direc,participant_list

def open_video(participant, vid_direc):
    print(vid_direc + participant + '/VisualSearchVideo.mp4')
    return cv2.VideoCapture(vid_direc + participant + '/VisualSearchVideo.mp4')

def close_video(video):
    video.release()
    cv2.destroyAllWindows()

def get_frame(video,i):

    video.set(cv2.CAP_PROP_POS_FRAMES,i)
    return get_next_frame(video)

def reset_video(video):
    video.set(cv2.CAP_PROP_POS_FRAMES,0)

def get_next_frame(video):
    ret,frame = video.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame

def get_patch(im,x,y,s):
    """
    Returns a rectangular patch of an image im, around coordinates x,y with size 2*s x 2*s.
    The image needs to have size 1080 x 1000 x 3, and the values should be RGB data o na scale from 0 to 255
    Note: The output of get_frame(video,i) meets these requirements
    """
    print(x,y,s)

    #check for nan inputs, which is useful when computing patches around a person's gaze
    #since gaze is sometimes undefined when a person is blinking or their gaze lands outside the image
    #or is otherwise set to NaN in an earlier filtering stage
    if np.isnan(x) or np.isnan(y):
        return np.full(shape=(2*s,2*s,3),fill_value=np.nan)

    #Interpolate (linearly) to deal with non-integer patch size or pixel coordinates
    #Note that we have to invert the y-axis since images are defined to have the top-left as (0,0),
    #whereas we compute gaze coordinates with (0,0) at the bottom left
    return np.stack([scipy.interpolate.interp2d(np.arange(0,1080,1),
                                                np.arange(0,1000,1)[::-1],
                                                im[:,:,i], kind='linear',fill_value=np.nan)(np.linspace(x-s,x+s,2*s),np.linspace(y-s,y+s,2*s))
                     for i in range(3)],axis=2)/255

def process_timestamp(s):
    """
    Converts a timestamp as returned by Unity when calling
    int.Parse(DateTime.Now.ToString("yyyyMMddHHmmssfff"));
    into a Python datetime object

    Caveats to look out for:
    - This functions breaks when using a different time format string in Unity
    - This function does not correct for time zone differences
    - Note that Python datetime objects have milliseconds as their final field,
    whereas our Unity convention has seconds. Hence the magical number 1000
    """
    return dt.datetime(int(s[0:4]),int(s[4:6]),int(s[6:8]),int(s[8:10]),
                       int(s[10:12]),int(s[12:14]),int(s[-3:])*1000).timestamp()

def process_element(element,attributename, default):
    """
    Returns an attribute from an element in an xmltree (as returned by Tobii's
    Unity plugin), and performs minor processing if necessary.
    Returns default whenever the element is invalid (as indicated by a field
    "Valid = False" in the element) which Tobii inserts if it failed to track
    the user's eyes
    """
    if attributename not in element.attrib or ("Valid" in element.attrib and element.attrib["Valid"]=="False"):
        return default
    if attributename == "TimeStamp":
        return process_timestamp(element.attrib[attributename])
    if attributename in element.attrib:
        return element.attrib[attributename] if default == "" else ast.literal_eval(element.attrib[attributename])

def get_list_from_xmltree(xmltree, attributename, filterstring,default):
    """
    Returns a list or numpy array of a single attribute from each element
    in an xmltree (as returned by Tobii's Unity plugin).
    We do this because performing further computations using list comprehension
    or numpy array operations is generally faster and easier to write
    """
    if default == "":
        return [process_element(element,attributename, default) for element in xmltree.findall('./GazeData' + filterstring)]
    else:
        return np.array([process_element(element,attributename, default) for element in xmltree.findall('./GazeData' + filterstring)])

def get_forward(A):
    """
    Computes the forward direction of the user given a quaternion that indicates
    the user's orientation. Does so by transforming the vector [0,0,1] in
    the user's reference frame to world coordinates
    """
    return np.hstack([qt.rotate_vectors(q,np.array([0,0,1])) for q in A])

def get_event_times(lines,event_name):
    """
    Finds all the times that a specific event happened in the behavioral data
    file log
    """
    return [ts for ts,line in lines if event_name in line[0]]

def get_event_times_plus_vector(lines,event_name):
    """
    Finds all the times that a specific event happened, along with a vector
    with information about the event. An example event might be
    Time :20180627112857655, Target instantiated at (-21.999850, 1.794788, 0.001362)
    """
    return [(ts,ast.literal_eval(line[0][line[0].find('(',line[0].find(event_name)):
                                 line[0].find(')',line[0].find(event_name))+1]))
            for ts,line in lines if event_name in line[0]]

def find_substring(s):
    """
    Searches through the string s for the first instance of a substring
    delimited by quotation marks ("") on either end
    """
    first = s.find('\"')
    last = s.find('\"',first+1)
    return s[first+1:last]

def get_event_times_plus_string(lines,event_name):
    """
    Finds all the times that a specific event happened, along with a string
    with information about the event. An example event might be
    Time :20180627112842478, Trial 17 with name "Bedroom Location 5 Trial 1" loaded
    """
    return [(ts,find_substring(line[0])) for ts,line in lines if event_name in line[0]]

def get_event_times_plus_line(lines,event_name):
    """
    Finds all the times that a specific event happened, along with the entire line
    in the xdf file that contains the information for that event
    """
    return [(ts,line[0]) for ts,line in lines if event_name in line[0]]

def get_aligned_data(event_times,tpre,tpost,n,d,key):
    """
    Aligns data in dataframe d[key] in a specific column
    by the events in event_times. Specifically, returns an N-by-n numpy array,
    with one row for each event, and on that row the data in d[key][:,column]
    from tpre before the event to tpost after the event. The data is resampled
    and interpolated to ensure equal sampling rates

    Primarily used to plot averaged aligned data, such as average Pupil diameter
    before/after a trial start event
    """
    return np.array([interp_multi(np.linspace(et-tpre,et+tpost,n),d['TimeStamp'],d[key])
                     for et in event_times])

def fix_lines(lines, startline , endline):
    """
    Data cleaning function that corrects for improperly saved Eye tracking files.
    Sometimes these files do not contain a header, or they contain two headers,
    as the eye tracker was restarted after calibration.
    """
    startind = 0
    endind = len(lines)
    for i,line in enumerate(lines):
        if line[1][0].startswith("<?xml") or (line[1][0].startswith("</Data>") and i<len(lines)-1):
            startind = i+1
    if lines[-1][1][0].startswith("</Data>"):
        endind = len(lines)-1
    print(startline,endline)
    return lines[startind:endind]

def parse_condition(s):
    """
    Data formatting function which maps the logged information at the begginging
    of each block to the corresponding block type (static, dynamic or no)
    """
    if s == 'Blocktype: RecommendationsOn = False, AdaptiveOn = False':
        return 'No'
    elif s == 'Blocktype: RecommendationsOn = True, AdaptiveOn = False':
        return 'Static'
    elif s == 'Blocktype: RecommendationsOn = True, AdaptiveOn = True':
        return 'Dynamic'
    else:
        return ''

def get_condition_order(d):
    """
    Helper function which returns a list of the block type for each of the 300
    trials. This is needed since trials appear in 12 blocks of 25, and each
    block has 3 sublocks of 8 trials/condition plus one choose-your-condition trial
    """
    blocks = sum([[(t,parse_condition(s))]*8 for t,s in d['Blocktype']],[])
    chosen = sum([[(t,name) for t in d[name + 'AssistTimes']] for name in ['No','Static','Dynamic']],[])
    return [c for t,c in sorted(blocks + chosen)]

def load_eye_tracking_data(x):
    """
    No comments - data formats are explained in the quip document
    """
    print("Loading Eye Tracking Data")
    startline = "<?xml version=\"1.0\" encoding=\"utf-8\"?><Data>\n"
    endline = "</Data>"
    for outlet in x[0]:
        if outlet['info']['name'][0]=="TobiiData":
            lines = list(zip(outlet['time_stamps'],outlet['time_series']))
    lines = fix_lines(lines, startline, endline)
    xmltree = et.XML(startline + "-".join([line[0] for ts,line in lines]) + endline)
    data = {"lines" : lines,
            "UnityTimeStamp" : get_list_from_xmltree(xmltree, 'TimeStamp','/CombinedGazeRayWorld',np.nan),
            "DeviceTimeStamp" : get_list_from_xmltree(xmltree, 'DeviceTimeStamp','/OriginalGaze',np.nan)/1000000,
            "LeftPupilDiameter" : get_list_from_xmltree(xmltree, 'Value','/Left/PupilDiameter',np.nan)*1000,
            "RightPupilDiameter" : get_list_from_xmltree(xmltree, 'Value','/Right/PupilDiameter',np.nan)*1000,
            "PlayerLocation" : get_list_from_xmltree(xmltree, 'Position','/Pose',[np.nan] *3),
            "PlayerRotation" : get_list_from_xmltree(xmltree, 'Rotation','/Pose',[np.nan] *4),
            "GazeLocation" : get_list_from_xmltree(xmltree, 'Origin','/CombinedGazeRayWorld',[np.nan] *3),
            "GazeDirection" : get_list_from_xmltree(xmltree, 'Direction','/CombinedGazeRayWorld',[np.nan] *3),
            "HitObject" : get_list_from_xmltree(xmltree, 'HitObject','/CombinedGazeRayWorld',""),
            "HitDistance" : get_list_from_xmltree(xmltree, 'HitObjectDistance','/CombinedGazeRayWorld',np.nan)}
    data['GazeDirectionEgocentric'] = rotate_gaze(data['PlayerRotation'],data['GazeDirection'])
    data["LSLTimeStamp"] = [ts for ts,line in lines]
    data["TimeStamp"] = data["LSLTimeStamp"]
    print("Done")
    return data

def load_controller_data(x):
    for outlet in x[0]:
        if outlet['info']['name'][0]=="ControllerData":
            lines = list(zip(outlet['time_stamps'],outlet['time_series']))
    print("Loading Controller Data")
    data = {"lines" : lines,
            "TimeStamp": np.array([ts for ts,line in lines]),
            "Position": np.array([ast.literal_eval(line[0].split('\t')[0]) for ts,line in lines]),
            "Rotation": np.array([ast.literal_eval(line[0].split('\t')[1]) for ts,line in lines]),
            "CameraPosition": np.array([ast.literal_eval(line[0].split('\t')[2]) for ts,line in lines]),
            "CameraRotation": np.array([ast.literal_eval(line[0].split('\t')[3]) for ts,line in lines])}
    data['RotationPolar'] = convert_to_polar(data['Rotation'])
    print("Done")
    return data

def load_behavior_data(x):
    print("Loading Behavior Data")
    for outlet in x[0]:
        if outlet['info']['name'][0]=="BehaviorData":
            lines = list(zip(outlet['time_stamps'],outlet['time_series']))
    data = {"lines" : lines,
            "TimeStamp": np.array([ts for ts,line in lines]),
            "NoAssistTimes" : get_event_times(lines,"No assist button pressed"),
            "StaticAssistTimes" : get_event_times(lines,"Static Assist button pressed"),
            "DynamicAssistTimes" : get_event_times(lines,"Adaptive assist button pressed"),
            "LikeTimes" : get_event_times(lines,"Like button pressed"),
            "DislikeTimes" : get_event_times(lines,"Dislike button pressed"),
            "ControllerUpTimesCorrect" : get_event_times(lines,"Controller button up, correct"),
            "ControllerUpTimesIncorrect" : get_event_times(lines,"Controller button up, incorrect"),
            "ControllerDownTimes" : get_event_times(lines,"Controller button down"),
            "TargetFoundTimes" : get_event_times(lines,"Target pulled to"),
            "TrialStartTimes" : get_event_times(lines,"Target moved inside to"),
            "TrialFinishedTimes" : get_event_times(lines,"Finished"),
            "PlayerOutsideTimes" : get_event_times(lines,"loaded"),
            "ExperimentStartTimes": get_event_times(lines,"Start Experiment"),
            "ExperimentEndTimes": get_event_times(lines,"Experiment End"),
            "Showperformance": get_event_times_plus_line(lines," out of "),
            "PlayerMovedTimes" : get_event_times_plus_line(lines,"Player moved from"),
            "TargetLocation" : get_event_times_plus_vector(lines, 'Target moved inside to '),
            "TrialOrder" : get_event_times_plus_string(lines, 'loaded'),
            "Blocktype" : get_event_times_plus_line(lines, 'Blocktype: '),
            "BlockParams" : get_event_times_plus_line(lines, 'Choose params: '),
            }
    data["NothingFoundTimes"] = [t+8 for t,s in pair_times(data["TrialStartTimes"],data["TargetFoundTimes"] +[np.Infinity]) if s-t>8.5]
    data["SearchOverTimes"] = sorted(data["TargetFoundTimes"] + data["NothingFoundTimes"])
    data["ControllerUpTimes"] = sorted(data["ControllerUpTimesCorrect"] + data["ControllerUpTimesIncorrect"])
    data["ControllerDownDuringTrials"] = [[t for t in data["ControllerDownTimes"] if t>t1 and t<t2]
                                          for t1,t2 in zip(data["TrialStartTimes"],data["TrialFinishedTimes"])]
    data["ResponseTime"] = [s-t for t,s in zip(data["TrialStartTimes"],data["SearchOverTimes"])]
    data["ChooseConditionTimes"] = sum([[t for t,line in data['PlayerMovedTimes'] if t>t1 and t<t2]
                                    for (t1,_),t2 in zip(data['Showperformance'],sorted(data['StaticAssistTimes'] + data['NoAssistTimes'] + data['DynamicAssistTimes']))],[])
    data["Condition"] = get_condition_order(data)
    print("Done")
    return data

def load_highlight_data(x):
    print("Loading Highlight Data")
    for outlet in x[0]:
        if outlet['info']['name'][0]=="RecommendationAgent":
            lines = list(zip(outlet['time_stamps'],outlet['time_series']))
    data = {"TimeStamp" : [ts for ts,line in lines if line[0].startswith('Highlights')],
            "Highlights" : [line[0].replace('Highlights:','') for ts,line in lines if line[0].startswith('Highlights')]}
    print("Done")
    return data

def load_processed_data_from_xdf_one_key(x,key):
    if key=='Behavior':
        return load_behavior_data(x)
    if key=='Controller':
        return load_controller_data(x)
    if key=='Eye':
        return load_eye_tracking_data(x)
    if key=='Highlights':
        return load_highlight_data(x)

def load_processed_data_from_xdf(xdf_data,keys):
    return [{key: load_processed_data_from_xdf_one_key(x,key) for key in keys} for x in xdf_data]

def my_load_xdf(fname):
    """
    Simple wrapper around xdf.load_xdf which also prints the file path
    """
    print(fname)
    return load_xdf(fname, verbose=False, synchronize_clocks=False)

def write_xdf_to_pickle(direc, xdf_data ,participant_list):
    for x,p in zip(xdf_data, participant_list):
        with(open(direc + p + '_pickled_raw.txt','wb')) as f:
            pickle.dump(x,f)
            print(f.name)

def write_processed_data_to_pickle(direc,data,participant_list, keys):
    for d,p in zip(data,participant_list):
        with(open(direc + p + '_pickled_processed.txt','wb')) as f:
            pickle.dump(d,f)
            print(f.name)
        for key in keys:
            with(open(direc + p + '_pickled_processed_' + key + '.txt','wb')) as f:
                pickle.dump(d[key],f)
                print(f.name)

def load_processed_data_from_raw(direc, participant_list, keys):
    xdf_data = [my_load_xdf(direc + 'xdf_data_' + participant + '.xdf') for participant in participant_list]
    write_xdf_to_pickle(direc, xdf_data,participant_list)
    data = load_processed_data_from_xdf(xdf_data, keys)
    write_processed_data_to_pickle(direc, data,participant_list, keys)
    return data

def pickle_load(fname):
    """
    Simple wrapper around pickle.load which also prints the file path
    """
    with open(fname,'rb') as f:
        print(fname)
        return pickle.load(f)

def load_xdf_from_pickle(direc,participant_list):
    return [pickle_load(direc + p + '_pickled_raw.txt') for p in participant_list]

def load_processed_data_from_pickle_one_key(direc, p, key):
    return pickle_load(direc + p + '_pickled_processed_' + key + '.txt')

def load_processed_data_from_pickle(direc, participant_list, keys):
    return [{key: load_processed_data_from_pickle_one_key(direc,p,key) for key in keys} for p in participant_list]

def load_trial(fname):
    with open(fname,'r') as f:
        return f.read().splitlines()

def update_data(direc,participant_list,xdf_data,data,keys,redump):
    """
    reloads data for a set of key, and saves all pickled files.
    Useful when changing code in this notebook.
    """

    for x,d in zip(xdf_data,data):
        for key in keys:
            d[key]=load_processed_data_from_xdf_one_key(x,key)
    if redump:
        write_processed_data_to_pickle(direc, data, participant_list, keys)

def interp_multi(x, xp, fp):
    """
    Slightly wordy wrapper around np.interp() to allow for inputs to contain
    NaNs (which will be ignored), and to pass arrays, in which case the
    interpolation happens independently on each column. This is old research code,
    and for future applications I would recommend using scipy.interp instead
    """
    if len(fp.shape)>1:
        return np.array([np.interp(x, np.array(xp)[~np.isnan(fp[:,i])], fp[~np.isnan(fp[:,i]),i]) for i in range(fp.shape[1])]).T
    else:
        return np.array(np.interp(x, np.array(xp)[~np.isnan(fp)], fp[~np.isnan(fp)]))

def parse_camera_location(s):
    return np.array(ast.literal_eval(s[0])),np.array(ast.literal_eval(s[1]))

def load_camera_location(direc):
    """
    Loads the file CameraLocations.txt in the provided directory and parses it into
    a formatted structure, namely a dictionary with names as keys, and location plus
    orientation for each camera name. For example:
    {'Kitchen Location 1': (array([5.013, 1.7  , 4.636]), array([ 0.      , -0.997287,  0.      , -0.073618]))}
    """
    with open(direc + 'CameraLocations.txt') as f:
        return {" ".join(line.split('\t')[0:2]):parse_camera_location(line.split('\t')[2:4]) for line in f.read().splitlines()}

def get_gaze_direction(d,t):
    """
    Resamples gaze direction at provided time intervals. useful for computing gaze
    coordinates for each screenshot/frame of the experiment movies
    """
    return scipy.interpolate.interp1d(d['Eye']['TimeStamp'],d['Eye']['GazeDirection'],kind='previous',axis=0)(t)

def get_head_direction(d,t):
    """
    Resamples player location (specifically head location) at provided time intervals.
    Head location is present in both eye and controller data, this function uses the most
    recent stream for each time provided.
    """
    return scipy.interpolate.interp1d(np.hstack([d['Eye']['TimeStamp'],
                                                 d['Controller']['TimeStamp']]),
                                      np.vstack([d['Eye']['PlayerRotation'],
                                                 d['Controller']['CameraRotation']]),kind='previous',axis=0)(t)

def get_sorted_messages(direc,p):
    filename = "replay_messages_" + p + "_pickled.txt"
    with open(direc + filename,"rb") as f:
        print(filename)
        return pickle.load(f)

def load_screen_coords(direc,participant):
    """
    Load a file with the gaze coordinates for each screenshot
    """
    with open(direc + participant + '/Screenshot_coords.txt','r') as f:
        return np.loadtxt(map(lambda each:each.split("\t")[-1].strip(")").strip("("), f.read().splitlines()),delimiter=',')

def get_target_object(direc,trialname):
    """
    Returns the name of the target object on the trial with the provided name.
    """
    with open(direc + trialname+ '.txt','r') as f:
        num_header_lines = 5 #Number of lines in the text files wich contain metadata but no objects. Fixed when generating these files in Unity
        lines=f.readlines()

        return lines[num_header_lines].split('\t')[-1].strip().replace('(Clone)','')

def get_all_objects(direc,trialname):
    """
    Returns the name of all the objects on the trial with the provided name.
    """
    with open(direc + trialname+ '.txt','r') as f:
        num_header_lines = 5 #Number of lines in the text files wich contain metadata but no objects. Fixed when generating these files in Unity
        lines=f.readlines()

        all_objects = []
        for l in lines[num_header_lines:]:
            all_objects.append(l.split("\t")[-1].strip().replace('(Clone)',''))
        return all_objects

def get_object_locations(direc,trialname):
    """
    Returns location of all the objects on the trial with the provided name.
    """
    with open(direc + trialname+ '.txt','r') as f:
        num_header_lines = 5 #Number of lines in the text files wich contain metadata but no objects. Fixed when generating these files in Unity
        lines=f.readlines()

        object_locations = []
        for l in lines[num_header_lines:]:
            object_locations.append(l.split("\t")[0])
        return object_locations

def load_scene_info(direc, trialname):
    """
    Loads a text file containing scene information and returns data as a pandas dataframe.
    """

    with open(direc + trialname+ '.txt','r') as f:
        num_header_lines = 5 #Number of lines in the text files wich contain metadata but no objects. Fixed when generating these files in Unity
        lines=f.readlines()
    
    lines = lines[num_header_lines:]

    location = []
    rotation = []
    scale = []
    obj = []

    for l in np.arange(len(lines)):
        location.append(np.array(literal_eval(lines[l].split("\t")[0])))
        rotation.append(np.array(literal_eval(lines[l].split("\t")[1])))
        scale.append(np.array(literal_eval(lines[l].split("\t")[2])))
        obj.append(lines[l].split("\t")[3].strip().replace('(Clone)',''))
        
    scene_info = pd.DataFrame(np.concatenate((np.vstack(location), np.vstack(rotation)), axis=1))
    scene_info.columns = ['loc_x', 'loc_y', 'loc_z', 'rot_x', 'rot_y', 'rot_z', 'rot_w']
    scale = np.vstack(scale)
    scene_info['scale'] = scale[:,0]
    scene_info['object'] = np.vstack(obj)

    return scene_info


    
