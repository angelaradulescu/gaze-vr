
# This library contains functions for visualizing gaze data from the visual search in VR experiment.
#
# Authors: Bas van Opheusden, Angela Radulescu

import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt

def plot_gaze_episode(im, gaze, target_coords, load_fixations, connect_points=True):

    """
    Plots eye traces for one episode overlaid on a scene image. Can be used as a general purpose
    2D fixation plotting function. 
    """

    ## Get enough colors. 
    n_samples = gaze.shape[0]
    cm_subsection = np.linspace(0.0, 1.0, n_samples)
    colors = [cm.plasma(x) for x in np.linspace(0.0, 1, n_samples)]

    fig = plt.figure()
    fig.set_size_inches(16,8)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    fig.add_axes(ax)
    ax.imshow(im)

    # Plot target. 
    ax.plot(target_coords[0],target_coords[1],'.',markersize=60, color='green', alpha=0.75, markeredgecolor='white')

    for i, color in enumerate(colors):

        # Plot point. 
        if load_fixations: 
            ax.plot(gaze[i,0],gaze[i,1],'.',markersize=20,color=color, fillstyle='none', markeredgewidth=4)
        else: 
            ax.plot(gaze[i,0],gaze[i,1],'.',markersize=15,color=color, markeredgewidth=4)
        
        if connect_points: 
            if i < len(colors)-1:
                x = [gaze[i,0], gaze[i+1,0]]
                y = [gaze[i,1], gaze[i+1,1]]
                ax.plot(x, y,'-', color=color, linewidth=3)

    ax.set_axis_off();

def plot_gaze_episode_stepwise(im, gaze, target_coords, fig_path, connect_points=True):

    """
    Plots eye traces for one episode overlaid on a scene image. Can be used as a general purpose
    2D fixation plotting function. 
    """

    ## Get enough colors. 
    n_samples = gaze.shape[0]
    cm_subsection = np.linspace(0.0, 1.0, n_samples)
    colors = [cm.plasma(x) for x in np.linspace(0.0, 1, n_samples)]

    for i, color in enumerate(colors):

        fig = plt.figure()
        fig.set_size_inches(16,8)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        fig.add_axes(ax)
        ax.imshow(im)

        # Plot target. 
        ax.plot(target_coords[0],target_coords[1],'.',markersize=80, color='green', alpha=0.8, markeredgecolor='white')

        # Plot point. 
        ax.plot(gaze[i,0],gaze[i,1],'.',markersize=40,color=color, markeredgewidth=4, markeredgecolor='white');
        # ax.plot(gaze[i,0],gaze[i,1],'ow',markersize=15,fillstyle='none', markeredgewidth=4);
        
        if connect_points: 
            if i < len(colors)-1:
                x = [gaze[i,0], gaze[i+1,0]]
                y = [gaze[i,1], gaze[i+1,1]]
                ax.plot(x, y,'-', color=colors[i+1], linewidth=3)
                # ax.plot(x, y,'.',markersize=20,color=colors[i+1], fillstyle='none', markeredgewidth=4);
                ax.plot(x, y,'.',markersize=40,color=colors[i+1]);
                # ax.plot(x, y,'ow',markersize=15,fillstyle='none', markeredgewidth=4);

        ax.set_axis_off()
        plt.savefig(fig_path + 'sim_t_' + str(i+1) + '.png');

def plot_gaze_from_camera_on_equirects(gaze_from_camera_polar,nonextreme_inds):
    """
    Plots eye traces for all participants on top of the equirect images for each trial. 
    Deprecated, as it uses the non-processed data format, keeping as legacy code. 
    """
    for c in camera_locations.keys():
        for n in range(1,11):
            #loop over all trials (camera location and trial number from 1-10)
            trial_name = c + ' Trial ' + str(n)
            print(trial_name)

            #load equirect for this trial
            im = imageio.imread(equirect_direc + trial_name + '.png')

            fig = plt.figure()
            fig.set_size_inches(8,4)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            fig.add_axes(ax)
            #plot the equirect image
            ax.imshow(im,extent=[-np.pi,np.pi,-np.pi/2,np.pi/2],aspect='auto')
            #initialize all traces, this is used to save trajectories into a text file "Fixation coords ....txt"
            allg = np.empty(shape=(0,3))

            #loop over all participants
            for j,(d,g,valid_inds) in enumerate(zip(data,gaze_from_camera_polar,nonextreme_inds)):
                #find the trial number where participant j performed the trial with name trial_name
                ind = [i for i,(_,name) in enumerate(d["Behavior"]["TrialOrder"]) if name==trial_name]

                #replace all "extreme" events with nans, thus eliminates tracking errors
                #For times where the eye tracker could not locate gaze, g already contains NaNs
                g[~valid_inds,:]=np.nan

                #this loop should only have 1 entry
                for i in ind:
                    t1 = d['Behavior']['TrialStartTimes'][i]
                    t2 = d['Behavior']['SearchOverTimes'][i]

                    #Find gaze trajectory of participant j on the correct trial
                    gtrial = g[np.logical_and(d['Eye']['TimeStamp']>t1,d['Eye']['TimeStamp']<t2)]

                    #plot the trajectory
                    ax.plot(gtrial[:,0],gtrial[:,1],'-',alpha=0.5,linewidth=0.5,color='C'+str(j%10))
                    ax.plot(gtrial[:,0],gtrial[:,1],'.',alpha=0.1,markersize=4,color='C'+str(j%10))

                    #add the trajectory to the collection of all trajectories
                    allg = np.vstack([allg,np.hstack([np.full([gtrial.shape[0],1],fill_value=j),gtrial])])

            #Before saving trajectories, remove all nans (invalid entries) and rescale to pixel space
            allg = (allg[~np.any(np.isnan(allg),axis=1),:]+[0,np.pi,np.pi/2])*[1,4096/np.pi,4096/np.pi]
            #currently saving is commented out, this only plots the figures in the notebook
            #np.savetxt(equirect_direc + 'Fixation coords ' + trial_name + '.txt',allg,fmt='%i')
            ax.set_axis_off()
            #fig.savefig(Screenshot_direc + 'Fixations ' + trial_name + '.png',transparent=True, pad_inches=0,dpi=8192/8)
            plt.show()
