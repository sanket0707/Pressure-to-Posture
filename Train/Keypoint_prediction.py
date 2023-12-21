#Setup Environment
import platform
import os

print(platform.system())

from google.colab import drive
drive.mount('/google_drive', force_remount=True)
ppit_folder = "/google_drive/Shareddrives/Grants & Competitions/Innovate UK_Smart Grant 2022"
kingston_data_folder = os.path.join(ppit_folder, "Practice Data Gathering Day/Kingston Data")
ml_api = "/google_drive/My Drive/Python_Projects/ml_PPIT"

import pandas as pd
import numpy as np
import os
import sys
import plotly
import pickle
import yaml


#Visualise predicted keypoint from model
body = [ [13, 11 , 9, 10, 12, 14], # arm joints
         [ 1, 3, 5, 7, 8, 6, 4,2], #leg joints
         [9, 7, 8 ,10], #torso joints
        ]

#import data
#config for the coords

################################################################################
# change this to your own data path
################################################################################
submodel = "uniform_loss"
data_type = "val"

################################################################################
# change this to your own data path
################################################################################
original_data_path = os.path.join(os.path.dirname(config_file_path), 'model_20231019_P27', submodel, '%s_data_batch1.p'%data_type)
with open(original_data_path, 'rb') as f:
  target_data = pickle.load(f)  # shape (num_batch, batch_number, joint, xyz)
joint_data = np.vstack([i[1] for i in target_data])[0000:1000] * 32767  # shape (frame, joint, xyz)
# joint_data = np.expand_dims(joint_data, axis=-1) * 32767


################################################################################
# change this to your own data path
################################################################################
arm_data = np.swapaxes([joint_data[:, i-1, :] for i in body[0]], 0, 1).astype(int)
arm_data = np.swapaxes(arm_data, 1, 2).astype(int)
leg_data = np.swapaxes([joint_data[:, i-1, :] for i in body[1]], 0, 1).astype(int)
leg_data = np.swapaxes(leg_data, 1, 2).astype(int)
torso_data = np.swapaxes([joint_data[:, i-1, :] for i in body[2]], 0, 1).astype(int)
torso_data = np.swapaxes(torso_data, 1, 2).astype(int)
#torso_data = np.swapaxes(torso_data, 1, 2).astype(int)

arm_data.shape

#Config for predicted coords
################################################################################
# change this to your own prediction data
################################################################################
prediction_path = os.path.join(os.path.dirname(config_file_path), 'model_20231019_P27',
                               submodel, 'predicted_result.p')
with open(prediction_path, 'rb') as f:
  predicted_data = pickle.load(f)  # shape (frame, joint, xyz)

################################################################################
# change this to your own prediction data
################################################################################
predicted_data[data_type] = np.squeeze(predicted_data[data_type])
print(predicted_data[data_type].shape)

################################################################################
# change this to your own prediction data
################################################################################
# heatmap_data = data['heatmap'][100:102]
predicted_joint_data = predicted_data[data_type][0000:1000] * 32767
print(predicted_joint_data.shape)
print(np.max(predicted_joint_data))

################################################################################
# change this to your own prediction data
################################################################################
predicted_arm_data = np.swapaxes([predicted_joint_data[:, i-1, :] for i in body[0]], 0, 1).astype(int)
predicted_arm_data = np.swapaxes(predicted_arm_data, 1, 2).astype(int)
predicted_leg_data = np.swapaxes([predicted_joint_data[:, i-1, :] for i in body[1]], 0, 1).astype(int)
predicted_leg_data = np.swapaxes(predicted_leg_data, 1, 2).astype(int)
predicted_torso_data = np.swapaxes([predicted_joint_data[:, i-1, :] for i in body[2]], 0, 1).astype(int)
predicted_torso_data = np.swapaxes(predicted_torso_data, 1, 2).astype(int)


#Draw Figure

# Define frames
import plotly.graph_objects as go

fig = go.Figure(frames=[go.Frame(
                                    data=[
                                            # go.Surface( x=mat_x,
                                            #             y=mat_y,
                                            #             z=pressure_map_data[k],
                                            #             name='Surface',
                                            #             visible=True
                                            #             ),

                                            go.Scatter3d(
                                                        x=arm_data[k,0,:],
                                                        y=arm_data[k,1,:],
                                                        z=arm_data[k,2,:],
                                                        marker=dict(
                                                                    size=0,
                                                                    color='black',
#                                                                     colorscale='Inferno',
                                                                ),
                                                        line=dict(
                                                                    width=5,
                                                                    color='black',
#                                                                     colorscale='Inferno'
                                                                 )
                                                        ),
                                            go.Scatter3d( # joint marker
                                                         x=leg_data[k,0,:],
                                                         y=leg_data[k,1,:],
                                                         z=leg_data[k,2,:],
                                                        marker=dict(
                                                                    size=0,
                                                                    color='black',
#                                                                     colorscale='Inferno',
                                                                ),
                                                        line=dict(
                                                                    width=5,
                                                                    color='black',
#                                                                     colorscale='Inferno'
                                                                 )
                                                        ),
                                            go.Scatter3d( # joint marker
                                                         x=torso_data[k,0,:],
                                                         y=torso_data[k,1,:],
                                                         z=torso_data[k,2,:],
                                                        marker=dict(
                                                                    size=0,
                                                                    color='black',
#                                                                     colorscale='Inferno',
                                                                ),
                                                        line=dict(
                                                                    width=5,
                                                                    color='black',
#                                                                     colorscale='Inferno'
                                                                 )
                                                        ),
                                            go.Scatter3d(
                                                        x=predicted_arm_data[k,0,:],
                                                        y=predicted_arm_data[k,1,:],
                                                        z=predicted_arm_data[k,2,:],
                                                        marker=dict(
                                                                    size=0,
                                                                    color='red',
#                                                                     colorscale='Inferno',
                                                                ),
                                                        line=dict(
                                                                    width=5,
                                                                    color='red',
#                                                                     colorscale='Inferno'
                                                                 )
                                                        ),
                                            go.Scatter3d( # joint marker
                                                         x=predicted_leg_data[k,0,:],
                                                         y=predicted_leg_data[k,1,:],
                                                         z=predicted_leg_data[k,2,:],
                                                        marker=dict(
                                                                    size=0,
                                                                    color='blue',
#                                                                     colorscale='Inferno',
                                                                ),
                                                        line=dict(
                                                                    width=5,
                                                                    color='blue',
#                                                                     colorscale='Inferno'
                                                                 )
                                                        ),
                                            go.Scatter3d( # joint marker
                                                         x=predicted_torso_data[k,0,:],
                                                         y=predicted_torso_data[k,1,:],
                                                         z=predicted_torso_data[k,2,:],
                                                        marker=dict(
                                                                    size=0,
                                                                    color='green',
#                                                                     colorscale='Inferno',
                                                                ),
                                                        line=dict(
                                                                    width=5,
                                                                    color='green',
#                                                                     colorscale='Inferno'
                                                                 )
                                                        ),
#                                             go.Scatter3d(
#                                                          x=predicted_joint_data[k,0,:],
#                                                          y=predicted_joint_data[k,1,:],
#                                                          z=predicted_joint_data[k,2,:],
#                                                             marker=dict(
#                                                                         size=10,
#                                                                         color='black'
#                                                                     ),

#                                                             mode='markers'
#                                                         ),


                                        ]
    ,
                                    name=str(k) # you need to name the frame for the animation to behave properly
                                )
                        for k in range(len(joint_data))])

# Add data to be displayed before animation starts
# fig.add_trace(go.Surface(
#                             x=mat_x,
#                             y=mat_y,
#                             z=pressure_map_data[0],
#                             cmin=0,
#                             cmax=100
#              ))


fig.add_trace(go.Scatter3d(
                            x=arm_data[0, 0, :],
                            y=arm_data[0, 1, :],
                            z=arm_data[0, 2, :],
                            marker=dict(
                                        size=0,
                                        color='black',
#                                                                     colorscale='Inferno',
                                    ),
                            line=dict(
                                        width=5,
                                        color='black',
#                                                                     colorscale='Inferno'
                                     )
                            ),
             )


fig.add_trace(go.Scatter3d( # joint marker
                             x=leg_data[0, 0, :],
                             y=leg_data[0, 1, :],
                             z=leg_data[0, 2, :],
                            marker=dict(
                                        size=0,
                                        color='black',
    #                                                                     colorscale='Inferno',
                                    ),
                            line=dict(
                                        width=5,
                                        color='black',
    #                                                                     colorscale='Inferno'
                                     )
                            ),
             )
fig.add_trace(go.Scatter3d( # joint marker
                             x=torso_data[0, 0, :],
                             y=torso_data[0, 1, :],
                             z=torso_data[0, 2, :],
                            marker=dict(
                                        size=0,
                                        color='black',
#                                                                     colorscale='Inferno',
                                    ),
                            line=dict(
                                        width=5,
                                        color='black',
#                                                                     colorscale='Inferno'
                                     )
                            ),
             )


fig.add_trace(go.Scatter3d(
                            x=predicted_arm_data[0, 0, :],
                            y=predicted_arm_data[0, 1, :],
                            z=predicted_arm_data[0, 2, :],
                            marker=dict(
                                        size=0,
                                        color='red',
#                                                                     colorscale='Inferno',
                                    ),
                            line=dict(
                                        width=5,
                                        color='red',
#                                                                     colorscale='Inferno'
                                     )
                            ),
             )


fig.add_trace(go.Scatter3d( # joint marker
                             x=predicted_leg_data[0, 0, :],
                             y=predicted_leg_data[0, 1, :],
                             z=predicted_leg_data[0, 2, :],
                            marker=dict(
                                        size=0,
                                        color='blue',
    #                                                                     colorscale='Inferno',
                                    ),
                            line=dict(
                                        width=5,
                                        color='blue',
    #                                                                     colorscale='Inferno'
                                     )
                            ),
             )
fig.add_trace(go.Scatter3d( # joint marker
                             x=predicted_torso_data[0, 0, :],
                             y=predicted_torso_data[0, 1, :],
                             z=predicted_torso_data[0, 2, :],
                            marker=dict(
                                        size=0,
                                        color='green',
#                                                                     colorscale='Inferno',
                                    ),
                            line=dict(
                                        width=5,
                                        color='green',
#                                                                     colorscale='Inferno'
                                     )
                            ),
             )

def frame_args(duration):
    return {
            "frame": {"duration": duration},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"},
        }

sliders = [
            {
                "pad": {"b": 10, "t": 60},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": [
                    {
                        "args": [[f.name], frame_args(0)],
                        "label": str(k),
                        "method": "animate",
                    }
                    for k, f in enumerate(fig.frames)
                ],
            }
        ]

# Layout
fig.update_layout(
         title='body movement',
         width=600,
         height=500,
#         paper_bgcolor='black',
#         plot_bgcolor='rgba(0,0,0,0)',
         scene=dict(
                    zaxis=dict(
                            range=[-40, 32747],
                              autorange=False,
# #                                color='white',
#                               ticksuffix = "%"
                                ),
                    aspectratio=dict(x=2, y=2, z=1),
                    xaxis=dict(
                            range=[-32747, 32747],
                              autorange=False,
#                         showgrid=False,
#                               showticklabels=False,
                    ),

#
                    yaxis=dict(
                            range=[-32747, 32747],
                              autorange=False,
#                             showgrid=False,
#                               showticklabels=False,
                    ),
                    xaxis_title = 'x',
                    yaxis_title = 'y',
                    zaxis_title = 'z',
                    ),
         updatemenus = [
            {
                "buttons": [
                    {
                        "args": [None, frame_args(50)],
                        "label": "&#9654;", # play symbol
                        "method": "animate",
                    },
                    {
                        "args": [[None], frame_args(0)],
                        "label": "&#9724;", # pause symbol
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 70},
                "type": "buttons",
                "x": 0.1,
                "y": 0,
            }
         ],
         sliders=sliders,
        showlegend=False,
)

fig.show()
fig.write_html(os.path.join(os.path.dirname(config_file_path),
                            'model_20231019_P27', submodel,
                            "%s_body_movement_visualisation_1.html"%data_type)
            )
