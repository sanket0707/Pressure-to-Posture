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


# Create the data for the 3D skeleton, integer refers to the index of joint
body = [ [5, 3, 1, 2, 4, 6], # arm joints
         [ 13, 11, 9, 7, 8, 10, 12, 14], #leg joints
         [7, 1, 2, 8], #torso joints
        ]

#Visualise predicted result from model
#import data

config_file_path = os.path.join(ppit_folder, 'Software', 'machine_learning', 'Database', 'processed', '20230413', 'config.yml')

config_path = os.path.join(os.path.dirname(config_file_path), 'process_log.yml')
with open(config_path, 'rb') as f:
  data_log = yaml.safe_load(f)
  
original_data_path = os.path.join(os.path.dirname(config_file_path), 'model_20230616_P2', 'target_data_corresponds_to_prediciton.p')
with open(original_data_path, 'rb') as f:
  target_data = pickle.load(f)  # shape (num_batch, batch_number, joint, xyz)
target_data = np.vstack(target_data) # shape (frame, joint, xyz)

target_data.shape

prediction_path = os.path.join(os.path.dirname(config_file_path), 'model_20230616_P2', 'checkpoint05-14.36', 'predicted_result.p')
with open(prediction_path, 'rb') as f:
  predicted_data = pickle.load(f)  # shape (frame, joint, xyz)


predicted_data['keypoint'].shape

# heatmap_data = data['heatmap'][100:102]
joint_data = np.squeeze(predicted_data['keypoint'])
joint_data.shape

joint_data *= 32767


# arm_data = np.swapaxes([joint_data[:, i-1, :] for i in body[0]], 0, 1).astype(int)
# arm_data = np.swapaxes(arm_data, 1, 2).astype(int)
# leg_data = np.swapaxes([joint_data[:, i-1, :] for i in body[1]], 0, 1).astype(int)
# leg_data = np.swapaxes(leg_data, 1, 2).astype(int)
# torso_data = np.swapaxes([joint_data[:, i-1, :] for i in body[2]], 0, 1).astype(int)
# torso_data = np.swapaxes(torso_data, 1, 2).astype(int)
# # leg_data = joint_data[:, body[1], :]
# # torso_data = joint_data[:, body[2], :]

y, x, z = [np.linspace(-32767., 32767., int(heatmap_data.shape[-3:][i])) \
            for i in range(3)]
pos_xyz = np.meshgrid(x, y, z)
joint=4



*******************Draw Figure*********************************

# Define frames
import plotly.graph_objects as go

fig = go.Figure(frames=[go.Frame(
                                    data=[

                                          # go.Volume(
                                          #           x=pos_xyz[0].ravel(),
                                          #           y=pos_xyz[1].ravel(),
                                          #           z=pos_xyz[2].ravel(),
                                          #           value=heatmap_data[k,joint].ravel(),
                                          #           opacity=0.3,
                                          #           surface_count=15,
                                          #           colorscale='Blues',

                                          #       ),


                                            go.Scatter3d(
                                                        x=arm_data[k,0,:],
                                                        y=arm_data[k,1,:],
                                                        z=arm_data[k,2,:],
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
                                                         x=leg_data[k,0,:],
                                                         y=leg_data[k,1,:],
                                                         z=leg_data[k,2,:],
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
                                                         x=torso_data[k,0,:],
                                                         y=torso_data[k,1,:],
                                                         z=torso_data[k,2,:],
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
                                            go.Scatter3d(
                                                         x=joint_data[k,0,:],
                                                         y=joint_data[k,1,:],
                                                         z=joint_data[k,2,:],
                                                            marker=dict(
                                                                        size=10,
                                                                        color='black'
                                                                    ),

                                                            mode='markers'
                                                        ),


                                        ]
    ,
                                    name=str(k) # you need to name the frame for the animation to behave properly
                                )
                        for k in range(len(joint_data))])

# Add data to be displayed before animation starts



fig.add_trace(go.Scatter3d(
                            x=arm_data[0, 0, :],
                            y=arm_data[0, 1, :],
                            z=arm_data[0, 2, :],
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
                             x=leg_data[0, 0, :],
                             y=leg_data[0, 1, :],
                             z=leg_data[0, 2, :],
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
                             x=torso_data[0, 0, :],
                             y=torso_data[0, 1, :],
                             z=torso_data[0, 2, :],
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

# fig.add_trace(go.Volume(
#                         x=pos_xyz[0].ravel(),
#                         y=pos_xyz[1].ravel(),
#                         z=pos_xyz[2].ravel(),
#                         value=heatmap_data[0,joint].ravel(),
#                         opacity=0.1,
#                         surface_count=15,
#                         colorscale='Blues',

#                     )
#              )



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
         width=1000,
         height=1000,
#         paper_bgcolor='black',
#         plot_bgcolor='rgba(0,0,0,0)',
         scene=dict(
                    zaxis=dict(
                            range=[-32747, 32747],
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
fig.write_html("./body_movement_visualisation.html")
