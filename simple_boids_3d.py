#%%
import numpy as np 
import pandas as pd 
import plotly.express as px 

from sklearn.neighbors import *

from tqdm import tqdm, trange

#%% Defining parameters
N = 1000

# We will constrain positions to unit sphere
positions = np.random.uniform(0, 1, (N, 3))
positions = positions / np.linalg.norm(positions, axis=1).repeat(3).reshape(positions.shape)

directions = np.random.random(((N, 3)))

# Projecting into tangent space (i.e. removing normal component)
directions - np.array([position @ direction for position, direction in zip(positions, directions)]).repeat(3).reshape(positions.shape) * positions

# normalizing directions
directions = directions / np.linalg.norm(directions, axis=1).reshape(-1, 1).repeat(3, axis=1)

top_speeds = np.random.choice([1, 0.6], N)

dt = 0.01
T = np.arange(0, 10, dt)

# Collision physics
collision_r = 0.03 
collision_epsilon = 0.1

# Boundary parameters
boundary_r = 0.1
boundary_epsilon = 0.3


# Running simulation
r = 0.1
historical_positions = []
for t in tqdm(T) : 
    historical_positions.append(positions)

    # Get neighbors 
    tree = BallTree(positions)
    neighbors = tree.query_radius(positions, r)

    # Get mean directions
    mean_directions = np.array([directions[idx].mean(axis=0) for idx in neighbors])

    # NOTE: Could encounter divide by zero error
    directions = directions + mean_directions / np.linalg.norm(mean_directions, axis=1).reshape(-1, 1).repeat(3, axis=1)
    directions = directions / np.linalg.norm(directions, axis=1).reshape(-1, 1).repeat(3, axis=1)

    # Get collisions 
    collision_neighbors = tree.query_radius(positions, collision_r)
    mean_coll_directions = [(positions[ids] - positions[i]).mean(axis=0) if len(ids) > 0 else 0j for i, ids in enumerate(collision_neighbors)]
    normalized_coll_directions = np.array([dir/np.linalg.norm(dir) if np.linalg.norm(dir) != 0 else dir for dir in mean_coll_directions])
    directions = directions - normalized_coll_directions * collision_epsilon
    
    positions = (positions + directions * dt) 

    positions = positions = positions / np.linalg.norm(positions, axis=1).repeat(3).reshape(positions.shape)
# %% Plotting
plotdf = pd.DataFrame(columns=['t', 'n', 'x', 'y', 'z'])

for t, pos in enumerate(historical_positions) : 
    dat = pd.DataFrame(columns=['t', 'n', 'x', 'y', 'z'])
    dat['t'] = [t] * N
    dat['n'] = np.arange(N) 
    dat['x'] = pos[:, 0]
    dat['y'] = pos[:, 1]
    dat['z'] = pos[:, 2]

    plotdf = pd.concat([plotdf, dat])

fig = px.scatter_3d(
    plotdf, x='x', y='y', z='z',
    opacity=0.1,
    animation_frame='t', 
    range_x = [0,1], 
    range_y = [0,1],
    range_z = [0,1],
    template='plotly_dark'
    )

fig.layout['updatemenus'][0]['buttons'][0]['args'] = (None,
        {'frame': {'duration': 0.0, 'redraw': False},
        'mode': 'immediate',
        'fromcurrent': True,
        'transition': {'duration': 0, 'easing': 'linear'}})
'''
fig.update_layout(
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
    zaxis=dict(visible=False)
)
'''
fig.show()

# %%
