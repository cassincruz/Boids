#%%
import numpy as np 
import pandas as pd 
import plotly.express as px 

from sklearn.neighbors import *

from tqdm import tqdm, trange

#%% Defining parameters
N = 1000 # Number of boids

# Initializing boids
positions = np.random.uniform(0, 1, (N, 2))
directions = np.exp(1j * np.random.uniform(0, 2 * np.pi, N))

dt = 0.01
T = np.arange(0, 10, dt)

top_speeds = np.ones(N)

# Randomness
dir_epsilon = 0.0
wind_epsilon = 0.1

# Collision physics
collision_r = 0.03 
collision_alpha = 10

# Boundary parameters
HORIZONTAL_BOUNDARIES = False
VERTICAL_BOUNDARIES = True
boundary_r = dt * top_speeds.max() * 10
boundary_alpha = 0.0
boundary_epsilon = 10

# Boid neighbor parameters
r = 0.05
neighbor_alpha = 0.5

# Running simulation

historical_positions = []
for t in tqdm(T) : 
    historical_positions.append(positions)

    # Get neighbors 
    tree = BallTree(positions)
    neighbors = tree.query_radius(positions, r)

    # Get mean directions
    neighbor_directions = np.array([(directions[idx] - directions[i]).mean() for i, idx in enumerate(neighbors)])
    
    directions = directions + neighbor_directions * neighbor_alpha 

    # Get collisions 
    collision_neighbors = tree.query_radius(positions, collision_r)
    collision_directions = np.array([complex(*(positions[i] - positions[ids]).mean(axis=0)) if len(ids) > 0 else 0j for i, ids in enumerate(collision_neighbors)])
    
    directions = directions + collision_directions * collision_alpha
    
    # Adding boundary forces
    if VERTICAL_BOUNDARIES : 
        # bottom
        near_boundary = np.where(positions[:, 1] < boundary_r)
        idx = np.where(directions[near_boundary].imag < 0)
        directions[near_boundary[0][idx]] = directions[near_boundary[0][idx]] + boundary_alpha * 1j

        # top
        near_boundary = np.where(positions[:, 1] > boundary_r)
        idx = np.where(directions[near_boundary].imag > 0)
        directions[near_boundary[0][idx]] = directions[near_boundary[0][idx]] - boundary_alpha * 1j
    
    if HORIZONTAL_BOUNDARIES : 
        # left
        near_boundary = np.where(positions[:, 0] < boundary_r)
        directions[near_boundary] = directions[near_boundary] + boundary_epsilon

        # right
        near_boundary = np.where(positions[:, 0] > boundary_r)
        directions[near_boundary] = directions[near_boundary] - boundary_epsilon 

    directions = directions / np.abs(directions)
    arr_directions = np.array([[d.real, d.imag] for d in directions * top_speeds])
    positions = (positions + arr_directions * dt) % 1

    # Enforcing boundaries
    if VERTICAL_BOUNDARIES : 
        positions[np.where(positions[:, 1] < 0)] = dt * top_speeds.max()
        positions[np.where(positions[:, 1] > 1)] = 1 - dt * top_speeds.max()

    if HORIZONTAL_BOUNDARIES : 
        positions[np.where(positions[:, 0] < 0)] = 0
        positions[np.where(positions[:, 0] > 1)] = 1

    positions = positions % 1

# %% Plotting
plotdf = pd.DataFrame(columns=['t', 'n', 'x', 'y'])

for t, pos in enumerate(historical_positions) : 
    dat = pd.DataFrame(columns=['t', 'n', 'x', 'y'])
    dat['t'] = [t] * N
    dat['n'] = np.arange(N) 
    dat['x'] = pos[:, 0]
    dat['y'] = pos[:, 1]

    plotdf = pd.concat([plotdf, dat])

fig = px.scatter(
    plotdf, x='x', y='y', 
    animation_frame='t', 
    range_x = [0,1], 
    range_y = [0,1],
    template='plotly_dark'
    )

fig.layout['updatemenus'][0]['buttons'][0]['args'] = (None,
        {'frame': {'duration': 0.0, 'redraw': False},
        'mode': 'immediate',
        'fromcurrent': True,
        'transition': {'duration': 0, 'easing': 'linear'}})

fig.update_layout(
    xaxis=dict(visible=False),
    yaxis=dict(visible=False)
)

fig.show()

# %%
