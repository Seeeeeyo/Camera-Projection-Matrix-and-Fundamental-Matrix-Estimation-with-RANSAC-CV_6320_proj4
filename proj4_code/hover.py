import matplotlib.pyplot as plt
from feature_matching.utils import load_image, PIL_resize, rgb2gray


image1_path = '../data/new1.jpg'
image2_path = '../data/new2.jpg'

img1 = load_image(image1_path)
img2 = load_image(image2_path)

import plotly.graph_objects as go
import numpy as np

# Create a Plotly figure with the image
fig = go.Figure(go.Image(z=img1))

# JavaScript to display hover coordinates in the console
fig.update_layout(
    hovermode='closest'
)
fig.update_traces(
    hovertemplate='x: %{x}<br>y: %{y}',
)

fig.show()