import pandas as pd
import numpy as np
import plotly.graph_objects as go

if __name__ == "__main__":
    df = pd.read_csv("dreamer/logdir/walker_walk_pixels/plots/model_loss.csv")
    fig = go.Figure(go.Scatter(y=df["kl_divergence_loss"]))
    fig.show()
