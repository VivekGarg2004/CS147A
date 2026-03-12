import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


with open("data/adj_mat_volume.pkl", "rb") as f:
    adj_mat = pickle.load(f)
    #print(len(adj), type(adj), type(adj[2]), adj[2])

adj = np.array(adj_mat[2])
for i in np.arange(adj.shape[0]):
    for j in np.arange(adj.shape[1]):
        if adj[i][j] != 0:
            print(adj_mat[0][i], adj_mat[0][j], adj[i][j])

# save adj to csv
np.savetxt("adj_matrix.csv", adj, delimiter=",")
print(adj.shape if hasattr(adj, 'shape')else "error: no shape attribute")
print(type(adj))
print(f"Min: {adj.min():.4f}, Max: {adj.max():.4f}, Sparsity: {(adj == 0).mean():.2%}")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Heatmap
im = axes[0].imshow(adj, cmap="hot_r", aspect="auto")
axes[0].set_title("Adjacency Matrix Heatmap")
axes[0].set_xlabel("Sensor Index")
axes[0].set_ylabel("Sensor Index")
plt.colorbar(im, ax=axes[0])

# Degree distribution
degrees = (adj > 0).sum(axis=1)
axes[1].hist(degrees, bins=20, color="steelblue", edgecolor="black")
axes[1].set_title("Node Degree Distribution")
axes[1].set_xlabel("Degree (# of connected neighbors)")
axes[1].set_ylabel("Count")

plt.tight_layout()
plt.savefig("adj_matrix_viz.png", dpi=150)
plt.show()
print("Saved to adj_matrix_viz.png")


print("Diagonal (should be 1 or 0):", np.diag(adj)[:5])
print("Symmetric?", np.allclose(adj, adj.T))

np.fill_diagonal(adj, 0)  # temporarily remove self-loops
neighbors = (adj > 0).sum(axis=1)
print("Avg neighbors:", neighbors.mean())
print("Min neighbors:", neighbors.min())
print("Max neighbors:", neighbors.max())
print("Isolated nodes (0 neighbors):", (neighbors == 0).sum())


# plot this on LA actual map
loc_file = "../data/sensor_location_150.csv"
if os.path.exists(loc_file):
    df = pd.read_csv(loc_file)
    lats = df["Latitude"].values
    lons = df["Longitude"].values
    
    plt.figure(figsize=(10, 8))
    connected = (adj > 0).sum(axis=1) > 0
    
    plt.scatter(lons[~connected], lats[~connected], c='gray', s=20, alpha=0.5, zorder=5, label='Isolated Sensors')
    plt.scatter(lons[connected], lats[connected], c='blue', s=40, zorder=6, label='Connected Sensors')
    
    # Draw edges
    edges_drawn = 0
    for i in range(len(lats)):
        for j in range(len(lats)):
            if i != j and adj[i, j] > 0:
                plt.plot([lons[i], lons[j]], [lats[i], lats[j]], 'r-', 
                         alpha=min(1.0, max(0.1, adj[i, j])), linewidth=1.5, zorder=1)
                edges_drawn += 1
                
    plt.title(f"LA Freeway Sensor Map & Connectivity\n(Nodes: {len(lats)}, Connected: {connected.sum()}, Edges: {edges_drawn})")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("sensor_map_viz.png", dpi=150)
    plt.show()
    print("Saved map visualization to sensor_map_viz.png")
    
    # --- Interactive Folium Map ---
    try:
        import folium
        print("Generating interactive Folium map...")
        # Center the map around the average coordinates
        center_lat = lats.mean()
        center_lon = lons.mean()
        m = folium.Map(location=[center_lat, center_lon], zoom_start=10, tiles='CartoDB positron')
        
        # Add edges first so they are under the nodes
        for i in range(len(lats)):
            for j in range(len(lats)):
                if i != j and adj[i, j] > 0:
                    weight = adj[i, j]
                    folium.PolyLine(
                        locations=[(float(lats[i]), float(lons[i])), (float(lats[j]), float(lons[j]))],
                        color="red",
                        weight=2,
                        opacity=min(1.0, max(0.1, float(weight))),
                        tooltip=f"Weight: {weight:.4f}"
                    ).add_to(m)
                    
        for i in range(len(lats)):
            is_connected = bool(connected[i])
            color = "blue" if is_connected else "gray"
            radius = 4 if is_connected else 2
            sensor_id = df.iloc[i]['ID'] if 'ID' in df.columns else i
            folium.CircleMarker(
                location=(float(lats[i]), float(lons[i])),
                radius=radius,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                tooltip=f"Sensor ID: {sensor_id}<br>Connected: {is_connected}"
            ).add_to(m)
            
        m.save("sensor_map_interactive.html")
        print("Saved interactive map to sensor_map_interactive.html")
        print("You can open sensor_map_interactive.html in your web browser.")
    except ImportError:
        print("Install folium (`pip install folium`) to generate an interactive HTML map.")

else:
    print(f"Could not find {loc_file}")