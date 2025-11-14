# %%
import matplotlib.pyplot as plt

# Data
categories = [
    "Aphid", "Black Rust", "Brown Rust", "Fusarium Head Blight", "Healthy",
    "Leaf Blight", "Mildew", "Mite", "Septoria", "Smut", "Stripe Rust", "Yellow Dwarf"
]
values = [
    967, 1543, 3363, 1275, 3878,
    842, 1105, 800, 2405, 1310, 3753, 633
]

categories = categories[::-1]
values = values[::-1]

# Creating horizontal bar chart with value annotations
plt.figure(figsize=(12, 6))
bars = plt.barh(categories, values, color='skyblue')

# Annotating values on the bars
for bar in bars:
    plt.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2,
             f'{int(bar.get_width())}', va='center', fontsize=12)

# Labels and title
plt.yticks(fontsize=12)
plt.xlabel('Count')
plt.ylabel('Category')
plt.title('Class Distribution Wheat Disease Dataset')
plt.tight_layout()

# Display the chart
plt.show()


# %%
import matplotlib.pyplot as plt

# Data
labels = ['Train', 'Validation', 'Test']
sizes = [60, 20, 20]
colors = ['skyblue', 'lightgreen', 'lightcoral']

# Adding explode effect for 3D-like appearance
explode = (0.05, 0.05, 0.05)  # Separate each slice slightly

# Create the pie chart
plt.figure(figsize=(8, 8))
wedges, texts, autotexts = plt.pie(
    sizes,
    labels=labels,
    autopct='%1.1f%%',
    colors=colors,
    explode=explode,  # Add the explode effect
    startangle=90,
    textprops={'fontsize': 22},
    shadow=True  # Add shadow for better 3D appearance
)

# Add title
plt.title('Dataset Split', fontsize=22)

# Display the chart
plt.tight_layout()
plt.show()


# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data
data = {
    "Class": [
        "Aphid", "Black Rust", "Brown Rust", "Fusarium Head Blight", "Healthy",
        "Leaf Blight", "Mildew", "Mite", "Septoria", "Smut", "Stripe Rust", "Yellow Dwarf"
    ],
    "Class Weight": [
        1.97298851, 1.23711712, 0.43198691, 1.49586057, 0.49197478,
        2.2660066, 1.72599296, 2.38402778, 0.79302379, 1.45589483,
        0.50836665, 3.18755803
    ]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Adjust data to 4 decimal places
df["Class Weight"] = df["Class Weight"].round(4)

# Normalize the weights for color coding again
normalized_weights = (df["Class Weight"] - df["Class Weight"].min()) / (
    df["Class Weight"].max() - df["Class Weight"].min())

# Create a blue gradient color palette
cmap = sns.light_palette("green", as_cmap=True)

# Plot a table
fig, ax = plt.subplots(figsize=(6, 5))
ax.axis('tight')
ax.axis('off')

# Color code the weights
colors = [[cmap(val)] * len(df.columns) for val in normalized_weights]

# Create the table with adjustments
table = ax.table(
    cellText=df.values,
    colLabels=df.columns,
    cellColours=colors,
    cellLoc='center',
    loc='center'
)

# Adjust table properties
table.auto_set_font_size(False)
table.set_fontsize(14)
table.auto_set_column_width(col=list(range(len(df.columns))))

# Add more space around cells
for key, cell in table.get_celld().items():
    cell.set_height(0.1)
    cell.set_edgecolor("black")

# Display the table
plt.show()



# %%
import matplotlib.pyplot as plt
import numpy as np

# Data
models = ['CNN', 'Mobilenet V2', 'ResNet 50', 'ViT', 'CViT']
with_class = [0.7606, 0.8844, 0.8434, 0.8781, 0.8024]
without_class = [0.6306, 0.8567, 0.8306, 0.8284, 0.6766]

# Bar chart parameters
x = np.arange(len(models))
width = 0.35

# Plot with values displayed
plt.figure(figsize=(10, 6))
bars_with_class = plt.bar(x - width/2, with_class, width, label='With Class Balance', color='lightgreen')
bars_without_class = plt.bar(x + width/2, without_class, width, label='Without Class Balance', color='lightcoral')

# Add value labels
for bar in bars_with_class:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 0.01, f'{height:.2f}', ha='center', va='bottom')

for bar in bars_without_class:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 0.01, f'{height:.2f}', ha='center', va='bottom')

# Labels, title, and legend
plt.xlabel('Models')
plt.ylabel('Validation Accuracy')
plt.title('Validation Accuracy Comparison by Model')
plt.xticks(x, models)
plt.ylim(0, 1)
plt.legend()

# Show plot
plt.tight_layout()
plt.show()



# %% [markdown]
# MobileNet F1

# %%
# Data for F-1 Scores
diseases = [
    'Aphid', 'Black Rust', 'Brown Rust', 'Fusarium Head Blight', 'Healthy',
    'Leaf Blight', 'Mildew', 'Mite', 'Septoria', 'Smut', 'Stripe Rust', 'Yellow Dwarf'
]
with_f1 = [0.77, 0.75, 0.81, 0.92, 0.93, 0.55, 0.84, 0.8, 0.98, 0.93, 0.86, 0.96]
without_f1 = [0.81, 0.8, 0.85, 0.92, 0.94, 0.61, 0.88, 0.82, 0.95, 0.95, 0.9, 0.98]

# Bar chart parameters
x = np.arange(len(diseases))
width = 0.35

# Plot
plt.figure(figsize=(12, 6))
plt.bar(x - width/2, with_f1, width, label='With Class Balance', color='lightgreen')
plt.bar(x + width/2, without_f1, width, label='Without Class Balance', color='lightcoral')

# Add value labels
for i, v in enumerate(with_f1):
    plt.text(i - width/2, v + 0.01, f'{v:.2f}', ha='center', va='bottom')
for i, v in enumerate(without_f1):
    plt.text(i + width/2, v + 0.01, f'{v:.2f}', ha='center', va='bottom')

# Labels, title, and legend
plt.xlabel('Diseases')
plt.ylabel('F-1 Score')
plt.title('F-1 Score Comparison by Disease')
plt.xticks(x, diseases, rotation=45, ha='right')
plt.ylim(0, 1)
plt.legend()

# Show plot
plt.tight_layout()
plt.show()


# %% [markdown]
# ResNet50

# %%
# New F-1 Scores Data
diseases_new = [
    'Aphid', 'Black Rust', 'Brown Rust', 'Fusarium Head Blight', 'Healthy',
    'Leaf Blight', 'Mildew', 'Mite', 'Septoria', 'Smut', 'Stripe Rust', 'Yellow Dwarf'
]
with_f1_new = [0.76, 0.73, 0.8, 0.95, 0.92, 0.56, 0.84, 0.74, 0.95, 0.92, 0.81, 0.94]
without_f1_new = [0.73, 0.7, 0.77, 0.94, 0.92, 0.5, 0.84, 0.72, 0.93, 0.94, 0.83, 0.91]

# Bar chart parameters
x_new = np.arange(len(diseases_new))
width_new = 0.35

# Plot
plt.figure(figsize=(12, 6))
plt.bar(x_new - width_new/2, with_f1_new, width_new, label='With Class Balance', color='lightgreen')
plt.bar(x_new + width_new/2, without_f1_new, width_new, label='Without Class', color='lightcoral')

# Add value labels
for i, v in enumerate(with_f1_new):
    plt.text(i - width_new/2, v + 0.01, f'{v:.2f}', ha='center', va='bottom')
for i, v in enumerate(without_f1_new):
    plt.text(i + width_new/2, v + 0.01, f'{v:.2f}', ha='center', va='bottom')

# Labels, title, and legend
plt.xlabel('Diseases')
plt.ylabel('F-1 Score')
plt.title('F-1 Score Comparison by Disease')
plt.xticks(x_new, diseases_new, rotation=45, ha='right')
plt.ylim(0, 1)
plt.legend()

# Show plot
plt.tight_layout()
plt.show()

# %% [markdown]
# CViT

# %%
import pandas as pd
import matplotlib.pyplot as plt

# Data Preparation
data = {
    "Disease": [
        "Aphid", "Black Rust", "Brown Rust", "Fusarium Head Blight", "Healthy",
        "Leaf Blight", "Mildew", "Mite", "Septoria", "Smut", "Stripe Rust", "Yellow Dwarf", "Weighted Average"
    ],
    "With Class Balance": [0.56, 0.66, 0.79, 0.89, 0.88, 0.45, 0.78, 0.41, 0.93, 0.9, 0.83, 0.9, 0.8],
    "Without Class Balance": [0.49, 0.54, 0.55, 0.78, 0.8, 0.34, 0.74, 0.44, 0.88, 0.73, 0.75, 0.74, 0.68]
}
df = pd.DataFrame(data)

# Define custom function to compare row-wise and assign colors
def row_wise_colors(row):
    colors = []
    for col in ["With Class Balance", "Without Class Balance"]:
        if row[col] == row[["With Class Balance", "Without Class Balance"]].max():
            colors.append("lightgreen")
        else:
            colors.append("lightcoral")
    return ["white"] + colors  # Add white for the 'Disease' column

# Generate color matrix for the table
cell_colors = [row_wise_colors(row) for _, row in df.iterrows()]

# Add header row colors
header_colors = ["white"] * len(df.columns)
cell_colors.insert(0, header_colors)

# Visualize the table with proper spacing
fig, ax = plt.subplots(figsize=(8, 8))
table = ax.table(
    cellText=[df.columns] + df.values.tolist(),
    cellColours=cell_colors,
    loc="center",
    cellLoc="center",
)

# Enhance table visuals with padding
table.auto_set_font_size(False)
table.set_fontsize(12)
table.auto_set_column_width(list(range(len(df.columns))))
for (row, col), cell in table.get_celld().items():
    cell.set_text_props(ha="center", va="center")
    cell.set_edgecolor("black")
    cell.set_height(0.05)  # Adjust cell height for better readability
    cell.set_width(0.3)   # Adjust cell width for better spacing

ax.axis("off")
plt.show()


# %% [markdown]
# ViT

# %%
import pandas as pd
import matplotlib.pyplot as plt

# Data Preparation
data = {
    "Disease": [
        "Aphid", "Black Rust", "Brown Rust", "Fusarium Head Blight", "Healthy",
        "Leaf Blight", "Mildew", "Mite", "Septoria", "Smut", "Stripe Rust", "Yellow Dwarf", "Weighted Average"
    ],
    "With Class Balance": [0.84, 0.77, 0.84, 0.96, 0.95, 0.6, 0.89, 0.83, 0.95, 0.93, 0.87, 0.98, 0.88],
    "Without Class Balance": [0.6, 0.66, 0.76, 0.95, 0.94, 0.54, 0.9, 0.79, 0.88, 0.93, 0.86, 0.89, 0.83]
}
df = pd.DataFrame(data)

# Define custom function to compare row-wise and assign colors
def row_wise_colors(row):
    colors = []
    for col in ["With Class Balance", "Without Class Balance"]:
        if row[col] == row[["With Class Balance", "Without Class Balance"]].max():
            colors.append("lightgreen")
        else:
            colors.append("lightcoral")
    return ["white"] + colors  # Add white for the 'Disease' column

# Generate color matrix for the table
cell_colors = [row_wise_colors(row) for _, row in df.iterrows()]

# Add header row colors
header_colors = ["white"] * len(df.columns)
cell_colors.insert(0, header_colors)

# Visualize the table with proper spacing
fig, ax = plt.subplots(figsize=(8, 8))
table = ax.table(
    cellText=[df.columns] + df.values.tolist(),
    cellColours=cell_colors,
    loc="center",
    cellLoc="center",
)

# Enhance table visuals with padding
table.auto_set_font_size(False)
table.set_fontsize(12)
table.auto_set_column_width(list(range(len(df.columns))))
for (row, col), cell in table.get_celld().items():
    cell.set_text_props(ha="center", va="center")
    cell.set_edgecolor("black")
    cell.set_height(0.05)  # Adjust cell height for better readability
    cell.set_width(0.3)   # Adjust cell width for better spacing

ax.axis("off")
plt.show()


# %% [markdown]
# CNN

# %%
import pandas as pd
import matplotlib.pyplot as plt

# Data Preparation
data = {
    "Disease": [
        "Aphid", "Black Rust", "Brown Rust", "Fusarium Head Blight", "Healthy",
        "Leaf Blight", "Mildew", "Mite", "Septoria", "Smut", "Stripe Rust", "Yellow Dwarf", "Weighted Average"
    ],
    "With Class Balance": [0.55, 0.61, 0.75, 0.8, 0.85, 0.39, 0.74, 0.51, 0.89, 0.62, 0.84, 0.91, 0.76],
    "Without Class Balance": [0.48, 0.51, 0.6, 0.77, 0.76, 0.31, 0.53, 0.44, 0.73, 0.52, 0.74, 0.57, 0.64]
}
df = pd.DataFrame(data)

# Define custom function to compare row-wise and assign colors
def row_wise_colors(row):
    colors = []
    for col in ["With Class Balance", "Without Class Balance"]:
        if row[col] == row[["With Class Balance", "Without Class Balance"]].max():
            colors.append("lightgreen")
        else:
            colors.append("lightcoral")
    return ["white"] + colors  # Add white for the 'Disease' column

# Generate color matrix for the table
cell_colors = [row_wise_colors(row) for _, row in df.iterrows()]

# Add header row colors
header_colors = ["white"] * len(df.columns)
cell_colors.insert(0, header_colors)

# Visualize the table with proper spacing
fig, ax = plt.subplots(figsize=(8, 8))
table = ax.table(
    cellText=[df.columns] + df.values.tolist(),
    cellColours=cell_colors,
    loc="center",
    cellLoc="center",
)

# Enhance table visuals with padding
table.auto_set_font_size(False)
table.set_fontsize(12)
table.auto_set_column_width(list(range(len(df.columns))))
for (row, col), cell in table.get_celld().items():
    cell.set_text_props(ha="center", va="center")
    cell.set_edgecolor("black")
    cell.set_height(0.05)  # Adjust cell height for better readability
    cell.set_width(0.3)   # Adjust cell width for better spacing

ax.axis("off")
plt.show()


# %% [markdown]
# 
# MobileNetv2

# %%
import pandas as pd
import matplotlib.pyplot as plt

# Data Preparation
data = {
    "Disease": [
        "Aphid", "Black Rust", "Brown Rust", "Fusarium Head Blight", "Healthy",
        "Leaf Blight", "Mildew", "Mite", "Septoria", "Smut", "Stripe Rust", "Yellow Dwarf", "Weighted Average"
    ],
    "With Class Balance": [0.81, 0.8, 0.85, 0.92, 0.94, 0.61, 0.88, 0.82, 0.95, 0.95, 0.9, 0.98, 0.8702],
    "Without Class Balance": [0.77, 0.75, 0.81, 0.92, 0.93, 0.55, 0.84, 0.8, 0.98, 0.93, 0.86, 0.96, 0.8418],
}
df = pd.DataFrame(data)

# Define custom function to compare row-wise and assign colors
def row_wise_colors(row):
    colors = []
    for col in ["With Class Balance", "Without Class Balance"]:
        if row[col] == row[["With Class Balance", "Without Class Balance"]].max():
            colors.append("lightgreen")
        else:
            colors.append("lightcoral")
    return ["white"] + colors  # Add white for the 'Disease' column

# Generate color matrix for the table
cell_colors = [row_wise_colors(row) for _, row in df.iterrows()]

# Add header row colors
header_colors = ["white"] * len(df.columns)
cell_colors.insert(0, header_colors)

# Visualize the table with proper spacing
fig, ax = plt.subplots(figsize=(8, 8))
table = ax.table(
    cellText=[df.columns] + df.values.tolist(),
    cellColours=cell_colors,
    loc="center",
    cellLoc="center",
)

# Enhance table visuals with padding
table.auto_set_font_size(False)
table.set_fontsize(12)
table.auto_set_column_width(list(range(len(df.columns))))
for (row, col), cell in table.get_celld().items():
    cell.set_text_props(ha="center", va="center")
    cell.set_edgecolor("black")
    cell.set_height(0.05)  # Adjust cell height for better readability
    cell.set_width(0.3)   # Adjust cell width for better spacing

ax.axis("off")
plt.show()


# %% [markdown]
# Resnet50

# %%
import pandas as pd
import matplotlib.pyplot as plt

# Data Preparation
data = {
    "Disease": [
        "Aphid", "Black Rust", "Brown Rust", "Fusarium Head Blight", "Healthy",
        "Leaf Blight", "Mildew", "Mite", "Septoria", "Smut", "Stripe Rust", "Yellow Dwarf", "Weighted Average"
    ],
    "With Class Balance": [0.76, 0.73, 0.8, 0.95, 0.92, 0.56, 0.84, 0.74, 0.95, 0.92, 0.81, 0.94, 0.83],
    "Without Class Balance": [0.73, 0.7, 0.77, 0.94, 0.92, 0.5, 0.84, 0.72, 0.93, 0.94, 0.83, 0.91, 0.81],
}
df = pd.DataFrame(data)

# Define custom function to compare row-wise and assign colors
def row_wise_colors(row):
    colors = []
    for col in ["With Class Balance", "Without Class Balance"]:
        if row[col] == row[["With Class Balance", "Without Class Balance"]].max():
            colors.append("lightgreen")
        else:
            colors.append("lightcoral")
    return ["white"] + colors  # Add white for the 'Disease' column

# Generate color matrix for the table
cell_colors = [row_wise_colors(row) for _, row in df.iterrows()]

# Add header row colors
header_colors = ["white"] * len(df.columns)
cell_colors.insert(0, header_colors)

# Visualize the table with proper spacing
fig, ax = plt.subplots(figsize=(8, 8))
table = ax.table(
    cellText=[df.columns] + df.values.tolist(),
    cellColours=cell_colors,
    loc="center",
    cellLoc="center",
)

# Enhance table visuals with padding
table.auto_set_font_size(False)
table.set_fontsize(12)
table.auto_set_column_width(list(range(len(df.columns))))
for (row, col), cell in table.get_celld().items():
    cell.set_text_props(ha="center", va="center")
    cell.set_edgecolor("black")
    cell.set_height(0.05)  # Adjust cell height for better readability
    cell.set_width(0.3)   # Adjust cell width for better spacing

ax.axis("off")
plt.show()


# %% [markdown]
# ResNet50

# %%
import pandas as pd
import matplotlib.pyplot as plt

# Data Preparation
data = {
    "Disease": [
        "Aphid", "Black Rust", "Brown Rust", "Fusarium Head Blight", "Healthy",
        "Leaf Blight", "Mildew", "Mite", "Septoria", "Smut", "Stripe Rust", "Yellow Dwarf", "Weighted Average"
    ],
    "With Class Balance": [0.73, 0.7, 0.77, 0.94, 0.92, 0.5, 0.84, 0.72, 0.93, 0.94, 0.83, 0.91, 0.83],
    "Without Class Balance": [0.76, 0.73, 0.8, 0.95, 0.92, 0.56, 0.84, 0.74, 0.95, 0.92, 0.81, 0.94, 0.84],
}
df = pd.DataFrame(data)

# Define custom function to compare row-wise and assign colors
def row_wise_colors(row):
    colors = []
    for col in ["With Class Balance", "Without Class Balance"]:
        if row[col] == row[["With Class Balance", "Without Class Balance"]].max():
            colors.append("lightgreen")
        else:
            colors.append("lightcoral")
    return ["white"] + colors  # Add white for the 'Disease' column

# Generate color matrix for the table
cell_colors = [row_wise_colors(row) for _, row in df.iterrows()]

# Add header row colors
header_colors = ["white"] * len(df.columns)
cell_colors.insert(0, header_colors)

# Visualize the table with proper spacing
fig, ax = plt.subplots(figsize=(8, 8))
table = ax.table(
    cellText=[df.columns] + df.values.tolist(),
    cellColours=cell_colors,
    loc="center",
    cellLoc="center",
)

# Enhance table visuals with padding
table.auto_set_font_size(False)
table.set_fontsize(12)
table.auto_set_column_width(list(range(len(df.columns))))
for (row, col), cell in table.get_celld().items():
    cell.set_text_props(ha="center", va="center")
    cell.set_edgecolor("black")
    cell.set_height(0.05)  # Adjust cell height for better readability
    cell.set_width(0.3)   # Adjust cell width for better spacing

ax.axis("off")
plt.show()


# %%



