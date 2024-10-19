import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Data from the table
data = {
    'Assignee': ['Harish', 'Jayaram', 'Sankar'],
    'Closed': [0, 8, 5],
    'Feedback': [0, 17, 4],
    'In Progress': [0, 1, 1],
    'New': [5, 27, 8],
    'Resolved': [0, 2, 4],
    'Clarified': [0, 0, 1]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Set Assignee as index
df.set_index('Assignee', inplace=True)

# Plot grouped bar chart
ax = df.plot(kind='bar', figsize=(10, 6), color=['#5DADE2', '#F5B041', '#58D68D', '#EC7063', '#AF7AC5', '#F4D03F'])

# Adding titles and labels
plt.title('Grouped Task Distribution per Assignee')
plt.xlabel('Assignee')
plt.ylabel('Number of Tasks')
plt.xticks(rotation=45, ha='right')

# Show values on each bar
for container in ax.containers:
    ax.bar_label(container, label_type='edge', fontsize=10)

plt.tight_layout()
plt.show()
