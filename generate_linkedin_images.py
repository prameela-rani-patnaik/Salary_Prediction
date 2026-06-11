import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ==========================================
# IMAGE 1: Scatter Plot with Regression Line
# ==========================================
fig, ax = plt.subplots(figsize=(12, 8))

# Sample data (from your training set)
X_train_sample = np.array([1.2, 1.4, 1.6, 2.1, 2.3, 3.0, 3.1, 3.3, 3.3, 3.8, 4.0, 4.1, 4.1, 4.2, 4.6, 5.0, 5.2, 5.4, 6.0, 6.1])
Y_train_sample = np.array([39344., 46206., 37732., 43526., 39892., 56643., 60151., 54446., 64446., 57190., 63219., 55795., 56958., 57082., 61112., 67939., 66030., 83089., 81364., 93941.])

# Model parameters (from your training)
m = 9426
b = 24382

# Plot scatter points
ax.scatter(X_train_sample, Y_train_sample, color='#0066CC', s=150, alpha=0.7, label='Training Data', edgecolors='black', linewidth=1.5)

# Plot regression line
X_line = np.array([X_train_sample.min() - 0.5, X_train_sample.max() + 0.5])
Y_line = m * X_line + b
ax.plot(X_line, Y_line, color='#FF6B35', linewidth=3, label=f'Fitted Line (y = {m}x + {b})')

# Styling
ax.set_xlabel('Years of Experience', fontsize=16, fontweight='bold')
ax.set_ylabel('Salary ($)', fontsize=16, fontweight='bold')
ax.set_title('Salary Prediction Model | Linear Regression\nR² Score: 0.9553 (95.5% Accuracy)', 
             fontsize=18, fontweight='bold', pad=20)
ax.legend(fontsize=13, loc='upper left', framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_facecolor('#F8F9FA')

# Format y-axis as currency
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

plt.tight_layout()
plt.savefig('linkedin_image_1_scatter_plot.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Image 1 saved: linkedin_image_1_scatter_plot.png")
plt.close()

# ==========================================
# IMAGE 2: Metrics Dashboard Card
# ==========================================
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Background
rect = FancyBboxPatch((0.3, 0.3), 9.4, 9.4, boxstyle="round,pad=0.1", 
                       edgecolor='#0066CC', facecolor='#F0F4FF', linewidth=3)
ax.add_patch(rect)

# Title
ax.text(5, 9, 'Day 5 Learning Update', 
        fontsize=28, fontweight='bold', ha='center', color='#0066CC')
ax.text(5, 8.4, 'Custom Linear Regression Model', 
        fontsize=18, ha='center', color='#666666', style='italic')

# Divider line
ax.plot([1, 9], [8, 8], color='#0066CC', linewidth=2)

# Metrics in a grid
metrics = [
    ('R² Score', '0.9553', '#FF6B35'),
    ('Accuracy', '95.5%', '#FF6B35'),
    ('Training Samples', '20', '#00A86B'),
    ('Test Samples', '10', '#00A86B'),
    ('Slope (m)', '9,426', '#6A4C93'),
    ('Intercept (b)', '24,382', '#6A4C93'),
]

positions = [(1.5, 6.8), (5, 6.8), (8.5, 6.8),
             (1.5, 5), (5, 5), (8.5, 5)]

for (metric, value, color), (x, y) in zip(metrics, positions):
    # Box background
    box = FancyBboxPatch((x-0.8, y-0.5), 1.6, 1, boxstyle="round,pad=0.05",
                          edgecolor=color, facecolor='white', linewidth=2)
    ax.add_patch(box)
    
    # Metric name
    ax.text(x, y+0.25, metric, fontsize=11, ha='center', color='#666666', fontweight='bold')
    # Metric value
    ax.text(x, y-0.15, value, fontsize=16, ha='center', color=color, fontweight='bold')

# Bottom text
ax.text(5, 3.2, '✓ Built from scratch (not using sklearn)', 
        fontsize=12, ha='center', color='#333333')
ax.text(5, 2.7, '✓ Train-test split methodology applied', 
        fontsize=12, ha='center', color='#333333')
ax.text(5, 2.2, '✓ Understanding the math behind the model', 
        fontsize=12, ha='center', color='#333333')

# Hashtags at bottom
ax.text(5, 1.2, '#MachineLearning #LinearRegression #DataScience #AILearning #PythonDeveloper', 
        fontsize=10, ha='center', color='#0066CC', fontweight='bold')

plt.tight_layout()
plt.savefig('linkedin_image_2_metrics_dashboard.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Image 2 saved: linkedin_image_2_metrics_dashboard.png")
plt.close()

print("\n🎉 Both images generated successfully!")
print("📸 linkedin_image_1_scatter_plot.png - Show your model fit")
print("📊 linkedin_image_2_metrics_dashboard.png - Show your metrics")
