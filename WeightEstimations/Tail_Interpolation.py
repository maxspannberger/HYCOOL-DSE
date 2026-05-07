import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

data = pd.read_csv('WeightEstimations\Torenbeek_Htail_Vtail.csv')
x_data = data['x'].values
y_data = data['y'].values

def empirical_model(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

params, _ = curve_fit(empirical_model, x_data, y_data)

def get_weight_factor(x):
    return empirical_model(x, *params)


'''fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(x_data, y_data, color='red', label='Digitized Points', zorder=5)

x_range = np.linspace(x_data.min(), x_data.max(), 100)
y_fit = get_weight_factor(x_range)

ax.plot(x_range, y_fit, color='blue', label='Polynomial Fit (3rd Degree)', linewidth=2)

ax.set_title("Torenbeek Empirical Weight Factor: H-Tail & V-Tail")
ax.set_xlabel("Input Parameter (e.g., Surface Area or Tail Volume)")
ax.set_ylabel("Weight Factor")
ax.grid(True, linestyle='--', alpha=0.7)

eq_text = f"y = {params[0]:.4e}x² + {params[1]:.4e}x + {params[2]:.4f}"
ax.text(0.05, 0.95, eq_text, transform=ax.transAxes, fontsize=10, 
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

ax.legend()
plt.show() '''