import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

project_path = 'D:\\R0\\Supporting Information\\'


df_raw_U = pd.read_excel(f'{project_path}single_conversion_layer.xlsx')
df_y1_U = df_raw_U.iloc[0:, 13:44:1]
time_U = df_raw_U.iloc[0:, 1]

df_raw_C = pd.read_excel(f'{project_path}epoxy_coated.xlsx')
df_y1_C = df_raw_C.iloc[0:, 13:44:1]
time_C = df_raw_C.iloc[0:, 1]

df_combined_y1 = pd.concat([df_y1_U, df_y1_C], axis=0)
y1 = df_combined_y1.iloc[:, -1]

time = pd.concat([time_U, time_C], axis=0, ignore_index=True)
X_df = pd.concat([time, pd.Series([0]*80 + [1]*80, name='coating')], axis=1)

freq = df_combined_y1.columns.astype(float)

df_raw_xrd = pd.read_excel(f'{project_path}XRD.xlsx')
two_theta_values = df_raw_xrd.columns[12:2819].astype(float)
mask = (two_theta_values >= 40) & (two_theta_values <= 70)
selected_column_indices = np.where(mask)[0] + 12
df_X_xrd_filtered = df_raw_xrd.iloc[:, selected_column_indices]
filtered_headers = two_theta_values[mask]
two_theta_filtered = pd.DataFrame(filtered_headers)

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Scale features (X)
scaler_X = StandardScaler()
df_scaled = scaler_X.fit_transform(df_X_xrd_filtered)

pca = PCA(n_components=0.995)
df_pca = pca.fit_transform(df_scaled)

n_components = df_pca.shape[1]
df_reduced = pd.DataFrame(
    df_pca, 
    columns=[f'PC{i+1}' for i in range(n_components)]
)

df_reduced.index = df_X_xrd_filtered.index

df_repeated = pd.concat([df_reduced] * 8, axis=0, ignore_index=True)

X_df_final = pd.concat([X_df, df_repeated], axis=1)

# Scale target variable (y)
scaler_y = StandardScaler()
y1_scaled = scaler_y.fit_transform(y1.values.reshape(-1, 1)).ravel()

# Reconstruct spectra from PCA
df_pca_reconstructed = pca.inverse_transform(df_pca)
df_reconstructed_scaled = scaler_X.inverse_transform(df_pca_reconstructed)
df_reconstructed = pd.DataFrame(
    df_reconstructed_scaled,
    columns=df_X_xrd_filtered.columns
)

# Plot original and reconstructed XRD spectra in 5x4 array
n_rows, n_cols = 5, 4
n_samples_to_plot = n_rows * n_cols

sample_indices = np.linspace(0, len(df_X_xrd_filtered)-1, n_samples_to_plot, dtype=int)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 12))
fig.suptitle('Original vs PCA-Reconstructed XRD Spectra', fontsize=16, fontweight='bold')

fig.subplots_adjust(
    hspace=0.5,
    wspace=0.3,
    top=0.94,
    bottom=0.05
)

# ============================================================================
# Plot 1: Comparison of original and reconstructed XRD
# ============================================================================
from matplotlib.lines import Line2D

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Palatino', 'Georgia', 'serif']
plt.rcParams['mathtext.fontset'] = 'stix'

sample_names = {
    0: 'Pure Cr Conversion Coating',
    1: 'Cr-10% Co Conversion Coating',
    2: 'Cr-20% Co Conversion Coating',
    3: 'Cr-10% Ni Conversion Coating',
    4: 'Cr-20% Ni Conversion Coating',
    5: 'Cr-10% Ce Conversion Coating',
    6: 'Cr-20% Ce Conversion Coating',
    7: 'Cr-10% Sr Conversion Coating',
    8: 'Cr-20% Sr Conversion Coating',
    9: 'Cr-10% La Conversion Coating',
    10: 'Cr-20% La Conversion Coating',
    11: 'Cr-10% Al Conversion Coating',
    12: 'Cr-20% Al Conversion Coating',
    13: 'Cr-10% Cd Conversion Coating',
    14: 'Cr-20% Cd Conversion Coating',
    15: 'Cr-10% Mg Conversion Coating',
    16: 'Cr-20% Mg Conversion Coating',
    17: 'Cr-10% Zn Conversion Coating',
    18: 'Cr-20% Zn Conversion Coating',
    19: 'Blank Steel (No Coating)'
}

for i, ax in enumerate(axes.flat):
    if i < len(sample_indices):
        idx = sample_indices[i]
        
        # Get sample name from mapping
        sample_name = sample_names.get(idx, f'Sample {idx+1}')
        
        original = df_X_xrd_filtered.iloc[idx, :].values
        reconstructed = df_reconstructed.iloc[idx, :].values
        
        ax.plot(two_theta_filtered.values.ravel(), original, 'b-', 
                linewidth=2, alpha=0.8)
        ax.plot(two_theta_filtered.values.ravel(), reconstructed, 'r--', 
                linewidth=1.5, alpha=0.8)
        
        ax.set_xlabel('2θ (degrees)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Intensity (a.u.)', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(40, 70)
        ax.tick_params(axis='both', labelsize=9)
        
        # Calculate R²
        corr_matrix = np.corrcoef(original, reconstructed)
        r2 = corr_matrix[0, 1]**2
        
        # Place R² in upper right corner
        ax.text(0.95, 0.95, f'R² = {r2:.3f}', 
                transform=ax.transAxes, fontsize=8, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
                fontfamily='serif')
        
        legend_elements = [
            Line2D([0], [0], color='blue', linewidth=2, linestyle='-', label='Original XRD'),
            Line2D([0], [0], color='red', linewidth=1.5, linestyle='--', label='Reconstructed')
        ]
        
        legend = ax.legend(handles=legend_elements, 
                     loc='lower left', 
                     bbox_to_anchor=(0.0, 1.01, 1.0, 0.10),
                     fontsize=8, ncol=2, framealpha=1.0,
                     mode='expand',
                     borderaxespad=0,
                     handletextpad=0.3, columnspacing=0.3,
                     title=sample_name, title_fontsize=9)
        legend.get_frame().set_facecolor('#CCFFCC')
        legend.get_frame().set_edgecolor('#999999')
        
        # Set legend text to serif
        for text in legend.get_texts():
            text.set_fontfamily('serif')
        legend.get_title().set_fontfamily('serif')

plt.tight_layout()
plt.savefig(f'{project_path}comparison_original_reconstructed_xrd.png', dpi=300, bbox_inches='tight')
plt.show()

#########################################
# Model Selection (60/20/20 split)
#########################################

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
import seaborn as sns

# Set seed for reproducibility
SEED = 42
np.random.seed(SEED)

# Split data: 60% train, 20% validation, 20% test
# First split: 80% temp, 20% test
X_temp, X_test, y_temp_scaled, y_test_scaled = train_test_split(
    X_df_final, y1_scaled, test_size=0.2, random_state=SEED
)

# Second split: 60% train, 20% validation (from the 80% temp)
X_train, X_val, y_train_scaled, y_val_scaled = train_test_split(
    X_temp, y_temp_scaled, test_size=0.25, random_state=SEED  # 0.25 × 0.8 = 0.2
)

print(f"\nTraining set size: {X_train.shape[0]} samples (60%)")
print(f"Validation set size: {X_val.shape[0]} samples (20%)")
print(f"Test set size: {X_test.shape[0]} samples (20%)")
print(f"Number of features: {X_train.shape[1]}")
print("="*60)

# Define models
models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=SEED),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=SEED),
    'XGBoost': XGBRegressor(n_estimators=100, random_state=SEED, verbosity=0),
    'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1)
}

results = []
predictions = {}

kfold = KFold(n_splits=3, shuffle=True, random_state=SEED)

# Train and evaluate each model
for name, model in models.items():
    print(f"\n Training {name}...")
    if name == 'SVR':
        scaler_X_for_svr = StandardScaler()
        X_train_scaled = scaler_X_for_svr.fit_transform(X_train)
        X_val_scaled = scaler_X_for_svr.transform(X_val)
        X_test_scaled = scaler_X_for_svr.transform(X_test)
        
        cv_scores = cross_val_score(model, X_train_scaled, y_train_scaled, 
                                   cv=kfold, scoring='r2')
        model.fit(X_train_scaled, y_train_scaled)
        y_train_pred_scaled = model.predict(X_train_scaled)
        y_val_pred_scaled = model.predict(X_val_scaled)
        y_test_pred_scaled = model.predict(X_test_scaled)
    else:
        cv_scores = cross_val_score(model, X_train, y_train_scaled, 
                                   cv=kfold, scoring='r2')
        model.fit(X_train, y_train_scaled)
        y_train_pred_scaled = model.predict(X_train)
        y_val_pred_scaled = model.predict(X_val)
        y_test_pred_scaled = model.predict(X_test)
    
    # Inverse transform predictions to original scale
    y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).ravel()
    y_val_pred = scaler_y.inverse_transform(y_val_pred_scaled.reshape(-1, 1)).ravel()
    y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).ravel()
    
    # Inverse transform actual values for metrics
    y_train_actual = scaler_y.inverse_transform(y_train_scaled.reshape(-1, 1)).ravel()
    y_val_actual = scaler_y.inverse_transform(y_val_scaled.reshape(-1, 1)).ravel()
    y_test_actual = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).ravel()
    
    # Calculate metrics on original scale for ALL sets
    r2_train = r2_score(y_train_actual, y_train_pred)
    r2_val = r2_score(y_val_actual, y_val_pred)
    r2_test = r2_score(y_test_actual, y_test_pred)
    
    rmse_train = np.sqrt(mean_squared_error(y_train_actual, y_train_pred))
    rmse_val = np.sqrt(mean_squared_error(y_val_actual, y_val_pred))
    rmse_test = np.sqrt(mean_squared_error(y_test_actual, y_test_pred))
    
    mae_train = mean_absolute_error(y_train_actual, y_train_pred)
    mae_val = mean_absolute_error(y_val_actual, y_val_pred)
    mae_test = mean_absolute_error(y_test_actual, y_test_pred)
    
    # Store predictions for plotting (on original scale)
    predictions[name] = {
        'train': (y_train_actual, y_train_pred),
        'val': (y_val_actual, y_val_pred),
        'test': (y_test_actual, y_test_pred)
    }
    
    # Store results with ALL metrics
    results.append({
        'Model': name,
        'R² (Train)': r2_train,
        'R² (Val)': r2_val,
        'R² (Test)': r2_test,
        'R² CV Mean': cv_scores.mean(),
        'R² CV Std': cv_scores.std(),
        'RMSE (Train)': rmse_train,
        'RMSE (Val)': rmse_val,
        'RMSE (Test)': rmse_test,
        'MAE (Train)': mae_train,
        'MAE (Val)': mae_val,
        'MAE (Test)': mae_test
    })
    
    print(f"  R² CV Mean: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  R² Train: {r2_train:.4f}, R² Val: {r2_val:.4f}, R² Test: {r2_test:.4f}")
    print(f"  RMSE Train: {rmse_train:.4f}, RMSE Val: {rmse_val:.4f}, RMSE Test: {rmse_test:.4f}")

# Create results DataFrame
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('R² CV Mean', ascending=False)

print("\n" + "="*60)
print("MODEL COMPARISON SUMMARY")
print("="*60)
print(results_df.to_string(index=False, float_format=lambda x: '{:.4f}'.format(x)))

# ============================================================================
# Plot 2: Parity plots for all models
# ============================================================================

# Create parity plots for all 4 models
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
plt.rcParams['font.family'] = 'serif'
axes = axes.ravel()

# Get min and max values for consistent axis limits (on original scale)
all_train_actual = np.concatenate([predictions[name]['train'][0] for name in predictions.keys()])
all_val_actual = np.concatenate([predictions[name]['val'][0] for name in predictions.keys()])
all_test_actual = np.concatenate([predictions[name]['test'][0] for name in predictions.keys()])
all_values = np.concatenate([all_train_actual, all_val_actual, all_test_actual])
y_min, y_max = all_values.min(), all_values.max()
margin = (y_max - y_min) * 0.08
y_min, y_max = y_min - margin, y_max + margin

from matplotlib.ticker import AutoMinorLocator, MaxNLocator, MultipleLocator

labels = [f'({chr(97 + i)})' for i in range(len(axes))]

for idx, (name, preds) in enumerate(predictions.items()):
    ax = axes[idx]
    
    # Extract data
    y_train_true, y_train_pred = preds['train']
    y_val_true, y_val_pred = preds['val']
    y_test_true, y_test_pred = preds['test']
    
    # Calculate R² scores
    r2_train = r2_score(y_train_true, y_train_pred)
    r2_val = r2_score(y_val_true, y_val_pred)
    r2_test = r2_score(y_test_true, y_test_pred)
    
    # Plot training data
    ax.scatter(y_train_true, y_train_pred, alpha=0.5, s=60, 
              c="#4635A5D3", marker='s', edgecolors="#34248E76", linewidths=1,
              label=f'Training (R² = {r2_train:.3f})')
    
    # Plot validation data
    ax.scatter(y_val_true, y_val_pred, alpha=0.6, s=60, 
              c="#0784166e", marker='^', edgecolors="#074e0f6d", linewidths=1,
              label=f'Validation (R² = {r2_val:.3f})')
    
    # Plot test data
    ax.scatter(y_test_true, y_test_pred, alpha=0.7, s=60, 
              c="#e3000091", marker='o', edgecolors="#77090991", linewidths=1,
              label=f'Test (R² = {r2_test:.3f})')
    
    # Perfect prediction line
    ax.plot([y_min, y_max], [y_min, y_max], 'k--', lw=2, 
            label='Perfect Prediction')
    
    # Axis labels and formatting
    ax.set_xlabel('Measured Impedance |Z| (Ω·cm²)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Predicted Impedance |Z| (Ω·cm²)', fontsize=13, fontweight='bold')
    ax.set_title(f'{name}', fontsize=14, fontweight='bold')
    ax.set_xlim([y_min, y_max])
    ax.set_ylim([y_min, y_max])
    
    # Tick formatting
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax.tick_params(axis='both', which='minor', bottom=True, left=True, 
                   length=4, width=1)
    ax.tick_params(axis='both', which='major', labelsize=11, 
                   length=7, width=1.5)
    
    # Grid
    ax.grid(which='major', alpha=0.3, linestyle='--', linewidth=0.8)
    ax.grid(which='minor', alpha=0.15, linestyle=':', linewidth=0.5)
    
    # Legend with all R² values
    legend = ax.legend(loc='lower right', fontsize=10, framealpha=0.95,
                      ncol=1, handletextpad=0.5, columnspacing=0.5)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('#999999')
    legend.get_frame().set_linewidth(1.0)
    
    # Add panel label (a, b, c, d)
    ax.text(0.02, 0.98, labels[idx],
            transform=ax.transAxes,
            fontsize=18, fontweight='bold', 
            va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='none', alpha=0.9, 
                     edgecolor='none', linewidth=0.5))
    
    # Add CV R² in upper right corner for reference
    model_results = results_df[results_df['Model'] == name].iloc[0]
    ax.text(0.98, 0.04, 
            f"CV R² = {model_results['R² CV Mean']:.3f} ± {model_results['R² CV Std']:.3f}", 
            transform=ax.transAxes, fontsize=9, 
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='none', alpha=0.8,
                     edgecolor='none', linewidth=0.5))

plt.tight_layout()
plt.savefig(f'{project_path}parity_plots_all_models.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close('all')

# ============================================================================
# Plot 3: Validation R² & Validation RMSE Scores for Models
# ============================================================================
models_sorted = results_df['Model'].values
r2_val = results_df['R² (Val)'].values
rmse_val = results_df['RMSE (Val)'].values

# Create bar plot comparing model performances on validation set
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Left panel: Validation R² Scores
ax1.bar(models_sorted, r2_val, 
        color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], alpha=0.7,
        edgecolor='none', linewidth=0.5)
ax1.set_xlabel('Models', fontsize=12, fontweight='bold')
ax1.set_ylabel('Validation R² Score', fontsize=12, fontweight='bold')
ax1.set_title('Validation Set R² Scores', fontsize=13, fontweight='bold')
ax1.set_ylim([0, 1])
ax1.yaxis.set_major_locator(MultipleLocator(0.1))
ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
ax1.tick_params(axis='y', which='major', labelsize=11, length=7, width=1.5)
ax1.tick_params(axis='y', which='minor', left=True, length=4, width=1)
ax1.tick_params(axis='x', which='major', labelsize=11, rotation=45)

ax1.grid(which='major', alpha=0.3, linestyle='--', axis='y')
ax1.grid(which='minor', alpha=0.15, linestyle=':', axis='y')

# Add value labels on bars
for i, r2 in enumerate(r2_val):
    ax1.text(i, r2 + 0.02, f'{r2:.3f}', ha='center', fontsize=10, fontweight='bold')

# Right panel: Validation RMSE Scores
ax2.bar(models_sorted, rmse_val, 
        color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], alpha=0.7,
        edgecolor='none', linewidth=0.5)
ax2.set_xlabel('Models', fontsize=12, fontweight='bold')
ax2.set_ylabel('Validation RMSE (Ω·cm²)', fontsize=12, fontweight='bold')
ax2.set_title('Validation Set RMSE', fontsize=13, fontweight='bold')
ax2.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
ax2.yaxis.set_minor_locator(AutoMinorLocator(2))
ax2.tick_params(axis='y', which='major', labelsize=11, length=7, width=1.5)
ax2.tick_params(axis='y', which='minor', left=True, length=4, width=1)
ax2.tick_params(axis='x', which='major', labelsize=11, rotation=45)
ax2.grid(which='major', alpha=0.3, linestyle='--', axis='y')
ax2.grid(which='minor', alpha=0.15, linestyle=':', axis='y')

# Add value labels on bars
for i, rmse in enumerate(rmse_val):
    ax2.text(i, rmse + (rmse_val.max() * 0.02), f'{rmse:.0f}', 
             ha='center', fontsize=10, fontweight='bold')

# Add panel labels
ax1.text(0.02, 0.98, '(a)', transform=ax1.transAxes,
         fontsize=16, fontweight='bold', va='top', ha='left',
         bbox=dict(boxstyle='round', facecolor='none', alpha=0.8,
                   edgecolor='none'))

ax2.text(0.02, 0.98, '(b)', transform=ax2.transAxes,
         fontsize=16, fontweight='bold', va='top', ha='left',
         bbox=dict(boxstyle='round', facecolor='none', alpha=0.8,
                   edgecolor='none'))

plt.tight_layout()
plt.savefig(f'{project_path}model_comparison_validation_bars.png', dpi=300, bbox_inches='tight')
plt.show()


# Print best model summary
best_model = results_df.iloc[0]
print("\n" + "="*60)
print("BEST MODEL SUMMARY")
print("="*60)
print(f"Best Model: {best_model['Model']}")
print(f"Cross-Validation R²: {best_model['R² CV Mean']:.4f} ± {best_model['R² CV Std']:.4f}")
print(f"Test Set R²: {best_model['R² (Test)']:.4f}")
print(f"Test Set RMSE: {best_model['RMSE (Test)']:.4f}")
print(f"Test Set MAE: {best_model['MAE (Test)']:.4f}")

# Optional: Save scalers for later use
import joblib
joblib.dump(scaler_X, f'{project_path}scaler_X.pkl')
joblib.dump(scaler_y, f'{project_path}scaler_y.pkl')
joblib.dump(pca, f'{project_path}pca.pkl')
print("\nSaved scalers and PCA for future use:")
print("  - scaler_X.pkl")
print("  - scaler_y.pkl")
print("  - pca.pkl")