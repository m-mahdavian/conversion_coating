import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Set seed for reproducibility
SEED = 42
np.random.seed(SEED)

print("="*80)
print("PREDICTING y1 VALUES FOR EXPERIMENTAL VALIDATION SAMPLE (RANDOM FOREST)")
print("="*80)

# Load models and preprocessors
project_path = 'D:\\R0\\Supporting Information\\'

print("\n Loading saved models and preprocessors...")
# Load tuned Random Forest model
try:
    best_rf = joblib.load(project_path + 'best_random_forest_tuned.pkl')
    print(" Loaded tuned Random Forest model: best_random_forest_tuned.pkl")
except:
    print(" Tuned RF not found, trying baseline RF...")
    try:
        best_rf = joblib.load(project_path + 'best_random_forest_final.pkl')
        print(" Loaded baseline Random Forest model: best_random_forest_final.pkl")
    except:
        print(" ERROR: No Random Forest model found!")
        raise

scaler_X = joblib.load(project_path + 'scaler_X.pkl')
scaler_y = joblib.load(project_path + 'scaler_y.pkl')
pca = joblib.load(project_path + 'pca.pkl')

print(" Models and preprocessors loaded successfully!")
print(f" Model type: {type(best_rf).__name__}")
print(f" Number of trees: {best_rf.n_estimators}")

# ============================================================================
# LOAD AND PROCESS VALIDATION XRD SAMPLE
# ============================================================================
print("\n Loading experimental validation sample...")

# Load validation XRD data
validation_file = project_path + 'XRD_validation.xlsx'
df_validation = pd.read_excel(validation_file)

print(f"Validation data shape: {df_validation.shape}")

# Get two_theta values (columns 12 to 2819 as in original data)
two_theta_values_full = df_validation.columns[12:2819].astype(float)

# Apply same mask (40-70 degrees)
mask = (two_theta_values_full >= 40) & (two_theta_values_full <= 70)

# Get filtered column indices
selected_column_indices = np.where(mask)[0] + 12

# Extract filtered XRD data
df_validation_filtered = df_validation.iloc[:, selected_column_indices]

# Get the 2θ values for the filtered data
two_theta_values = two_theta_values_full[mask].values

print(f"   Filtered validation data shape: {df_validation_filtered.shape}")
print(f"   2θ range: {two_theta_values.min():.1f}° to {two_theta_values.max():.1f}°")
print(f"   Number of data points: {len(two_theta_values)}")

# ============================================================================
# APPLY PCA TRANSFORMATION AND RECONSTRUCTION
# ============================================================================
print("\n Applying PCA transformation...")

# Scale the validation data using the same scaler
df_validation_scaled = scaler_X.transform(df_validation_filtered)

# Apply PCA transformation
df_validation_pca = pca.transform(df_validation_scaled)

print(f"   PCA components shape: {df_validation_pca.shape}")
print(f"   Number of PCA components: {df_validation_pca.shape[1]}")
print(f"   Explained variance ratio (first 10 PCs): {pca.explained_variance_ratio_[:10].sum():.3f}")

# Reconstruct XRD pattern from PCA components
df_validation_reconstructed_scaled = pca.inverse_transform(df_validation_pca)
df_validation_reconstructed = scaler_X.inverse_transform(df_validation_reconstructed_scaled)

# Calculate reconstruction error
mse_reconstruction = np.mean((df_validation_filtered.iloc[0].values - df_validation_reconstructed[0])**2)
rmse_reconstruction = np.sqrt(mse_reconstruction)
mae_reconstruction = np.mean(np.abs(df_validation_filtered.iloc[0].values - df_validation_reconstructed[0]))

print(f"   Reconstruction MSE: {mse_reconstruction:.6f}")
print(f"   Reconstruction RMSE: {rmse_reconstruction:.6f}")
print(f"   Reconstruction MAE: {mae_reconstruction:.6f}")

# Create DataFrame with PCA components
pca_columns = [f'PC{i+1}' for i in range(df_validation_pca.shape[1])]
df_pca_validation = pd.DataFrame(df_validation_pca, columns=pca_columns)

# ============================================================================
# LOAD ACTUAL EXPERIMENTAL DATA
# ============================================================================
print("\n Loading actual experimental validation data...")

# Load actual experimental data
actual_data_file = project_path + 'epoxy_coated_validation.xlsx'
df_actual = pd.read_excel(actual_data_file)

print(f"Actual experimental data shape: {df_actual.shape}")

# Time points matching prediction times
time_actual = np.array([1, 5, 24, 168])

# Extract y1_actual values from the last row, columns 13 to 44 (inclusive)
# Assuming the data structure has the last row containing y1_actual values
y1_actual_values = df_actual.iloc[0:, 43].values.astype(float)

# Since we have 32 columns (13 to 44 inclusive) but only 4 time points,
# we need to select only the relevant columns that correspond to our time points
if len(y1_actual_values) > 4:
    print(f"   Warning: Found {len(y1_actual_values)} values, using first 4")
    y1_actual_values = y1_actual_values[:4]
elif len(y1_actual_values) < 4:
    print(f"   Warning: Found only {len(y1_actual_values)} values, padding with NaN")
    y1_actual_values = np.pad(y1_actual_values, (0, 4-len(y1_actual_values)), 
                              constant_values=np.nan)

print(f"   Time points: {time_actual}")
print(f"   y1_actual values: {y1_actual_values}")

# ============================================================================
# PREPARE FEATURES AND MAKE PREDICTIONS
# ============================================================================
print("\n Preparing features and making predictions...")

# Time points to predict
time_points = np.array([1, 5, 24, 168])
coating_type = 1  # Fixed coating type

predictions_results = []

for time in time_points:
    print(f"\n Processing time = {time} hours...")
    
    # Create feature vector: [time, coating, PC1, PC2, ..., PCn]
    features = np.concatenate([[time, coating_type], df_validation_pca[0]])
    features = features.reshape(1, -1)
    
    # Make prediction using Random Forest (scaled)
    y_pred_scaled = best_rf.predict(features)[0]
    
    # Inverse transform to original scale
    y_pred = scaler_y.inverse_transform([[y_pred_scaled]])[0][0]
    
    predictions_results.append({
        'time': time,
        'coating': coating_type,
        'predicted_y1': y_pred
    })
    
    print(f"    Predicted |Z| at {time}h: {y_pred:.2f} Ω.cm²")

results_df = pd.DataFrame(predictions_results)

# ============================================================================
# COMPARE PREDICTIONS WITH ACTUAL DATA
# ============================================================================
print("\n" + "="*80)
print(" COMPARISON WITH ACTUAL EXPERIMENTAL DATA")
print("="*80)

# Create comparison dataframe (same time points for both)
comparison_results = []
for i, time_pred in enumerate(time_points):
    pred_val = results_df[results_df['time'] == time_pred]['predicted_y1'].values[0]
    actual_val = y1_actual_values[i]
    
    if not np.isnan(actual_val):
        abs_error = abs(pred_val - actual_val)
        percent_error = (abs_error / actual_val) * 100 if actual_val != 0 else np.inf
    else:
        abs_error = np.nan
        percent_error = np.nan
    
    comparison_results.append({
        'Time_h': time_pred,
        'Predicted_Z': pred_val,
        'Actual_Z': actual_val,
        'Absolute_Error': abs_error,
        'Percent_Error': percent_error
    })

comparison_df = pd.DataFrame(comparison_results)
print("\n" + comparison_df.to_string(index=False, float_format=lambda x: '{:.2f}'.format(x) if not np.isnan(x) else 'N/A'))

# Calculate overall metrics (excluding NaN values)
valid_indices = ~np.isnan(y1_actual_values)
if np.any(valid_indices):
    predicted_vals = comparison_df['Predicted_Z'].values[valid_indices]
    actual_vals = comparison_df['Actual_Z'].values[valid_indices]
    
    r2 = r2_score(actual_vals, predicted_vals)
    mae = mean_absolute_error(actual_vals, predicted_vals)
    rmse = np.sqrt(mean_squared_error(actual_vals, predicted_vals))
    
    print(f"\n Overall Performance Metrics:")
    print(f"   R² Score: {r2:.4f}")
    print(f"   MAE: {mae:.2f} Ω.cm²")
    print(f"   RMSE: {rmse:.2f} Ω.cm²")
else:
    print("\n No valid actual data for comparison")
    r2 = np.nan
    mae = np.nan
    rmse = np.nan

# ============================================================================
# CREATE VISUALIZATIONS
# ============================================================================
print("\n Generating visualizations...")
plt.rcParams['font.family'] = 'serif'
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
# ============================================================================
# Plot 1: XRD Pattern - Original vs PCA Reconstruction
# ============================================================================
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

ax1.plot(two_theta_values, df_validation_filtered.iloc[0], 
         'b-', linewidth=1.5, alpha=0.8, label='Original XRD Pattern')
ax1.plot(two_theta_values, df_validation_reconstructed[0], 
         'r--', linewidth=1.5, alpha=0.8, label='Reconstructed from PCs')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_xlim([40, 70])

ax1.xaxis.set_minor_locator(AutoMinorLocator(4))
ax1.yaxis.set_minor_locator(AutoMinorLocator(4))
ax1.xaxis.set_major_locator(MultipleLocator(5))
ax1.tick_params(axis='x', which='minor', bottom=True)
ax1.tick_params(axis='y', which='minor', left=True)
ax1.tick_params(axis='x', which='major', labelsize=14)
ax1.tick_params(axis='y', which='major', labelsize=14)

ax1.set_xlabel('2θ (degrees)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Intensity (a.u.)', fontsize=14, fontweight='bold')
ax1.set_title('XRD Pattern: Original vs PCA Reconstruction', 
              fontsize=14, fontweight='bold', pad=15)
ax1.legend(loc='best', fontsize=14, framealpha=0.9)

error_text = f'Reconstruction Error:\nMSE: {mse_reconstruction:.4f}\nRMSE: {rmse_reconstruction:.4f}\nMAE: {mae_reconstruction:.4f}'
ax1.text(0.98, 0.9, error_text, 
         transform=ax1.transAxes, fontsize=9, verticalalignment='center',horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
         family='monospace')
ax1.text(0.02, 0.85, '(a)',
         transform=ax1.transAxes,
         fontsize=18, fontweight='bold', 
         va='top', ha='left')


# ============================================================================
# Plot 2: Impedance vs Time - Predicted vs Actual
# ============================================================================

ax2.plot(results_df['time'], results_df['predicted_y1'], 'o-', color='blue', 
         linewidth=2.5, markersize=10, markerfacecolor='red', markeredgecolor='darkred',
         label='Predicted |Z| from XRD (RF)', alpha=0.9, zorder=4)

ax2.plot(time_actual, y1_actual_values, 's-', color='green', linewidth=2, 
         markersize=10, markerfacecolor='limegreen', markeredgecolor='darkgreen',
         label='Actual Experimental |Z|', alpha=0.8, zorder=3)

for _, row in results_df.iterrows():
    ax2.annotate(f'{row["predicted_y1"]:.0f}', 
                xy=(row['time'], row['predicted_y1']),
                xytext=(10, 10), textcoords='offset points', 
                fontsize=9, fontweight='bold', color='darkred',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

for i, (time_val, actual_val) in enumerate(zip(time_actual, y1_actual_values)):
    if not np.isnan(actual_val):
        ax2.annotate(f'{actual_val:.0f}', 
                    xy=(time_val, actual_val),
                    xytext=(10, -15), textcoords='offset points', 
                    fontsize=9, color='darkgreen',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))

ax2.xaxis.set_minor_locator(AutoMinorLocator(4))
ax2.yaxis.set_minor_locator(AutoMinorLocator(4))
ax2.xaxis.set_major_locator(MultipleLocator(5))
ax2.tick_params(axis='x', which='minor', bottom=True)
ax2.tick_params(axis='y', which='minor', left=True)
ax2.tick_params(axis='x', which='major', labelsize=14)
ax2.tick_params(axis='y', which='major', labelsize=14)

ax2.set_xlabel('Time (hours)', fontsize=14, fontweight='bold')
ax2.set_ylabel('Impedance |Z| (Ω.cm²)', fontsize=14, fontweight='bold')
ax2.set_title('Impedance Prediction vs Actual Experimental Data\nEpoxy Coated Specimens (Random Forest)', 
              fontsize=13, fontweight='bold')
ax2.legend(loc='best', fontsize=10, framealpha=0.9)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_xscale('log')
ax2.set_xticks(time_points)
ax2.set_xticklabels([f'{t}h' for t in time_points])

# Add performance metrics if available
if not np.isnan(r2):
    metrics_text = f'Performance:\nR²: {r2:.4f}\nMAE: {mae:.2f}\nRMSE: {rmse:.2f}'
    ax2.text(0.2, 0.02, metrics_text, 
             transform=ax2.transAxes, fontsize=9, verticalalignment='bottom',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
             family='monospace')

ax2.text(0.02, 0.85, '(b)',
         transform=ax2.transAxes,
         fontsize=18, fontweight='bold', 
         va='top', ha='left')

plt.tight_layout()
plt.savefig(project_path + 'validation_predictions_rf.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n Saving results...")

# Save predictions
results_df.to_excel(project_path + 'validation_predictions_rf.xlsx', index=False)

# Save detailed results
with pd.ExcelWriter(project_path + 'validation_detailed_results_rf.xlsx') as writer:
    # Summary predictions
    results_df.to_excel(writer, sheet_name='Predictions', index=False)
    
    # Comparison with actual data
    comparison_df.to_excel(writer, sheet_name='Comparison', index=False)
    
    # XRD pattern data
    xrd_data = pd.DataFrame({
        '2θ_degrees': two_theta_values,
        'Original_Intensity': df_validation_filtered.iloc[0].values,
        'Reconstructed_Intensity': df_validation_reconstructed[0],
        'Residual': df_validation_filtered.iloc[0].values - df_validation_reconstructed[0]
    })
    xrd_data.to_excel(writer, sheet_name='XRD_Pattern', index=False)
    
    # PCA components
    pca_data = pd.DataFrame([df_validation_pca[0]], columns=pca_columns)
    pca_data.to_excel(writer, sheet_name='PCA_Components', index=False)
    
    # Performance metrics
    metrics_data = pd.DataFrame({
        'Metric': ['R²', 'MAE', 'RMSE', 'Reconstruction_MSE', 'Reconstruction_RMSE', 'Reconstruction_MAE'],
        'Value': [r2, mae, rmse, mse_reconstruction, rmse_reconstruction, mae_reconstruction]
    })
    metrics_data.to_excel(writer, sheet_name='Performance_Metrics', index=False)

print(" Results saved to:")
print(f"   - {project_path}validation_predictions_rf.xlsx")
print(f"   - {project_path}validation_detailed_results_rf.xlsx")
print(f"   - {project_path}validation_predictions_rf.png")

# ============================================================================
# PRINT FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print(" FINAL SUMMARY (RANDOM FOREST)")
print("="*80)

print("\nXRD Pattern Analysis:")
print("-" * 40)
print(f"  Data points after filtering (40-70°): {len(two_theta_values)}")
print(f"  PCA components used: {df_validation_pca.shape[1]}")
print(f"  Reconstruction MSE: {mse_reconstruction:.6f}")

print("\nImpedance Predictions vs Actual Data:")
print("-" * 40)
print(f"  {'Time (h)':<10} {'Predicted':<12} {'Actual':<12} {'Error':<10} {'Error %':<10}")
print(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*10} {'-'*10}")
for _, row in comparison_df.iterrows():
    if not np.isnan(row['Actual_Z']):
        print(f"  {row['Time_h']:<10.0f} {row['Predicted_Z']:<12.2f} {row['Actual_Z']:<12.2f} "
              f"{row['Absolute_Error']:<10.2f} {row['Percent_Error']:<10.1f}")
    else:
        print(f"  {row['Time_h']:<10.0f} {row['Predicted_Z']:<12.2f} {'N/A':<12} {'N/A':<10} {'N/A':<10}")

if not np.isnan(r2):
    print(f"\nOverall Performance:")
    print("-" * 40)
    print(f"  R² Score: {r2:.4f}")
    print(f"  MAE: {mae:.2f} Ω.cm²")
    print(f"  RMSE: {rmse:.2f} Ω.cm²")

# Calculate degradation rate
print("\nPredicted Degradation Rate:")
print("-" * 40)
for i in range(1, len(results_df)):
    time_diff = results_df.iloc[i]['time'] - results_df.iloc[i-1]['time']
    y_diff = results_df.iloc[i]['predicted_y1'] - results_df.iloc[i-1]['predicted_y1']
    rate = y_diff / time_diff
    print(f"  {results_df.iloc[i-1]['time']:3.0f}h → {results_df.iloc[i]['time']:3.0f}h: "
          f"Δ|Z|/Δt = {rate:.2f} Ω.cm²/h")

print("\n" + "="*80)
print(" ANALYSIS COMPLETED SUCCESSFULLY!")
print("="*80)