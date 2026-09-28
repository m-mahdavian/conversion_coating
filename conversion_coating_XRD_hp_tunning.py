import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import randint, uniform
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set seed for reproducibility
SEED = 42
np.random.seed(SEED)

print("="*80)
print("RANDOM FOREST HYPERPARAMETER TUNING - ENHANCED SEARCH")
print("="*80)

# ============================================================================
# STEP 1: LOAD SAVED PREPROCESSORS AND PREPARE DATA
# ============================================================================
print("\n STEP 1: Loading saved preprocessors and preparing data...")
print("-"*40)

project_path = 'D:\\R0\\Supporting Information\\'

# Load saved preprocessors
scaler_X = joblib.load(f'{project_path}scaler_X.pkl')
scaler_y = joblib.load(f'{project_path}scaler_y.pkl')
pca = joblib.load(f'{project_path}pca.pkl')

print(" Loaded saved preprocessors:")
print("   - scaler_X.pkl")
print("   - scaler_y.pkl")
print("   - pca.pkl")

# Load and prepare data
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

df_raw_xrd = pd.read_excel(f'{project_path}XRD.xlsx')
two_theta_values = df_raw_xrd.columns[12:2819].astype(float)
mask = (two_theta_values >= 40) & (two_theta_values <= 70)
selected_column_indices = np.where(mask)[0] + 12
df_X_xrd_filtered = df_raw_xrd.iloc[:, selected_column_indices]

# Apply saved scaler and PCA
df_scaled = scaler_X.transform(df_X_xrd_filtered)
df_pca = pca.transform(df_scaled)

n_components = df_pca.shape[1]
df_reduced = pd.DataFrame(df_pca, columns=[f'PC{i+1}' for i in range(n_components)])
df_reduced.index = df_X_xrd_filtered.index

df_repeated = pd.concat([df_reduced] * 8, axis=0, ignore_index=True)

X_df_final = pd.concat([X_df, df_repeated], axis=1)

# Apply saved scaler_y
y1_scaled = scaler_y.transform(y1.values.reshape(-1, 1)).ravel()

print(f" Data prepared successfully!")
print(f"   Number of PCA components: {n_components}")
print(f"   Feature matrix shape: {X_df_final.shape}")
print(f"   Target shape: {y1.shape}")

# ============================================================================
# STEP 2: TRAIN-VALIDATION-TEST SPLIT (60-20-20)
# ============================================================================
print("\n STEP 2: Splitting data into train, validation, and test sets...")
print("-"*40)

X_temp, X_test, y_temp_scaled, y_test_scaled = train_test_split(
    X_df_final, y1_scaled, test_size=0.2, random_state=SEED
)

X_train, X_val, y_train_scaled, y_val_scaled = train_test_split(
    X_temp, y_temp_scaled, test_size=0.25, random_state=SEED
)

print(f" Data split complete!")
print(f"   Training set: {X_train.shape[0]} samples")
print(f"   Validation set: {X_val.shape[0]} samples")
print(f"   Test set: {X_test.shape[0]} samples")

y_train_actual = scaler_y.inverse_transform(y_train_scaled.reshape(-1, 1)).ravel()
y_val_actual = scaler_y.inverse_transform(y_val_scaled.reshape(-1, 1)).ravel()
y_test_actual = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).ravel()

# ============================================================================
# STEP 3: BASELINE RANDOM FOREST MODEL
# ============================================================================
print("\n STEP 3: Training baseline Random Forest model...")
print("-"*40)

baseline_rf = RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1)
baseline_rf.fit(X_train, y_train_scaled)

y_train_pred_scaled = baseline_rf.predict(X_train)
y_val_pred_scaled = baseline_rf.predict(X_val)
y_test_pred_scaled = baseline_rf.predict(X_test)

y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).ravel()
y_val_pred = scaler_y.inverse_transform(y_val_pred_scaled.reshape(-1, 1)).ravel()
y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).ravel()

baseline_r2_train = r2_score(y_train_actual, y_train_pred)
baseline_r2_val = r2_score(y_val_actual, y_val_pred)
baseline_r2_test = r2_score(y_test_actual, y_test_pred)
baseline_rmse_val = np.sqrt(mean_squared_error(y_val_actual, y_val_pred))
baseline_rmse_test = np.sqrt(mean_squared_error(y_test_actual, y_test_pred))
baseline_mae_test = mean_absolute_error(y_test_actual, y_test_pred)

print(f" Baseline Random Forest:")
print(f"   Train R²: {baseline_r2_train:.4f}")
print(f"   Validation R²: {baseline_r2_val:.4f}")
print(f"   Validation RMSE: {baseline_rmse_val:.2f}")
print(f"   Test R²: {baseline_r2_test:.4f}")
print(f"   Test RMSE: {baseline_rmse_test:.2f}")
print(f"   Train-Val Gap: {baseline_r2_train - baseline_r2_val:.4f}")

# ============================================================================
# STEP 4: ENHANCED HYPERPARAMETER TUNING
# ============================================================================
print("\n STEP 4: Enhanced hyperparameter tuning...")
print("-"*40)

# EXPANDED parameter distributions using scipy.stats for continuous sampling
param_dist_enhanced = {
    'n_estimators': randint(100, 1000),  # Wider range: 100 to 1000
    'max_depth': [None, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80],
    'min_samples_split': randint(2, 30),  # Continuous integer range
    'min_samples_leaf': randint(1, 20),   # Wider range for regularization
    'max_features': ['sqrt', 'log2', 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'bootstrap': [True, False],
    'max_samples': uniform(0.5, 0.5),  # Continuous: 0.5 to 1.0
    'min_impurity_decrease': uniform(0, 0.01),  # New: pruning parameter
    'ccp_alpha': uniform(0, 0.02)  # New: complexity parameter for pruning
}

print(" Strategy: HalvingRandomSearchCV for efficient exploration")
print("   - 200 initial candidates with successive halving")
print("   - Scoring: neg_mean_squared_error")
print("   - 3-fold CV for stability")
print("   (This may take 5-10 minutes)")

# Use HalvingRandomSearchCV for more efficient search
halving_search = HalvingRandomSearchCV(
    RandomForestRegressor(random_state=SEED, n_jobs=-1),
    param_distributions=param_dist_enhanced,
    n_candidates=200,  # Start with 200 candidates
    factor=3,  # Keep top 1/3 each iteration
    cv=3,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    random_state=SEED,
    verbose=1
)

halving_search.fit(X_train, y_train_scaled)

# Get best parameters from halving search
best_params_halving = halving_search.best_params_
print(f"\n Best parameters from HalvingRandomSearchCV:")
for param, value in best_params_halving.items():
    print(f"   {param}: {value}")
print(f"   Best CV Score: {halving_search.best_score_:.4f}")

# ============================================================================
# STEP 5: REFINED RANDOM SEARCH AROUND BEST PARAMETERS
# ============================================================================
print("\n STEP 5: Refined random search around best parameters...")
print("-"*40)

# Create focused distributions around best parameters
refined_param_dist = {
    'n_estimators': randint(
        max(50, best_params_halving['n_estimators'] - 200),
        best_params_halving['n_estimators'] + 200
    ),
    'max_depth': [
        None,
        max(5, (best_params_halving['max_depth'] or 30) - 10),
        best_params_halving['max_depth'],
        (best_params_halving['max_depth'] or 30) + 10,
        (best_params_halving['max_depth'] or 30) + 20
    ],
    'min_samples_split': randint(
        max(2, best_params_halving['min_samples_split'] - 5),
        best_params_halving['min_samples_split'] + 5
    ),
    'min_samples_leaf': randint(
        max(1, best_params_halving['min_samples_leaf'] - 3),
        best_params_halving['min_samples_leaf'] + 3
    ),
    'max_features': [
        best_params_halving['max_features'],
        'sqrt' if best_params_halving['max_features'] != 'sqrt' else 'log2',
        max(0.1, best_params_halving['max_features'] - 0.2 if isinstance(best_params_halving['max_features'], float) else 0.5),
        min(1.0, best_params_halving['max_features'] + 0.2 if isinstance(best_params_halving['max_features'], float) else 0.7)
    ],
    'bootstrap': [best_params_halving['bootstrap']],
    'max_samples': uniform(
        max(0.5, best_params_halving['max_samples'] - 0.1),
        min(0.5, 1.0 - best_params_halving['max_samples'] + 0.1)
    ),
    'min_impurity_decrease': uniform(0, best_params_halving['min_impurity_decrease'] * 2),
    'ccp_alpha': uniform(0, best_params_halving['ccp_alpha'] * 2)
}

print("   Running RandomizedSearchCV with 100 iterations on refined space...")

refined_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=SEED, n_jobs=-1),
    param_distributions=refined_param_dist,
    n_iter=100,
    cv=5,  # More CV folds for final selection
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    random_state=SEED,
    verbose=1
)

refined_search.fit(X_train, y_train_scaled)

final_rf = refined_search.best_estimator_
best_params = refined_search.best_params_

print(f"\n Final best parameters after refined search:")
for param, value in best_params.items():
    print(f"   {param}: {value}")
print(f"   Best CV Score: {refined_search.best_score_:.4f}")

# ============================================================================
# STEP 6: FEATURE IMPORTANCE ANALYSIS
# ============================================================================
print("\n STEP 6: Feature importance analysis...")
print("-"*40)

feature_names = ['Time', 'Coating'] + [f'PC{i+1}' for i in range(n_components)]
importances = final_rf.feature_importances_
indices = np.argsort(importances)[::-1]

print(" Top 10 most important features:")
for i in range(min(10, len(feature_names))):
    print(f"   {i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

# ============================================================================
# STEP 7: FINAL EVALUATION
# ============================================================================
print("\n STEP 7: Final evaluation on all sets...")
print("-"*40)

# Validation predictions
y_val_pred_scaled_final = final_rf.predict(X_val)
y_val_pred_final = scaler_y.inverse_transform(y_val_pred_scaled_final.reshape(-1, 1)).ravel()

# Test predictions
y_test_pred_scaled_final = final_rf.predict(X_test)
y_test_pred_final = scaler_y.inverse_transform(y_test_pred_scaled_final.reshape(-1, 1)).ravel()

# Training predictions
y_train_pred_scaled_final = final_rf.predict(X_train)
y_train_pred_final = scaler_y.inverse_transform(y_train_pred_scaled_final.reshape(-1, 1)).ravel()

# Calculate all metrics
final_r2_train = r2_score(y_train_actual, y_train_pred_final)
final_r2_val = r2_score(y_val_actual, y_val_pred_final)
final_r2_test = r2_score(y_test_actual, y_test_pred_final)
final_rmse_val = np.sqrt(mean_squared_error(y_val_actual, y_val_pred_final))
final_rmse_test = np.sqrt(mean_squared_error(y_test_actual, y_test_pred_final))
final_mae_val = mean_absolute_error(y_val_actual, y_val_pred_final)
final_mae_test = mean_absolute_error(y_test_actual, y_test_pred_final)

# Cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=SEED)
final_cv_scores = cross_val_score(final_rf, X_train, y_train_scaled, cv=kfold, scoring='r2')

print(f" Final Tuned Random Forest Performance:")
print(f"   Training R²: {final_r2_train:.4f}")
print(f"   Validation R²: {final_r2_val:.4f}")
print(f"   Validation RMSE: {final_rmse_val:.2f}")
print(f"   Validation MAE: {final_mae_val:.2f}")
print(f"   Test R²: {final_r2_test:.4f}")
print(f"   Test RMSE: {final_rmse_test:.2f}")
print(f"   Test MAE: {final_mae_test:.2f}")
print(f"   CV R²: {final_cv_scores.mean():.4f} ± {final_cv_scores.std():.4f}")

train_val_gap = final_r2_train - final_r2_val
print(f"   Train-Val Gap: {train_val_gap:.4f}")

# ============================================================================
# STEP 8: PERFORMANCE COMPARISON
# ============================================================================
print("\n STEP 8: Performance comparison...")
print("-"*40)

# Calculate improvements
improvement_r2_val = ((final_r2_val - baseline_r2_val) / abs(baseline_r2_val)) * 100 if baseline_r2_val != 0 else 0
improvement_r2_test = ((final_r2_test - baseline_r2_test) / abs(baseline_r2_test)) * 100 if baseline_r2_test != 0 else 0
improvement_rmse_val = ((baseline_rmse_val - final_rmse_val) / baseline_rmse_val) * 100 if baseline_rmse_val != 0 else 0
improvement_rmse_test = ((baseline_rmse_test - final_rmse_test) / baseline_rmse_test) * 100 if baseline_rmse_test != 0 else 0

comparison_df = pd.DataFrame({
    'Model': ['Baseline RF', 'Tuned RF'],
    'Val_R2': [baseline_r2_val, final_r2_val],
    'Val_RMSE': [baseline_rmse_val, final_rmse_val],
    'Test_R2': [baseline_r2_test, final_r2_test],
    'Test_RMSE': [baseline_rmse_test, final_rmse_test],
    'CV_R2_Mean': [np.nan, final_cv_scores.mean()],
    'CV_R2_Std': [np.nan, final_cv_scores.std()],
    'Train_Val_Gap': [baseline_r2_train - baseline_r2_val, train_val_gap]
})

print("\n" + comparison_df.to_string(index=False, float_format=lambda x: '{:.4f}'.format(x)))

print(f"\n Improvements:")
print(f"   Validation R²: {improvement_r2_val:+.2f}%")
print(f"   Test R²: {improvement_r2_test:+.2f}%")
print(f"   Validation RMSE: {improvement_rmse_val:+.2f}%")
print(f"   Test RMSE: {improvement_rmse_test:+.2f}%")

if abs(improvement_r2_test) < 1.0 and abs(improvement_rmse_test) < 3.0:
    print(f"\n ⚠ NOTE: Improvements are marginal (<1% R², <3% RMSE)")
    print(f"   The enhanced search confirms baseline model is near-optimal.")
    print(f"   Recommendation: Use baseline RandomForestRegressor(n_estimators=100)")

# ============================================================================
# STEP 9: VISUALIZATIONS
# ============================================================================
print("\n STEP 9: Creating visualizations...")
print("-"*40)

# SET FONT PARAMETERS BEFORE CREATING ANY FIGURES
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Palatino', 'serif'],
    'font.size': 12,
    'axes.titlesize': 13,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'mathtext.fontset': 'stix'
})

from matplotlib.ticker import AutoMinorLocator, MaxNLocator, MultipleLocator

fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# Plot 1: Parity plot for tuned model
ax = axes[0, 0]
all_values = np.concatenate([y_train_actual, y_val_actual, y_test_actual])
y_min, y_max = all_values.min(), all_values.max()
margin = (y_max - y_min) * 0.05

ax.scatter(y_train_actual, y_train_pred_final, alpha=0.5, s=40, 
           c='blue', marker='s', edgecolors='darkblue', linewidth=0.5,
           label=f'Training (R²={final_r2_train:.3f})', zorder=2)
ax.scatter(y_val_actual, y_val_pred_final, alpha=0.7, s=50, 
           c='green', marker='^', edgecolors='darkgreen', linewidth=1.5,
           label=f'Validation (R²={final_r2_val:.3f})', zorder=3)
ax.scatter(y_test_actual, y_test_pred_final, alpha=0.7, s=50, 
           c='red', marker='o', edgecolors='darkred', linewidth=1.5,
           label=f'Test (R²={final_r2_test:.3f})', zorder=3)
ax.plot([y_min-margin, y_max+margin], [y_min-margin, y_max+margin], 'k--', lw=2, label='Perfect', zorder=1)

ax.set_xlabel('Measured Impedance |Z| (Ω.cm²)', fontsize=14, fontweight='bold')
ax.set_ylabel('Predicted Impedance |Z| (Ω.cm²)', fontsize=14, fontweight='bold')
ax.set_title('Parity Plot - Tuned Random Forest', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
ax.set_xlim([y_min-margin, y_max+margin])
ax.set_ylim([y_min-margin, y_max+margin])

ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
ax.xaxis.set_minor_locator(AutoMinorLocator(4))
ax.yaxis.set_minor_locator(AutoMinorLocator(4))
ax.tick_params(axis='both', which='minor', bottom=True, left=True, length=4, width=1)
ax.tick_params(axis='both', which='major', labelsize=10, length=7, width=1.5)
ax.grid(which='major', alpha=0.3, linestyle='--', linewidth=0.8)
ax.grid(which='minor', alpha=0.15, linestyle=':', linewidth=0.5)
ax.text(0.02, 0.98, '(a)', transform=ax.transAxes, fontsize=16, fontweight='bold', va='top')

# Plot 2: Comparison bar chart with dual axes
ax = axes[0, 1]
r2_metrics = ['Val R²', 'Test R²']
rmse_metrics = ['Val RMSE', 'Test RMSE']

r2_baseline = [baseline_r2_val, baseline_r2_test]
r2_tuned = [final_r2_val, final_r2_test]
rmse_baseline = [baseline_rmse_val, baseline_rmse_test]
rmse_tuned = [final_rmse_val, final_rmse_test]

x_r2 = np.arange(len(r2_metrics))
x_rmse = np.arange(len(r2_metrics), len(r2_metrics) + len(rmse_metrics))
x_all = np.arange(len(r2_metrics) + len(rmse_metrics))
width = 0.35

ax2 = ax.twinx()

bars1_r2 = ax.bar(x_r2 - width/2, r2_baseline, width, label='Baseline RF', 
                  alpha=0.9, color='lightcoral', edgecolor='darkred', linewidth=1.5)
bars2_r2 = ax.bar(x_r2 + width/2, r2_tuned, width, label='Tuned RF', 
                  alpha=0.9, color='steelblue', edgecolor='darkblue', linewidth=1.5)
bars1_rmse = ax2.bar(x_rmse - width/2, rmse_baseline, width, 
                     alpha=0.9, color='lightcoral', edgecolor='darkred', linewidth=1.5, hatch='//')
bars2_rmse = ax2.bar(x_rmse + width/2, rmse_tuned, width, 
                     alpha=0.9, color='steelblue', edgecolor='darkblue', linewidth=1.5, hatch='//')

ax.set_xticks(x_all)
ax.set_xticklabels(r2_metrics + rmse_metrics, fontsize=10)
ax.set_ylabel('R² Score', fontsize=12, fontweight='bold', color='darkblue')
ax2.set_ylabel('RMSE (Ω.cm²)', fontsize=12, fontweight='bold', color='darkred')
ax.set_ylim([0, 1.1])
ax.yaxis.set_major_locator(MultipleLocator(0.2))
ax.yaxis.set_minor_locator(AutoMinorLocator(4))
ax.tick_params(axis='y', labelcolor='darkblue', labelsize=10)
ax.tick_params(axis='y', which='minor', left=True, length=4, width=1)
ax2.tick_params(axis='y', labelcolor='darkred', labelsize=10)
ax2.tick_params(axis='y', which='minor', right=True, length=4, width=1)
ax.axvline(x=1.5, color='black', linestyle='--', linewidth=1, alpha=0.5)

# Add value labels
for bars in [bars1_r2, bars2_r2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02, f'{height:.3f}', 
                ha='center', va='bottom', fontsize=8, fontweight='bold')
for bars in [bars1_rmse, bars2_rmse]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 20, f'{height:.0f}', 
                ha='center', va='bottom', fontsize=8, fontweight='bold')

# Improvement annotations
r2_improvements = [improvement_r2_val, improvement_r2_test]
for i, imp in enumerate(r2_improvements):
    color = 'green' if imp > 0.5 else ('red' if imp < -0.5 else 'gray')
    arrow = '↑' if imp > 0.5 else ('↓' if imp < -0.5 else '→')
    max_height = max(r2_baseline[i], r2_tuned[i])
    ax.text(i, max_height + 0.08, f'{arrow}{abs(imp):.1f}%', 
            ha='center', fontsize=8, color=color, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='none', edgecolor='none'))

rmse_improvements = [improvement_rmse_val, improvement_rmse_test]
for i, imp in enumerate(rmse_improvements):
    color = 'green' if imp > 1.0 else ('red' if imp < -1.0 else 'gray')
    arrow = '↓' if imp > 1.0 else ('↑' if imp < -1.0 else '→')
    max_height = max(rmse_baseline[i], rmse_tuned[i])
    ax2.text(i + 2, max_height + 350, f'{arrow}{abs(imp):.1f}%', 
        ha='center', fontsize=8, color=color, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='none', edgecolor='none'))

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='lightcoral', edgecolor='darkred', label='Baseline RF'),
    Patch(facecolor='steelblue', edgecolor='darkblue', label='Tuned RF'),
    Patch(facecolor='white', edgecolor='black', hatch='//', label='RMSE')
]
ax.legend(handles=legend_elements, bbox_to_anchor=(0.5, 0.99), fontsize=8.5, framealpha=0.95)
ax.set_title('Pre vs Post HP Tuning', fontsize=13, fontweight='bold')
ax.grid(which='major', alpha=0.3, linestyle='--', axis='y', linewidth=0.8)
ax.grid(which='minor', alpha=0.15, linestyle=':', axis='y')
rmse_max = max(rmse_baseline + rmse_tuned)
ax2.set_ylim([0, rmse_max * 1.2])
ax2.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
ax2.yaxis.set_minor_locator(AutoMinorLocator(4))
ax.text(0.02, 0.98, '(b)', transform=ax.transAxes, fontsize=16, fontweight='bold', va='top')

# Plot 3: Feature importance (with 4 minor ticks on x-axis)
ax = axes[1, 0]
top_n = min(15, len(feature_names))
ax.barh(range(top_n), importances[indices][:top_n][::-1], align='center',
        color='steelblue', edgecolor='darkblue')
ax.set_yticks(range(top_n))
ax.set_yticklabels([feature_names[indices[i]] for i in range(top_n)][::-1], fontsize=9)
ax.set_xlabel('Importance', fontsize=14, fontweight='bold')
ax.set_title('Feature Importance', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

# Add 4 minor ticks to x-axis
ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
ax.xaxis.set_minor_locator(AutoMinorLocator(4))
ax.tick_params(axis='x', which='minor', bottom=True, length=4, width=1)
ax.tick_params(axis='x', which='major', labelsize=10, length=7, width=1.5)
ax.grid(which='major', alpha=0.3, linestyle='--', axis='x', linewidth=0.8)
ax.grid(which='minor', alpha=0.15, linestyle=':', axis='x')

ax.text(0.02, 0.98, '(c)', transform=ax.transAxes, fontsize=16, fontweight='bold', va='top')

# Plot 4: Learning curves (with 4 minor ticks on both x and y axes)
ax = axes[1, 1]
from sklearn.model_selection import learning_curve
train_sizes, train_scores, val_scores = learning_curve(
    final_rf, X_train, y_train_scaled, cv=5, n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10), scoring='r2'
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color='blue')
ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2, color='green')
ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score', linewidth=2)
ax.plot(train_sizes, val_mean, 'o-', color='green', label='Cross-validation score', linewidth=2)
ax.set_xlabel('Training Examples', fontsize=12, fontweight='bold')
ax.set_ylabel('R² Score', fontsize=12, fontweight='bold')
ax.set_title('Learning Curves', fontsize=13, fontweight='bold')
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)

# Add 4 minor ticks to both x and y axes
ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
ax.xaxis.set_minor_locator(AutoMinorLocator(4))
ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
ax.yaxis.set_minor_locator(AutoMinorLocator(4))
ax.tick_params(axis='both', which='minor', bottom=True, left=True, length=4, width=1)
ax.tick_params(axis='both', which='major', labelsize=10, length=7, width=1.5)
ax.grid(which='major', alpha=0.3, linestyle='--', linewidth=0.8)
ax.grid(which='minor', alpha=0.15, linestyle=':', linewidth=0.5)

ax.text(0.02, 0.98, '(d)', transform=ax.transAxes, fontsize=16, fontweight='bold', va='top')

plt.tight_layout()
plt.savefig(f'{project_path}rf_tuning_results_enhanced.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# STEP 10: SAVE MODEL AND RESULTS
# ============================================================================
print("\n STEP 10: Saving tuned model and results...")
print("-"*40)

joblib.dump(final_rf, f'{project_path}best_random_forest_tuned.pkl')

with open(f'{project_path}rf_tuning_results_enhanced.txt', 'w') as f:
    f.write("="*60 + "\n")
    f.write("RANDOM FOREST ENHANCED TUNING RESULTS\n")
    f.write("="*60 + "\n\n")
    f.write("BASELINE MODEL (n_estimators=100):\n")
    f.write(f"  Validation R²: {baseline_r2_val:.4f}\n")
    f.write(f"  Validation RMSE: {baseline_rmse_val:.2f}\n")
    f.write(f"  Test R²: {baseline_r2_test:.4f}\n")
    f.write(f"  Test RMSE: {baseline_rmse_test:.2f}\n\n")
    f.write("TUNED MODEL:\n")
    f.write(f"Best Parameters:\n")
    for param, value in best_params.items():
        f.write(f"  {param}: {value}\n")
    f.write(f"\nFinal Performance:\n")
    f.write(f"  Training R²: {final_r2_train:.4f}\n")
    f.write(f"  Validation R²: {final_r2_val:.4f}\n")
    f.write(f"  Validation RMSE: {final_rmse_val:.2f}\n")
    f.write(f"  Validation MAE: {final_mae_val:.2f}\n")
    f.write(f"  Test R²: {final_r2_test:.4f}\n")
    f.write(f"  Test RMSE: {final_rmse_test:.2f}\n")
    f.write(f"  Test MAE: {final_mae_test:.2f}\n")
    f.write(f"  CV R²: {final_cv_scores.mean():.4f} ± {final_cv_scores.std():.4f}\n")
    f.write(f"  Train-Val Gap: {train_val_gap:.4f}\n\n")
    f.write(f"IMPROVEMENTS:\n")
    f.write(f"  Validation R²: {improvement_r2_val:+.2f}%\n")
    f.write(f"  Test R²: {improvement_r2_test:+.2f}%\n")
    f.write(f"  Validation RMSE: {improvement_rmse_val:+.2f}%\n")
    f.write(f"  Test RMSE: {improvement_rmse_test:+.2f}%\n\n")
    f.write(f"TOP FEATURES:\n")
    for i in range(min(10, len(feature_names))):
        f.write(f"  {i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}\n")

print(" Files saved:")
print("   - best_random_forest_tuned.pkl")
print("   - rf_tuning_results_enhanced.txt")
print("   - rf_tuning_results_enhanced.png")

print("\n" + "="*80)
print(" ENHANCED RANDOM FOREST TUNING COMPLETED!")
print("="*80)
print(f" Search method: HalvingRandomSearchCV (200 candidates) + RandomizedSearchCV (100 iterations)")
print(f" Parameters explored: {len(param_dist_enhanced)} with expanded ranges")
if abs(improvement_r2_test) < 1.0 and abs(improvement_rmse_test) < 3.0:
    print(f"\n CONCLUSION: Baseline model is near-optimal")
    print(f"   The enhanced search confirms minimal room for improvement.")
    print(f"   Recommend using baseline RandomForestRegressor(n_estimators=100)")
print("="*80)