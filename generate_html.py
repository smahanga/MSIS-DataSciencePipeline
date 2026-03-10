"""
Generate updated Jobs_Skills_ML_Pipeline.html with Part 2 sections.
Reads existing HTML, inserts new tabs for Hyperparameter Tuning, MLP, and Test-Set Comparison.
"""

import numpy as np
import pandas as pd
import json
import joblib
import os
import io
import base64
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")

# ──────────────────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────────────────
with open(f"{ARTIFACT_DIR}/config.json") as f:
    config = json.load(f)
feature_cols = config['feature_cols']

with open(f"{ARTIFACT_DIR}/test_results.json") as f:
    test_results = json.load(f)

y_test = np.load(f"{ARTIFACT_DIR}/y_test.npy")

grid_params = {}
for key in ['cart', 'rf', 'lgb']:
    with open(f"{ARTIFACT_DIR}/grid_{key}_best_params.json") as f:
        grid_params[key] = json.load(f)

with open(f"{ARTIFACT_DIR}/mlp_history.json") as f:
    mlp_history = json.load(f)

predictions = {}
for name in test_results.keys():
    safe = name.lower().replace(' ', '_').replace('(', '').replace(')', '')
    try:
        predictions[name] = np.load(f"{ARTIFACT_DIR}/y_pred_test_{safe}.npy")
    except FileNotFoundError:
        pass

# ──────────────────────────────────────────────────────────
# HELPER: Convert Plotly figure to embeddable div
# ──────────────────────────────────────────────────────────
_div_counter = [0]
def fig_to_div(fig):
    _div_counter[0] += 1
    div_id = f"part2-plot-{_div_counter[0]}"
    return pio.to_html(fig, include_plotlyjs=False, full_html=False, div_id=div_id)

def tree_to_base64():
    """Render CART tree as base64 PNG."""
    best_cart = joblib.load(f"{ARTIFACT_DIR}/model_cart_tuned.joblib")
    fig_tree, ax = plt.subplots(figsize=(20, 10))
    plot_tree(best_cart, feature_names=feature_cols, filled=True,
              rounded=True, max_depth=3, fontsize=8, ax=ax)
    ax.set_title("Best CART Decision Tree (top 3 levels)", fontsize=14)
    plt.tight_layout()
    buf = io.BytesIO()
    fig_tree.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return b64

def pred_vs_actual_fig(model_name):
    """Create a predicted-vs-actual scatter plot."""
    y_pred = predictions.get(model_name)
    if y_pred is None:
        return ""
    sample_idx = np.random.RandomState(42).choice(len(y_test), min(5000, len(y_test)), replace=False)
    fig = px.scatter(x=y_test[sample_idx], y=y_pred[sample_idx], opacity=0.3,
                     labels={'x': 'Actual num_skills', 'y': 'Predicted num_skills'},
                     title=f'{model_name}: Predicted vs Actual (Test Set)')
    max_val = max(y_test.max(), y_pred.max())
    fig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode='lines',
                             line=dict(dash='dash', color='red'), name='Perfect'))
    fig.update_layout(height=450)
    return fig_to_div(fig)


# ──────────────────────────────────────────────────────────
# BUILD CHART DIVS
# ──────────────────────────────────────────────────────────
print("Generating charts...")

# CART GridSearchCV heatmap
cart_cv = pd.read_json(f"{ARTIFACT_DIR}/grid_cart_cv_results.json")
pivot_cart = cart_cv.pivot_table(values='mean_test_score',
    index='param_min_samples_leaf', columns='param_max_depth')
pivot_cart_rmse = np.sqrt(-pivot_cart)
fig = px.imshow(pivot_cart_rmse, text_auto='.3f', color_continuous_scale='RdYlGn_r',
                title='CART GridSearchCV: Mean CV RMSE',
                labels=dict(x='max_depth', y='min_samples_leaf', color='RMSE'))
fig.update_layout(height=400)
cart_heatmap_div = fig_to_div(fig)

# RF GridSearchCV heatmap
rf_cv = pd.read_json(f"{ARTIFACT_DIR}/grid_rf_cv_results.json")
pivot_rf = rf_cv.pivot_table(values='mean_test_score',
    index='param_max_depth', columns='param_n_estimators')
pivot_rf_rmse = np.sqrt(-pivot_rf)
fig = px.imshow(pivot_rf_rmse, text_auto='.3f', color_continuous_scale='RdYlGn_r',
                title='Random Forest GridSearchCV: Mean CV RMSE',
                labels=dict(x='n_estimators', y='max_depth', color='RMSE'))
fig.update_layout(height=350)
rf_heatmap_div = fig_to_div(fig)

# LightGBM GridSearchCV heatmaps (one per learning rate)
lgb_cv = pd.read_json(f"{ARTIFACT_DIR}/grid_lgb_cv_results.json")
lgb_heatmap_divs = ""
for lr in [0.01, 0.05, 0.1]:
    subset = lgb_cv[lgb_cv['param_learning_rate'] == lr]
    if len(subset) > 0:
        pivot_lgb = subset.pivot_table(values='mean_test_score',
            index='param_max_depth', columns='param_n_estimators')
        pivot_lgb_rmse = np.sqrt(-pivot_lgb)
        fig = px.imshow(pivot_lgb_rmse, text_auto='.3f', color_continuous_scale='RdYlGn_r',
                        title=f'LightGBM GridSearchCV: RMSE (learning_rate={lr})',
                        labels=dict(x='n_estimators', y='max_depth', color='RMSE'))
        fig.update_layout(height=300)
        lgb_heatmap_divs += fig_to_div(fig)

# Tree visualization
tree_b64 = tree_to_base64()

# Predicted vs actual plots
cart_scatter = pred_vs_actual_fig('CART (Tuned)')
rf_scatter = pred_vs_actual_fig('Random Forest (Tuned)')
lgb_scatter = pred_vs_actual_fig('LightGBM (Tuned)')
mlp_scatter = pred_vs_actual_fig('MLP')

# MLP training history
fig = go.Figure()
fig.add_trace(go.Scatter(y=mlp_history['loss'], name='Training Loss', mode='lines'))
fig.add_trace(go.Scatter(y=mlp_history['val_loss'], name='Validation Loss', mode='lines'))
fig.update_layout(title='MLP Loss Curve (MSE)', xaxis_title='Epoch', yaxis_title='MSE', height=400)
mlp_loss_div = fig_to_div(fig)

fig = go.Figure()
fig.add_trace(go.Scatter(y=mlp_history['mae'], name='Training MAE', mode='lines'))
fig.add_trace(go.Scatter(y=mlp_history['val_mae'], name='Validation MAE', mode='lines'))
fig.update_layout(title='MLP MAE Curve', xaxis_title='Epoch', yaxis_title='MAE', height=400)
mlp_mae_div = fig_to_div(fig)

# Model comparison bar charts
summary_rows = []
for name, res in test_results.items():
    summary_rows.append({'Model': name, 'Test RMSE': res['test_rmse'],
                         'Test MAE': res['test_mae'], 'Test R²': res['test_r2']})
summary_df = pd.DataFrame(summary_rows).sort_values('Test RMSE')

fig = px.bar(summary_df, x='Model', y='Test RMSE', color='Test RMSE',
             color_continuous_scale='RdYlGn_r', title='Test-Set RMSE Comparison')
fig.update_layout(height=450, xaxis_tickangle=-30)
rmse_bar_div = fig_to_div(fig)

fig = px.bar(summary_df, x='Model', y='Test R²', color='Test R²',
             color_continuous_scale='RdYlGn', title='Test-Set R² Comparison')
fig.update_layout(height=450, xaxis_tickangle=-30)
r2_bar_div = fig_to_div(fig)

# ──────────────────────────────────────────────────────────
# BUILD HTML SNIPPETS
# ──────────────────────────────────────────────────────────
def metric_card(label, value):
    return f'<div class="metric"><div class="label">{label}</div><div class="value">{value}</div></div>'

cart_bp = grid_params['cart']
rf_bp = grid_params['rf']
lgb_bp = grid_params['lgb']
best_model = summary_df.iloc[0]
mlp_res = test_results.get('MLP', {})
cart_res = test_results.get('CART (Tuned)', {})
rf_res = test_results.get('Random Forest (Tuned)', {})
lgb_res = test_results.get('LightGBM (Tuned)', {})

summary_table_html = '<table><thead><tr><th>Model</th><th>Test RMSE</th><th>Test MAE</th><th>Test R²</th></tr></thead><tbody>'
for _, row in summary_df.iterrows():
    summary_table_html += f'<tr><td style="font-weight:600">{row["Model"]}</td>'
    summary_table_html += f'<td>{row["Test RMSE"]:.4f}</td><td>{row["Test MAE"]:.4f}</td><td>{row["Test R²"]:.4f}</td></tr>'
summary_table_html += '</tbody></table>'


# ──────────────────────────────────────────────────────────
# INSERT INTO EXISTING HTML
# ──────────────────────────────────────────────────────────
print("Reading existing HTML...")
with open('Jobs_Skills_ML_Pipeline.html', 'r') as f:
    html = f.read()

# Add new nav tab buttons
old_nav = '<button class="nav-tab" onclick="showTab(\'shap\')">🔍 SHAP Analysis</button>'
new_nav = '''<button class="nav-tab" onclick="showTab('tune')">🔧 Hyperparameter Tuning</button>
        <button class="nav-tab" onclick="showTab('mlp')">🧠 MLP Neural Network</button>
        <button class="nav-tab" onclick="showTab('testcomp')">📈 Test-Set Comparison</button>
        <button class="nav-tab" onclick="showTab('shap')">🔍 SHAP Analysis</button>'''

html = html.replace(old_nav, new_nav, 1)

# Insert new tab content divs before the SHAP tab
shap_tab_marker = '<!-- ═══ SHAP ANALYSIS ═══ -->' if '<!-- ═══ SHAP ANALYSIS ═══ -->' in html else '<div id="tab-shap"'

# If the comment marker doesn't exist, use the tab div directly
if '<!-- ═══ SHAP ANALYSIS ═══ -->' in html:
    insert_before = '<!-- ═══ SHAP ANALYSIS ═══ -->'
else:
    insert_before = '<div id="tab-shap" class="tab-content">'

new_tabs_html = f'''
<!-- ═══ HYPERPARAMETER TUNING ═══ -->
<div id="tab-tune" class="tab-content">
    <h2>Hyperparameter Tuning (GridSearchCV)</h2>
    <p>All models tuned with <strong>5-fold cross-validation</strong> using <code>GridSearchCV</code>
    (scoring = <code>neg_mean_squared_error</code>). Data split: <strong>70% train / 30% test</strong> with <code>random_state=42</code>.</p>

    <div class="card">
        <h3>2.3 Decision Tree (CART) &mdash; GridSearchCV</h3>
        <p><strong>Best hyperparameters:</strong> <code>max_depth={cart_bp["max_depth"]}</code>,
        <code>min_samples_leaf={cart_bp["min_samples_leaf"]}</code></p>
        <p><strong>Param grid:</strong> max_depth=[3, 5, 7, 10], min_samples_leaf=[5, 10, 20, 50] &rarr; 16 combos &times; 5 folds = 80 fits</p>
        {cart_heatmap_div}
        <div class="grid-4" style="grid-template-columns:repeat(3,1fr); margin-top:16px;">
            {metric_card("Test RMSE", f'{cart_res["test_rmse"]:.4f}')}
            {metric_card("Test MAE", f'{cart_res["test_mae"]:.4f}')}
            {metric_card("Test R²", f'{cart_res["test_r2"]:.4f}')}
        </div>
        <h3 style="margin-top:24px;">Best CART Tree Visualization (top 3 levels)</h3>
        <img src="data:image/png;base64,{tree_b64}" style="width:100%; border-radius:8px;">
    </div>

    <div class="card">
        <h3>2.4 Random Forest &mdash; GridSearchCV</h3>
        <p><strong>Best hyperparameters:</strong> <code>n_estimators={rf_bp["n_estimators"]}</code>,
        <code>max_depth={rf_bp["max_depth"]}</code></p>
        <p><strong>Param grid:</strong> n_estimators=[50, 100, 200], max_depth=[3, 5, 8] &rarr; 9 combos &times; 5 folds = 45 fits</p>
        {rf_heatmap_div}
        <div class="grid-4" style="grid-template-columns:repeat(3,1fr); margin-top:16px;">
            {metric_card("Test RMSE", f'{rf_res["test_rmse"]:.4f}')}
            {metric_card("Test MAE", f'{rf_res["test_mae"]:.4f}')}
            {metric_card("Test R²", f'{rf_res["test_r2"]:.4f}')}
        </div>
        {rf_scatter}
    </div>

    <div class="card">
        <h3>2.5 LightGBM &mdash; GridSearchCV (3 hyperparameters)</h3>
        <p><strong>Best hyperparameters:</strong> <code>n_estimators={lgb_bp["n_estimators"]}</code>,
        <code>max_depth={lgb_bp["max_depth"]}</code>, <code>learning_rate={lgb_bp["learning_rate"]}</code></p>
        <p><strong>Param grid:</strong> n_estimators=[50,100,200], max_depth=[3,4,5,6], learning_rate=[0.01,0.05,0.1] &rarr; 36 combos &times; 5 folds = 180 fits</p>
        {lgb_heatmap_divs}
        <div class="grid-4" style="grid-template-columns:repeat(3,1fr); margin-top:16px;">
            {metric_card("Test RMSE", f'{lgb_res["test_rmse"]:.4f}')}
            {metric_card("Test MAE", f'{lgb_res["test_mae"]:.4f}')}
            {metric_card("Test R²", f'{lgb_res["test_r2"]:.4f}')}
        </div>
        {lgb_scatter}
    </div>
</div>

<!-- ═══ MLP NEURAL NETWORK ═══ -->
<div id="tab-mlp" class="tab-content">
    <h2>MLP Neural Network (Section 2.6)</h2>

    <div class="card">
        <h3>Architecture</h3>
        <table>
            <thead><tr><th>Layer</th><th>Units</th><th>Activation</th></tr></thead>
            <tbody>
                <tr><td>Input</td><td>21 features (standardized)</td><td>&mdash;</td></tr>
                <tr><td>Hidden 1</td><td>128</td><td>ReLU</td></tr>
                <tr><td>Hidden 2</td><td>128</td><td>ReLU</td></tr>
                <tr><td>Output</td><td>1</td><td>Linear</td></tr>
            </tbody>
        </table>
        <p style="margin-top:12px;"><strong>Loss:</strong> Mean Squared Error &nbsp;|&nbsp; <strong>Optimizer:</strong> Adam (lr=0.001) &nbsp;|&nbsp;
        <strong>Early Stopping:</strong> patience=10 &nbsp;|&nbsp; <strong>Validation Split:</strong> 15% &nbsp;|&nbsp;
        <strong>Batch Size:</strong> 256 &nbsp;|&nbsp; <strong>Max Epochs:</strong> 100</p>
    </div>

    <div class="card">
        <h3>Training History</h3>
        <p><strong>Epochs trained:</strong> {len(mlp_history["loss"])} (early stopping activated)</p>
        <div class="grid-2">
            <div>{mlp_loss_div}</div>
            <div>{mlp_mae_div}</div>
        </div>
    </div>

    <div class="card">
        <h3>Test-Set Performance</h3>
        <div class="grid-4" style="grid-template-columns:repeat(3,1fr);">
            {metric_card("Test RMSE", f'{mlp_res["test_rmse"]:.4f}')}
            {metric_card("Test MAE", f'{mlp_res["test_mae"]:.4f}')}
            {metric_card("Test R²", f'{mlp_res["test_r2"]:.4f}')}
        </div>
        {mlp_scatter}
    </div>
</div>

<!-- ═══ TEST-SET COMPARISON ═══ -->
<div id="tab-testcomp" class="tab-content">
    <h2>Test-Set Model Comparison (Section 2.7)</h2>

    <div class="card">
        <p>All models trained on the <strong>70% training set</strong> (56,000 samples) and evaluated on
        the <strong>30% held-out test set</strong> (24,000 samples). Tree models tuned via GridSearchCV;
        MLP trained with early stopping on a validation split.</p>

        <h3>Model Performance Summary (Test Set)</h3>
        {summary_table_html}
    </div>

    <div class="card">
        <h3>Performance Comparison</h3>
        <div class="grid-2">
            <div>{rmse_bar_div}</div>
            <div>{r2_bar_div}</div>
        </div>
    </div>

    <div class="card">
        <h3>Analysis</h3>
        <p><strong>{best_model["Model"]}</strong> achieves the lowest test-set RMSE ({best_model["Test RMSE"]:.4f}) and highest
        R&sup2; ({best_model["Test R²"]:.4f}), making it the best-performing model. This confirms gradient-boosted trees&rsquo;
        ability to capture non-linear interactions among features.</p>
        <p>The baseline <strong>Linear Regression</strong> models (including Lasso and Ridge) provide a solid floor
        but struggle with non-linear relationships. <strong>CART</strong> (a single decision tree) underperforms
        due to high variance, even after tuning. <strong>Random Forest</strong> improves over CART through ensemble
        averaging. The <strong>MLP neural network</strong> provides competitive performance, demonstrating that deep
        learning can capture patterns in tabular data, though it requires longer training time.</p>
        <p><strong>Trade-offs:</strong> Linear models are the most interpretable and fastest to train but least accurate.
        CART offers visual interpretability (tree diagrams) at moderate accuracy. Random Forest and LightGBM offer the
        best accuracy but are less interpretable (addressed via SHAP analysis). The MLP is a black-box model that offers
        flexibility in architecture design.</p>
    </div>
</div>

'''

html = html.replace(insert_before, new_tabs_html + insert_before, 1)

# Write the updated HTML
print("Writing updated HTML...")
with open('Jobs_Skills_ML_Pipeline.html', 'w') as f:
    f.write(html)

print(f"Done! HTML file updated ({len(html):,} bytes)")
print("New tabs added: Hyperparameter Tuning, MLP Neural Network, Test-Set Comparison")
