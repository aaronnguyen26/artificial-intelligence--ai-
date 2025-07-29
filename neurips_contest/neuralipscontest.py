# GNN-based polymer property prediction setup
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader # Changed from torch_geometric.data
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from torch_geometric.data import Data as GeometricData, Batch as GeometricBatch
from torch_geometric.nn import GCNConv, global_mean_pool
import os
from sklearn.model_selection import train_test_split
from torch_geometric.nn import GATConv # Import GATConv


print('Torch version:', torch.__version__)
try:
    import torch_geometric
    print('torch_geometric version:', torch_geometric.__version__)
except ImportError:
    print('torch_geometric not found')

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # Removed Gasteiger charge computation and related error handling
    # AllChem.ComputeGasteigerCharges(mol)

    # Basic atom features
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append([
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            int(atom.GetIsAromatic()),
            # Removed atom.GetTotalNumHs(),
            # Removed atom.GetTotalDegree(),
            # Removed gasteiger_charge,
            # Removed hybridization_one_hot,
            # Removed chiral_one_hot
        ])

    x = torch.tensor(atom_features, dtype=torch.float)

    # Basic edge features
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        
        bond_features = [
            int(bond.GetIsConjugated()),
            int(bond.IsInRing()),
            # Removed bond_type_one_hot,
            # Removed bond_stereo_one_hot
        ]

        edge_index.append([i, j])
        edge_index.append([j, i])  # undirected
        edge_attr.append(bond_features)
        edge_attr.append(bond_features) # For undirected edge

    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    else:
        # Ensure edge_attr has correct dimension even if no edges
        edge_index = torch.empty((2, 0), dtype=torch.long)
        # Calculate expected bond_features length for empty tensor
        # Dummy bond to get feature length for basic features
        expected_bond_feature_len = 2 # Only IsConjugated and IsInRing
        edge_attr = torch.empty((0, expected_bond_feature_len), dtype=torch.float)
    
    data = GeometricData(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data

# --- Data Loading and Merging for Multi-Task Learning ---
# Initial data loading
train_df = pd.read_csv('extracted_dataset/train.csv')

# Create a master DataFrame for all data
master_df = train_df.copy()

# Load and merge supplementary datasets
supplement_paths = [
    'extracted_dataset/train_supplement/dataset1.csv', # TC_mean (Tc)
    'extracted_dataset/train_supplement/dataset2.csv', # Only SMILES, no property
    'extracted_dataset/train_supplement/dataset3.csv', # Tg
    'extracted_dataset/train_supplement/dataset4.csv', # FFV
]

# Map supplementary column names to main property names
supplement_col_map = {
    'TC_mean': 'Tc',
    'Tg': 'Tg',
    'FFV': 'FFV'
}

for i, path in enumerate(supplement_paths):
    try:
        df_supp = pd.read_csv(path)
        # Rename columns if necessary
        df_supp = df_supp.rename(columns={k: v for k, v in supplement_col_map.items() if k in df_supp.columns})
        
        # Merge based on SMILES. Use outer merge to keep all original data.
        master_df = pd.merge(master_df, df_supp, on='SMILES', how='outer', suffixes=('', '_supp'))
        
        # Resolve conflicts: if a property exists in both original and supplement,
        # fill NaNs in original column with values from supplement column.
        for col in supplement_col_map.values():
            if f"{col}_supp" in master_df.columns:
                master_df[col] = master_df[col].fillna(master_df[f"{col}_supp"])
                master_df = master_df.drop(columns=[f"{col}_supp"])

    except Exception as e:
        print(f"Could not load or merge {path}: {e}")

# Define properties
properties = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

# Now, create the unified_dataset from the master_df
unified_dataset = []
for idx, row in master_df.iterrows():
    smiles = row['SMILES']
    graph = smiles_to_graph(smiles)
    
    if graph is not None:
        target_dict = {}
        for prop in properties:
            val = row.get(prop) # Use .get() to handle cases where prop might not be in master_df
            if pd.notnull(val):
                target_dict[prop] = float(val)
            else:
                target_dict[prop] = np.nan # Use NaN for missing targets
        unified_dataset.append((graph, target_dict))
    # If graph is None, we skip this entry. This is consistent with current code.

print(f"Created unified dataset with {len(unified_dataset)} samples for multi-task training.")


# Dataset preparation: PyTorch Geometric Dataset for multi-task
def augment_graph(data):
    # Add random noise to continuous features
    if torch.rand(1) < 0.5:
        noise = torch.randn_like(data.x) * 0.01
        data.x = data.x + noise
    return data

class PolymerDataset(Dataset):
    def __init__(self, unified_data, properties, training=False):
        self.unified_data = unified_data
        self.properties = properties
        self.training = training
        self.prop_to_idx = {prop: i for i, prop in enumerate(properties)}
    
    def __len__(self):
        return len(self.unified_data)
    
    def __getitem__(self, idx):
        graph, target_dict = self.unified_data[idx]
        data = graph.clone() # Ensure we don't modify original graph objects

        # Create a tensor for all targets, with NaNs for missing ones
        targets_tensor = torch.full((len(self.properties),), float('nan'), dtype=torch.float)
        for prop, val in target_dict.items():
            if not np.isnan(val):
                targets_tensor[self.prop_to_idx[prop]] = val

        data.y = targets_tensor # Now data.y is a tensor of all targets (batch_size, num_properties)
        
        # Apply augmentation during training
        if self.training:
            data = augment_graph(data)
        return data

# Model setup: simple GCN for regression
class GATRegression(nn.Module): # Renamed from GCNRegression
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.2, heads=4): # Removed default from in_channels
        super().__init__()
        self.heads = heads
        # Use GATConv layers
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.bn1 = nn.BatchNorm1d(hidden_channels * heads) # Reverted eps to default
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout)
        self.bn2 = nn.BatchNorm1d(hidden_channels * heads) # Reverted eps to default
        self.conv3 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout)
        self.bn3 = nn.BatchNorm1d(hidden_channels * heads) # Reverted eps to default
        self.lin = nn.Linear(hidden_channels * heads, out_channels) # Adjust for multi-head output
        self.dropout = nn.Dropout(dropout)
        # Add residual connections
        self.residual1 = nn.Linear(in_channels, hidden_channels * heads) # Adjust for multi-head output
        # Removed self.bn_residual1
        # self.residual2 is not used in current forward, but kept for consistency if needed.
        self.residual2 = nn.Linear(hidden_channels * heads, hidden_channels * heads)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Check for NaNs in input x (diagnostic, kept)
        if torch.isnan(x).any():
            print("NaN detected in input x to GATRegression!")
            return torch.full((data.batch.max().item() + 1, self.lin.out_features), float('nan'), device=x.device)

        identity = self.residual1(x)
        # Removed identity = self.bn_residual1(identity)

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = torch.relu(x + identity)
        x = self.dropout(x)
        
        # Check for NaNs after first block (diagnostic, kept)
        if torch.isnan(x).any():
            print("NaN detected after first GAT block!")
            return torch.full((data.batch.max().item() + 1, self.lin.out_features), float('nan'), device=x.device)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout(x)

        # Check for NaNs after second block (diagnostic, kept)
        if torch.isnan(x).any():
            print("NaN detected after second GAT block!")
            return torch.full((data.batch.max().item() + 1, self.lin.out_features), float('nan'), device=x.device)

        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = torch.relu(x)
        
        # Check for NaNs after third block (diagnostic, kept)
        if torch.isnan(x).any():
            print("NaN detected after third GAT block!")
            return torch.full((data.batch.max().item() + 1, self.lin.out_features), float('nan'), device=x.device)

        x = global_mean_pool(x, data.batch)
        
        # Check for NaNs after pooling (diagnostic, kept)
        if torch.isnan(x).any():
            print("NaN detected after global_mean_pool!")
            return torch.full((data.batch.max().item() + 1, self.lin.out_features), float('nan'), device=x.device)

        out = self.lin(x)
        
        # Check for NaNs in final output (diagnostic, kept)
        if torch.isnan(out).any():
            print("NaN detected in final output!")
            return torch.full((data.batch.max().item() + 1, self.lin.out_features), float('nan'), device=x.device)

        return out

# Multi-Task GNN Model
class MultiTaskGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_tasks): # Removed default from in_channels
        super().__init__()
        # Use GATRegression as the shared backbone
        self.gnn = GATRegression(in_channels, hidden_channels, hidden_channels) # Changed to GATRegression
        self.task_heads = nn.ModuleList([
            nn.Linear(hidden_channels, 1) for _ in range(num_tasks) # Corrected: should be hidden_channels, not hidden_channels * self.gnn.heads
        ])
    
    def forward(self, data):
        shared_repr = self.gnn(data)
        # Each head takes the shared representation and outputs a prediction
        return [head(shared_repr) for head in self.task_heads]

# --- Weighted MAE (wMAE) metric implementation ---
def compute_wmae(y_true_dict, y_pred_dict, property_names):
    # y_true_dict, y_pred_dict: dict of property -> np.array (with np.nan for missing)
    K = len(property_names)
    n_i = []
    r_i = []
    for prop in property_names:
        y_true = y_true_dict[prop]
        mask = ~np.isnan(y_true)
        n = np.sum(mask)
        n_i.append(n)
        if n > 0:
            r = np.nanmax(y_true) - np.nanmin(y_true)
        else:
            r = 1.0  # avoid div by zero
        r_i.append(r)
    n_i = np.array(n_i)
    r_i = np.array(r_i)
    
    # Compute weights w_i based on the competition formula
    # Handle cases where n_i might be zero to avoid division by zero in sqrt_inv_n
    sqrt_inv_n = np.zeros_like(n_i, dtype=float)
    valid_n_mask = n_i > 0
    sqrt_inv_n[valid_n_mask] = np.sqrt(1.0 / n_i[valid_n_mask])

    sum_sqrt_inv_n = np.sum(sqrt_inv_n)
    if sum_sqrt_inv_n == 0: # Fallback if all n_i are zero (e.g., empty dataset or all NaNs)
        weight_norm = np.ones_like(n_i, dtype=float) # Assign a default if no valid data for any property
    else:
        weight_norm = (K * sqrt_inv_n) / sum_sqrt_inv_n

    w_i = (1.0 / r_i) * weight_norm
    
    # Compute weighted MAE
    total = 0.0
    count = 0
    
    # Re-calculating total and count based on actual available data points
    for j, prop in enumerate(property_names):
        y_true_prop = y_true_dict[prop]
        y_pred_prop = y_pred_dict[prop]
        
        # Ensure y_true_prop and y_pred_prop are numpy arrays for consistent masking
        y_true_prop = np.array(y_true_prop)
        y_pred_prop = np.array(y_pred_prop)

        mask = ~np.isnan(y_true_prop) # Only consider non-NaN true values
        
        if np.any(mask): # Only if there are valid data points for this property
            abs_errors = np.abs(y_pred_prop[mask] - y_true_prop[mask])
            total += np.sum(w_i[j] * abs_errors)
            count += np.sum(mask) # Add the number of valid data points for this property
            
    wmae = total / count if count > 0 else np.nan
    return wmae, w_i

# --- Multi-Task Training Loop ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Split the unified_dataset into training and validation sets
train_unified_data, val_unified_data = train_test_split(
    unified_dataset, test_size=0.2, random_state=42)

train_dataset = PolymerDataset(train_unified_data, properties, training=True)
val_dataset = PolymerDataset(val_unified_data, properties, training=False)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# --- Calculate w_i for training data ---
# This is crucial for aligning training loss with the competition metric
train_true_dict_for_weights = {prop: [] for prop in properties}
for graph, target_dict in train_unified_data:
    for i, prop in enumerate(properties):
        train_true_dict_for_weights[prop].append(target_dict[prop])

K = len(properties)
n_i_train = []
r_i_train = []

for prop in properties:
    y_true_prop = np.array(train_true_dict_for_weights[prop])
    mask = ~np.isnan(y_true_prop)
    n = np.sum(mask)
    n_i_train.append(n)
    if n > 0:
        r = np.nanmax(y_true_prop) - np.nanmin(y_true_prop)
    else:
        r = 1.0  # Avoid division by zero if no data for a property
    r_i_train.append(r)

n_i_train = np.array(n_i_train)
r_i_train = np.array(r_i_train)

# Compute weights w_i based on the competition formula
# Handle cases where n_i_train might be zero to avoid division by zero in sqrt_inv_n
sqrt_inv_n_train = np.zeros_like(n_i_train, dtype=float)
valid_n_mask = n_i_train > 0
sqrt_inv_n_train[valid_n_mask] = np.sqrt(1.0 / n_i_train[valid_n_mask])

sum_sqrt_inv_n_train = np.sum(sqrt_inv_n_train)
if sum_sqrt_inv_n_train == 0: # Fallback if all n_i are zero (e.g., empty dataset)
    weight_norm_train = np.ones_like(n_i_train, dtype=float)
else:
    weight_norm_train = (K * sqrt_inv_n_train) / sum_sqrt_inv_n_train

w_i_train = (1.0 / r_i_train) * weight_norm_train
w_i_train_tensor = torch.tensor(w_i_train, dtype=torch.float, device=device) # Move to device

print("\nCalculated training weights (w_i) for each property:")
for i, prop in enumerate(properties):
    print(f"  {prop}: {w_i_train[i]:.4f}")

num_properties = len(properties)
# Updated in_channels to 4 for basic atom features
model = MultiTaskGNN(in_channels=4, hidden_channels=64, num_tasks=num_properties)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
loss_fn = nn.L1Loss(reduction='none') # Use reduction='none' to handle NaNs manually

num_epochs = 30
best_val_loss = float('inf')
patience = 5
patience_counter = 0

print("\n=== Starting Multi-Task Training ===")

for epoch in range(1, num_epochs + 1):
    model.train()
    train_losses = []
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Multi-task forward pass
        out_per_task = model(batch) # This returns a list of tensors, one for each task
        
        batch_weighted_loss = 0 # Use this to accumulate weighted losses for the batch
        
        # Reshape batch.y to (batch_size, num_properties)
        # batch.num_graphs gives the actual number of graphs in the batch
        reshaped_targets = batch.y.view(batch.num_graphs, num_properties)

        # Calculate loss for each task, handling NaNs and applying w_i
        for i, prop_output in enumerate(out_per_task):
            true_val = reshaped_targets[:, i].squeeze() # Get targets for current property
            predicted_val = prop_output.squeeze()
            
            # Create a mask for non-NaN values in the true targets
            mask = ~torch.isnan(true_val)
            
            if mask.any(): # Only calculate loss if there are valid targets for this property in the batch
                loss = loss_fn(predicted_val[mask], true_val[mask])
                # Apply the pre-calculated w_i weight to the mean loss for this property
                batch_weighted_loss += w_i_train_tensor[i] * loss.mean()
        
        # The total loss for the batch is the sum of weighted losses, averaged by the number of samples in the batch
        # This directly mirrors the wMAE formula's outer (1/|X|) term
        if batch.num_graphs > 0:
            total_loss = batch_weighted_loss / batch.num_graphs
            total_loss.backward()
            # Removed gradient clipping
            optimizer.step()
            train_losses.append(total_loss.item())
        else:
            train_losses.append(0.0) # Should not happen with non-empty batches
            
    avg_train_loss = np.mean(train_losses)

    # Validation
    model.eval()
    val_losses = []
    val_true_dict_mt = {prop: [] for prop in properties}
    val_pred_dict_mt = {prop: [] for prop in properties}

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            out_per_task = model(batch)
            
            batch_weighted_loss = 0
            
            # Reshape batch.y to (batch_size, num_properties)
            reshaped_targets = batch.y.view(batch.num_graphs, num_properties)

            for i, prop_output in enumerate(out_per_task):
                true_val = reshaped_targets[:, i].squeeze()
                predicted_val = prop_output.squeeze()
                
                mask = ~torch.isnan(true_val)
                
                if mask.any():
                    loss = loss_fn(predicted_val[mask], true_val[mask])
                    # Apply the pre-calculated w_i weight to the mean loss for this property
                    batch_weighted_loss += w_i_train_tensor[i] * loss.mean() # Use training weights for validation loss too
                
                # Store validation results for wMAE calculation
                true_vals_to_add = true_val.cpu().numpy().tolist()
                pred_vals_to_add = predicted_val.cpu().numpy().tolist()

                val_true_dict_mt[properties[i]].extend(true_vals_to_add)
                val_pred_dict_mt[properties[i]].extend(pred_vals_to_add)

            if batch.num_graphs > 0:
                val_losses.append((batch_weighted_loss / batch.num_graphs).item())
            else:
                val_losses.append(0.0)

    avg_val_loss = np.mean(val_losses)
    print(f"Epoch {epoch}: Train Weighted MAE={avg_train_loss:.4f}, Val Weighted MAE={avg_val_loss:.4f}")
    scheduler.step(avg_val_loss)

    # Early stopping based on overall validation loss
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        # Optionally save the best model state here
        # torch.save(model.state_dict(), 'best_multitask_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

# --- Compute overall wMAE for multi-task validation predictions ---
wmae_mt, w_i_mt = compute_wmae(val_true_dict_mt, val_pred_dict_mt, properties)
print(f"\nOverall validation wMAE (Multi-Task Model): {wmae_mt:.4f}")
for i, prop in enumerate(properties):
    print(f"{prop}: weight={w_i_mt[i]:.4f}")

# --- Load test data and make predictions using the trained MultiTaskGNN ---
def predict_test_set_multitask(model, test_df, device, properties):
    predictions = {prop: [] for prop in properties}
    
    # Convert test SMILES to graphs
    test_graphs = []
    for smiles in test_df['SMILES']:
        graph = smiles_to_graph(smiles)
        test_graphs.append(graph)
    
    # Create a dummy target for the PolymerDataset, as targets are not needed for prediction
    # The PolymerDataset expects a list of (graph, target_dict)
    # For test data, we can create dummy target_dicts with NaNs
    test_unified_data = []
    for graph in test_graphs:
        if graph is not None:
            test_unified_data.append((graph, {prop: np.nan for prop in properties}))
        else:
            # Handle cases where smiles_to_graph returns None for test data
            # This means we won't have a prediction for this SMILES
            # You might want to fill with NaN or some default later
            pass 

    test_dataset = PolymerDataset(test_unified_data, properties, training=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out_per_task = model(batch) # List of predictions for each task
            
            for i, prop_output in enumerate(out_per_task):
                prop_predictions = prop_output.squeeze().cpu().numpy().tolist()
                predictions[properties[i]].extend(prop_predictions)
    
    return predictions

# Make predictions using the trained multi-task model
test_df = pd.read_csv('extracted_dataset/test.csv')
predictions_mt = predict_test_set_multitask(model, test_df, device, properties)

# Create submission DataFrame
submission_df_mt = pd.DataFrame()
submission_df_mt['id'] = test_df['id']

# Ensure predictions are aligned with test_df 'id' count,
# especially if some SMILES failed to convert to graph.
# For now, assuming all test SMILES convert successfully.
# If not, you'd need to map predictions back to original test_df indices.
for prop in properties:
    # Pad with NaN if the number of predictions doesn't match the test_df length
    # This can happen if some SMILES in test_df failed smiles_to_graph
    if len(predictions_mt[prop]) < len(test_df):
        print(f"Warning: Number of predictions for {prop} ({len(predictions_mt[prop])}) does not match test_df length ({len(test_df)}). Padding with NaN.")
        predictions_mt[prop].extend([np.nan] * (len(test_df) - len(predictions_mt[prop])))
    submission_df_mt[prop] = predictions_mt[prop][:len(test_df)] # Truncate if too many (shouldn't happen)

# Save predictions to CSV file
pd.DataFrame(predictions_mt).to_csv('extracted_dataset/submission.csv', index=False)

# Optional: If you also have a single-task model prediction block that you want to output as 'submission.csv',
# is now redundant and can be removed or commented out if you fully switch
# to the multi-task approach.
# For now, I'll comment it out to show the change.

# # Save best models during training
# model_dict = {}
# for prop in properties:
#     model = GCNRegression(in_channels=6, hidden_channels=64, out_channels=1, dropout=0.2)
#     model = model.to(device)
#     # Train model as before
#     # ... existing code ...
#     model_dict[prop] = model

# # Load test set
# test_df = pd.read_csv('extracted_dataset/test.csv')

# # Make predictions
# predictions = predict_test_set(model_dict, test_df, device)

# # Create submission DataFrame
# submission_df = pd.DataFrame()
# submission_df['id'] = test_df['id']
# for prop in properties:
#     submission_df[prop] = predictions[prop]

# # Save predictions to CSV file
# submission_df.to_csv('submission.csv', index=False)
# print("\nSubmission file created with format:")
# print(submission_df.head())

    


 



    






    
      