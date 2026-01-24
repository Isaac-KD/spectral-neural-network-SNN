import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Callable, Optional, Dict, List
from tqdm.auto import tqdm
import torch.optim as optim

# split et scale
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: Callable,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    scheduler: Optional[object] = None, 
    device: str = ("cuda" if torch.cuda.is_available()
                    else "mps" if torch.backends.mps.is_available()
                    else "cpu"),
    ortho_lambda: float = 0.01,         
    ortho_loss_fn: Callable = torch.nn.functional.mse_loss,
    clip_grad_norm: Optional[float] = None,
    log_interval: int = 100  # On imprime une ligne d'historique tous les 50 epochs
) -> Dict[str, List[float]]:
    
    # Setup
    model = model.to(device)
    history = {'train_loss': [], 'val_loss': [], 'lr': []}
    
    print(f"🚀 Start Training | Device: {device.upper()} | Total Epochs: {epochs}")

    # ====================================================
    # BARRE GLOBALE (Gère les Époques)
    # ====================================================
    # unit="epoch" permet d'afficher "12.3s/epoch"
    with tqdm(total=epochs, desc="Progression", unit="epoch") as pbar:
        
        for epoch in range(epochs):
            
            # --- 1. TRAINING ---
            model.train()
            running_train_loss = 0.0
            
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                
                optimizer.zero_grad()
                
                # Forward & Loss
                y_pred = model(Xb)
                main_loss = criterion(y_pred, yb)
                
                # Ortho Loss (Duck Typing)
                if hasattr(model, 'get_total_ortho_loss') and ortho_lambda > 0:
                    ortho_loss = model.get_total_ortho_loss(ortho_loss_fn)
                    loss = main_loss + (ortho_lambda * ortho_loss)
                else:
                    loss = main_loss

                # Backward
                loss.backward()
                
                if clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                
                optimizer.step()
                
                running_train_loss += loss.item() * Xb.size(0)
                
                # Cela permet de voir la loss bouger en temps réel.
                pbar.set_postfix({'batch_loss': f"{loss.item():.4f}", 'phase': 'train'})

            avg_train_loss = running_train_loss / len(train_loader.dataset)
            history['train_loss'].append(avg_train_loss)

            # --- 2. VALIDATION ---
            model.eval()
            running_val_loss = 0.0
            
            with torch.no_grad():
                for Xb, yb in val_loader:
                    Xb, yb = Xb.to(device), yb.to(device)
                    y_pred = model(Xb)
                    loss = criterion(y_pred, yb)
                    running_val_loss += loss.item() * Xb.size(0)
                    
                    # On indique qu'on est en validation
                    pbar.set_postfix({'phase': 'val'})

            avg_val_loss = running_val_loss / len(val_loader.dataset)
            history['val_loss'].append(avg_val_loss)

            # --- 3. SCHEDULER ---
            current_lr = optimizer.param_groups[0]['lr']
            history['lr'].append(current_lr)

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(avg_val_loss)
                else:
                    scheduler.step()

            # --- 4. MISE À JOUR FIN D'ÉPOQUE ---
            # On avance la barre de 1
            pbar.update(1)
            
            # On fixe les moyennes à droite pour l'utilisateur
            pbar.set_postfix({
                'train': f"{avg_train_loss:.4f}",
                'val': f"{avg_val_loss:.4f}",
                'lr': f"{current_lr:.1e}"
            })

            # --- 5. LOGGING PERMANENT (Optionnel) ---
            # Si on veut garder une trace écrite tous les X epochs
            if (epoch + 1) % log_interval == 0:
                tqdm.write(f"Epoch {epoch+1}/{epochs} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")

    print("🏁 Entraînement terminé.")
    return history


def split_dataset(X, y, seed=42):
    # 70% train, 30% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.30,
        random_state=seed,
        shuffle=True
    )

    # 15% val, 15% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        random_state=seed,
        shuffle=True
    )

    return X_train, y_train, X_val, y_val, X_test, y_test

def standardize_data(X_train, y_train, X_val, y_val, X_test, y_test):
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train = scaler_X.fit_transform(X_train)
    X_val   = scaler_X.transform(X_val)
    X_test  = scaler_X.transform(X_test)

    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val   = scaler_y.transform(y_val.reshape(-1, 1)).ravel()
    y_test  = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler_X, scaler_y


def fit(model, X_train, y_train, epochs=100, lr=0.01, verbose=True,loss_function= nn.MSELoss()):
    """
    Entraîne un DeepSpectralNet sur des données simples.

    Args:
        model (nn.Module): Le modèle DSN à entraîner.
        X_train (torch.Tensor): Entrées, shape (batch_size, in_features)
        y_train (torch.Tensor): Cibles, shape (batch_size, out_features)
        epochs (int): Nombre d'époques d'entraînement
        lr (float): Learning rate
        verbose (bool): Affiche la loss à chaque epoch
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    X_train, y_train = X_train.to(device), y_train.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = loss_function

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)


        loss.backward()
        optimizer.step()

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"Epoch {epoch}/{epochs} | Loss: {loss.item():.6f}")

    return model