import torch
import torch.nn as nn
import torch.nn.init as init
import math
from typing import List, Callable, Optional

class SpectralQuadraticLayer(nn.Module):
    def __init__(
        self, 
        in_features: int,          
        out_features: int, 
        ortho_mode: str = 'hard', 
        bias: bool = True,
        use_scaling: bool = True # NOUVEAU: Active le scaling 1/sqrt(d)
    ):
        """
        Couche Spectrale Quadratique Stabilisée.
        Formule : y = Lambda * ((Wx / sqrt(d))^2) + bias
        """
        super().__init__()

        valid_modes = {'hard', 'cayley', 'soft', None}
        assert ortho_mode in valid_modes, f"Mode invalide: {ortho_mode}"

        self.in_features = in_features
        self.out_features = out_features
        self.ortho_mode = ortho_mode
        self.use_scaling = use_scaling
        
        # Facteur de scaling pré-calculé (1 / sqrt(d))
        # Cela stabilise la variance avant l'élévation au carré.
        self.scaling_factor = 1.0 / math.sqrt(in_features) if use_scaling else 1.0

        # 1. La Base Partagée (Matrice de Rotation U)
        self.base_change = nn.Linear(in_features, in_features, bias=bias)
        
        # Gestion des contraintes d'orthogonalité
        if self.ortho_mode == 'hard':
            torch.nn.utils.parametrizations.orthogonal(self.base_change, "weight")
        elif self.ortho_mode == 'cayley':
            torch.nn.utils.parametrizations.orthogonal(self.base_change, "weight", orthogonal_map='cayley')
        elif self.ortho_mode == 'soft':
            init.orthogonal_(self.base_change.weight)
            self.register_buffer('eye_target', torch.eye(in_features), persistent=False)
        
        # 2. Les Valeurs Propres (Lambda - Mélange)
        self.eigen_weights = nn.Linear(in_features, out_features, bias=True)
        
        self._init_parameters()

    def _init_parameters(self) -> None:
        # A. Base Change : On garde l'orthogonale (c'est la structure)
        if self.ortho_mode is None:
            init.orthogonal_(self.base_change.weight)
        
        # B. Eigen Weights : ON AUGMENTE LE VOLUME ICI
        # Au lieu de normal_(std=0.02), on utilise Kaiming Uniform
        # Cela donne des poids initiaux plus variés, forçant le réseau à utiliser le quadratique.
        init.kaiming_uniform_(self.eigen_weights.weight, a=math.sqrt(5))
        
        if self.eigen_weights.bias is not None:
            init.zeros_(self.eigen_weights.bias)
        if self.base_change.bias is not None:
            init.zeros_(self.base_change.bias)

    def get_ortho_loss(self, loss_fn: Callable = torch.nn.MSELoss()) -> torch.Tensor:
        if self.ortho_mode != 'soft':
            return torch.tensor(0.0, device=self.base_change.weight.device)
        w = self.base_change.weight
        gram = torch.mm(w, w.t())
        return loss_fn(gram, self.eye_target)
    
    def forward(self, x):
        # 1. Projection (Rotation)
        h = self.base_change(x)
        
        # 2. Scaling (CRITIQUE pour la stabilité)
        # Empêche l'explosion des valeurs avant le carré
        if self.use_scaling:
            h = h * self.scaling_factor
        
        # 3. Activation Quadratique (Énergie)
        energy = h * h 
        # Note: on peut tester h.pow(2) mais h*h est souvent un poil plus rapide
        
        # 4. Recombinaison (Valeurs propres)
        y = self.eigen_weights(energy)
        
        return y

class SpectralResidualBlock(nn.Module):
    def __init__(self, layer: SpectralQuadraticLayer, dropout: float = 0.0):
        super().__init__()
        self.layer = layer
        self.norm = nn.LayerNorm(layer.in_features)
        self.dropout = nn.Dropout(dropout)
        
        # Le Gate Scalaire (ReZero trick)
        # On l'initialise à une petite valeur non-nulle (ex: 0.1) pour encourager le signal
        self.res_weight = nn.Parameter(torch.tensor(1)) 
        
        self.needs_projection = (layer.in_features != layer.out_features)
        if self.needs_projection:
            self.projection = nn.Linear(layer.in_features, layer.out_features, bias=False)
        
    def forward(self, x):
        residual = x
        
        # Branche Spectrale
        z = self.norm(x)
        z = self.layer(z)
        z = self.dropout(z)
        
        # Projection du résiduel si nécessaire
        if self.needs_projection:
            residual = self.projection(residual)
            
        # Connexion pondérée : x + alpha * f(x)
        return residual + (self.res_weight * z)

class DeepSpectralNet(nn.Module):
    def __init__(
        self, 
        layers_dim: List[int],          
        ortho_mode: str = 'hard', # 'hard' est recommandé pour la stabilité
        use_final_linear: bool = False, 
        use_residuals: bool = True, # Active les connexions résiduelles
        dropout: float = 0.1
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        
        num_layers = len(layers_dim) - 1
        
        for i in range(num_layers):
            in_d = layers_dim[i]
            out_d = layers_dim[i+1]
            is_last = (i == num_layers - 1)
            
            # Si c'est la dernière couche et qu'on veut du linéaire pur (classification)
            if is_last and use_final_linear:
                self.layers.append(nn.Linear(in_d, out_d))
            else:
                # Création de la couche spectrale
                spec_layer = SpectralQuadraticLayer(
                    in_d, out_d, 
                    ortho_mode=ortho_mode,
                    use_scaling=True # On active le scaling par défaut
                )
                
                # Si on veut des résiduels (et ce n'est pas une couche de sortie linéaire simple)
                if use_residuals:
                    block = SpectralResidualBlock(spec_layer, dropout=dropout)
                    self.layers.append(block)
                else:
                    self.layers.append(spec_layer)
                    # Si pas de résiduel, on met quand même une activation/norm si besoin
                    if not is_last:
                        self.layers.append(nn.LayerNorm(out_d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def get_total_ortho_loss(self, loss_fn: Callable = nn.MSELoss()) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        
        for module in self.layers:
            # Si c'est un Block Résiduel, on regarde dedans
            if isinstance(module, SpectralResidualBlock):
                total_loss += module.layer.get_ortho_loss(loss_fn)
            # Si c'est une couche directe
            elif isinstance(module, SpectralQuadraticLayer):
                total_loss += module.get_ortho_loss(loss_fn)
                
        return total_loss