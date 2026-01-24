import torch
import torch.nn as nn
import torch.nn.init as init
from typing import List, Callable

class SpectralBilinearLayer(nn.Module):
    """
    Version généralisée (bilinéaire) de la couche spectrale.
    Au lieu de x^T A x (quadratique), elle implémente la somme des termes spectraux :
    y = Σ λ_k * (x^T p_k) * (l_k^T x)
    """
    def __init__(
        self, 
        in_features: int,          
        out_features: int, 
        ortho_mode: str = 'hard', 
        bias: bool = True
    ):
        super().__init__()

        valid_modes = {'hard', 'cayley', 'soft', None}
        assert ortho_mode in valid_modes, f"Mode invalide: {ortho_mode}"

        self.in_features = in_features
        self.ortho_mode = ortho_mode
        
        # 1. Deux bases distinctes : Vecteurs propres à DROITE (P) et à GAUCHE (L)
        # Contrairement à la version quadratique, on ne suppose plus p_k = l_k
        self.right_projections = nn.Linear(in_features, in_features, bias=bias)
        self.left_projections = nn.Linear(in_features, in_features, bias=bias)
        
        # 2. Contraintes d'Orthogonalité (optionnelles) sur les deux bases
        if self.ortho_mode in ['hard', 'cayley']:
            map_type = 'cayley' if self.ortho_mode == 'cayley' else None
            torch.nn.utils.parametrizations.orthogonal(self.right_projections, "weight", orthogonal_map=map_type)
            torch.nn.utils.parametrizations.orthogonal(self.left_projections, "weight", orthogonal_map=map_type)

        elif self.ortho_mode == 'soft':
            init.orthogonal_(self.right_projections.weight)
            init.orthogonal_(self.left_projections.weight)
            self.register_buffer('eye_target', torch.eye(in_features), persistent=False)
        
        # 3. Les Valeurs Propres (Lambdas) - Mélange les interactions scalaires
        self.eigen_weights = nn.Linear(in_features, out_features, bias=True)
        
        self._init_parameters()

    def _init_parameters(self) -> None:
            # 1. Initialisation des bases de projection (P et L)
            # On utilise Xavier Uniform pour une distribution aléatoire forte et équilibrée
            nn.init.xavier_uniform_(self.right_projections.weight, gain=1.5)
            nn.init.xavier_uniform_(self.left_projections.weight, gain=1.5)
            
            # 2. Initialisation des Valeurs Propres (Lambdas)
            # Initialisation aléatoire uniforme pour le mélange spectral
            nn.init.xavier_uniform_(self.eigen_weights.weight, gain=1.5)
            
            # 3. Mise à zéro des biais uniquement
            # Comme demandé, on garde les biais neutres pour ne pas désharmoniser 
            # la géométrie des formes bilinéaires au départ [cite: 39, 123]
            nn.init.zeros_(self.eigen_weights.bias)
            
            if self.right_projections.bias is not None: 
                nn.init.zeros_(self.right_projections.bias)
            if self.left_projections.bias is not None: 
                nn.init.zeros_(self.left_projections.bias)

    def get_ortho_loss(self, loss_fn: Callable) -> torch.Tensor:
        if self.ortho_mode != 'soft':
            return torch.tensor(0.0, device=self.right_projections.weight.device)
            
        w_r = self.right_projections.weight
        w_l = self.left_projections.weight
        
        # On régularise les deux matrices pour qu'elles restent des bases orthogonales
        loss_r = loss_fn(torch.mm(w_r, w_r.t()), self.eye_target)
        loss_l = loss_fn(torch.mm(w_l, w_l.t()), self.eye_target)
        
        return loss_r + loss_l
    
    def forward(self, x):
        # x shape: (Batch, In_Features)
        
        # Étape A : Projections distinctes (Vecteurs propres gauche et droite)
        # h_right = x @ P.T  |  h_left = x @ L.T
        h_right = self.right_projections(x)
        h_left = self.left_projections(x)
        
        # Étape B : Interaction Bilinéaire (Produit point à point)
        # Au lieu de h^2, on fait h_right * h_left
        # Cela correspond à (x^T p_k) * (l_k^T x) dans le rapport 
        bilinear_interaction = h_right * h_left 
        
        # C. Combinaison par les valeurs propres
        y = self.eigen_weights(bilinear_interaction)
        
        return y

class SpectralBillinearNet(nn.Module):
        def __init__(
            self, 
            layers_dim: List[int],          
            ortho_mode: str = None, 
            use_final_linear: bool = False, 
            bias: bool = True, 
            use_layernorm: bool = False
        ):
            """
            Args:
                layers_dim (list): Liste des dimensions [in, hidden, ..., out].
                ortho_mode (str): 'hard', 'cayley', 'soft', ou None.
                use_final_linear (bool): Si True, la dernière couche est linéaire (standard pour classification).
                use_layernorm (bool): Si True, ajoute une normalisation entre les couches cachées. 
                                    Recommandé car les réseaux quadratiques peuvent faire exploser les valeurs.
            """
            super().__init__()
            self.layers = nn.ModuleList()
            self.use_layernorm = use_layernorm

            assert len(layers_dim) >= 2, "Il faut au moins une dimension d'entrée et de sortie"

            num_layers = len(layers_dim) - 1
            
            for i in range(num_layers):
                is_last_layer = (i == num_layers - 1)
                in_d = layers_dim[i]
                out_d = layers_dim[i+1]
                
                # --- A. Dernière Couche ---
                if is_last_layer and use_final_linear:
                    self.layers.append(nn.Linear(in_d, out_d))
                
                # --- B. Couches Cachées / Spectrales ---
                else:
                    self.layers.append(
                        SpectralBilinearLayer(in_d, out_d, ortho_mode=ortho_mode, bias=bias)
                    )
                    
                    # On ne met jamais de Norm après la toute dernière couche
                    if not is_last_layer and self.use_layernorm:
                        self.layers.append(nn.LayerNorm(out_d))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for layer in self.layers:
                x = layer(x)
            return x

        def get_total_ortho_loss(self, loss_fn: Callable) -> torch.Tensor:
            """Récupère la loss d'orthogonalité totale du réseau."""
            device = next(self.parameters()).device
            total_loss = torch.tensor(0.0, device=device)
            for layer in self.layers:
                if isinstance(layer, SpectralBilinearLayer):
                    total_loss += layer.get_ortho_loss(loss_fn)
            return total_loss

        @property
        def num_parameters(self) -> int:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
