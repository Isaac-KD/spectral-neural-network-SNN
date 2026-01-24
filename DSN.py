import torch
import torch.nn as nn
import torch.nn.init as init
from typing import List, Callable

class SpectralQuadraticLayer(nn.Module):
    def __init__(
        self, 
        in_features: int,          
        out_features: int, 
        ortho_mode: str = 'hard', 
        bias: bool = True
    ):
        """
        Args:
            in_features (int): Dimension d'entrée
            out_features (int): Dimension de sortie
            ortho_mode (str): 
                - 'hard': Force l'orthogonalité exacte (parametrization). Plus lent, très stable.
                - 'soft': Pas de contrainte forcée, mais permet de calculer une loss de régularisation. Rapide.
                - None  : Pas d'orthogonalité.
        """
        super().__init__()

        # 1. Defensive Programming : On valide les entrées tout de suite
        valid_modes = {'hard', 'cayley', 'soft', None}
        assert ortho_mode in valid_modes, f"Mode invalide: {ortho_mode}. Choisir parmi {valid_modes}"

        self.in_features = in_features
        self.ortho_mode = ortho_mode
        
         # 1. La Base Partagée (U)
        # On crée une couche linéaire qui servira de matrice de rotation
        # in_features -> in_features (c'est un changement de base interne)
        self.base_change = nn.Linear(in_features, in_features, bias=bias)
        
        # 2. Contrainte d'Orthogonalité (Hard Constraint)
        # On force la matrice de poids de self.base_change à rester orthogonale
        # Cela garantit que le signal est tourné mais jamais écrasé ou explosé à cette étape.
        if self.ortho_mode == 'hard':
            # Contrainte stricte : la matrice reste toujours orthogonale
            torch.nn.utils.parametrizations.orthogonal(self.base_change, "weight")

        elif self.ortho_mode == 'cayley':
            # Mode "Cayley" : Utilise la paramétrisation de Cayley explicite
            # Q = (I + A)(I - A)^-1
            torch.nn.utils.parametrizations.orthogonal(self.base_change, "weight", orthogonal_map='cayley')

        elif self.ortho_mode == 'soft':
            # Initialisation orthogonale de départ, mais liberté d'évolution ensuite
            init.orthogonal_(self.base_change.weight)
            # OPTIMISATION : On pré-calcule la matrice identité pour éviter de la 
            # recréer à chaque appel de get_ortho_loss. 
            # 'persistent=False' signifie qu'elle ne sera pas sauvegardée dans le state_dict (pas besoin de la save).
            self.register_buffer('eye_target', torch.eye(in_features), persistent=False)
        
        # 3. Les Valeurs Propres (Lambdas)
        # C'est la couche qui combine les énergies.
        # Elle prend les in_features au carré et sort out_features.
        # W_lambda représente les coefficients lambda_i pour chaque sortie.
        self.eigen_weights = nn.Linear(in_features, out_features, bias=True)
        
        # Initialisation personnalisée
        self._init_parameters()

    def _init_parameters(self)-> None:
        # Initialisation spéciale pour les lambdas pour éviter l'explosion initiale
        nn.init.xavier_uniform_(self.base_change.weight, gain=1.5)
        nn.init.xavier_uniform_(self.eigen_weights.weight, gain=1.5)
        init.zeros_(self.eigen_weights.bias)
        if self.base_change.bias is not None: init.zeros_(self.base_change.bias)

    def get_ortho_loss(self, loss_fn: Callable) -> torch.Tensor:
        """
        Calcule la perte d'orthogonalité pour le mode 'soft'.
        
        Args:
            loss_fn (callable): Fonction de perte à utiliser (ex: F.mse_loss, F.l1_loss).
                                Par défaut : Mean Squared Error.
        """
        if self.ortho_mode != 'soft':
            return torch.tensor(0.0, device=self.base_change.weight.device)
            
        w = self.base_change.weight
        gram = torch.mm(w, w.t())
        
        # Calcul de la perte avec la fonction fournie
        return loss_fn(gram, self.eye_target)
    
    def forward(self, x):
        # x shape: (Batch, In_Features)
        
        # Étape A : Changement de base (Projection sur les vecteurs propres)
        # h = x @ U.T
        h = self.base_change(x)
        
        # Étape B : Activation Quadratique (Calcul de l'énergie)
        # energy = h^2
        energy = h * h 
        
        # C. Combinaison linéaire
        y = self.eigen_weights(energy)
        
        return y
    
    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, mode={self.ortho_mode}'

class DeepSpectralNet(nn.Module):
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
                    SpectralQuadraticLayer(in_d, out_d, ortho_mode=ortho_mode, bias=bias)
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
            if isinstance(layer, SpectralQuadraticLayer):
                total_loss += layer.get_ortho_loss(loss_fn)
        return total_loss

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)