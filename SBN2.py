import torch
import torch.nn as nn
import torch.nn.init as init
from typing import List, Callable, Optional

class SpectralBilinearLayer(nn.Module):
    """
    Version Low-Rank Bilinéaire de la couche spectrale.
    Projection : x (In) -> h (Rank)
    Interaction : h_right * h_left (Rank)
    Mélange : h (Rank) -> y (Out)
    """
    def __init__(
        self, 
        in_features: int,          
        out_features: int, 
        rank: Optional[int] = None, # Nouveau paramètre Low-Rank
        ortho_mode: str = 'hard', 
        bias: bool = True
    ):
        super().__init__()

        valid_modes = {'hard', 'cayley', 'soft', None}
        assert ortho_mode in valid_modes, f"Mode invalide: {ortho_mode}"

        self.in_features = in_features
        # Si rank n'est pas spécifié, on utilise in_features (Full Rank)
        self.rank = rank if rank is not None else in_features
        self.ortho_mode = ortho_mode
        
        # 1. Projections vers l'espace de rang r (In -> Rank)
        self.right_projections = nn.Linear(in_features, self.rank, bias=bias)
        self.left_projections = nn.Linear(in_features, self.rank, bias=bias)
        
        # 2. Contraintes d'Orthogonalité
        if self.ortho_mode in ['hard', 'cayley']:
            map_type = 'cayley' if self.ortho_mode == 'cayley' else None
            torch.nn.utils.parametrizations.orthogonal(self.right_projections, "weight", orthogonal_map=map_type)
            torch.nn.utils.parametrizations.orthogonal(self.left_projections, "weight", orthogonal_map=map_type)

        elif self.ortho_mode == 'soft':
            init.orthogonal_(self.right_projections.weight)
            init.orthogonal_(self.left_projections.weight)
            # La cible d'identité dépend maintenant du Rank
            self.register_buffer('eye_target', torch.eye(self.rank), persistent=False)
        
        # 3. Les Valeurs Propres (Lambdas) : Rank -> Out
        self.eigen_weights = nn.Linear(self.rank, out_features, bias=True)
        
        self._init_parameters()

    def _init_parameters(self) -> None:
            nn.init.xavier_uniform_(self.right_projections.weight, gain=1.5)
            nn.init.xavier_uniform_(self.left_projections.weight, gain=1.5)
            nn.init.xavier_uniform_(self.eigen_weights.weight, gain=1.5)
            
            nn.init.zeros_(self.eigen_weights.bias)
            if self.right_projections.bias is not None: 
                nn.init.zeros_(self.right_projections.bias)
            if self.left_projections.bias is not None: 
                nn.init.zeros_(self.left_projections.bias)

    def get_ortho_loss(self, loss_fn: Callable) -> torch.Tensor:
        if self.ortho_mode != 'soft':
            return torch.tensor(0.0, device=self.right_projections.weight.device)
            
        w_r = self.right_projections.weight # Shape: (Rank, In)
        w_l = self.left_projections.weight # Shape: (Rank, In)
        
        # MM(W, W.t()) produit une matrice (Rank, Rank)
        loss_r = loss_fn(torch.mm(w_r, w_r.t()), self.eye_target)
        loss_l = loss_fn(torch.mm(w_l, w_l.t()), self.eye_target)
        
        return loss_r + loss_l
    
    def forward(self, x):
        # x shape: (Batch, In_Features)
        
        # A. Projection Low-Rank (Batch, Rank)
        h_right = self.right_projections(x)
        h_left = self.left_projections(x)
        
        # B. Interaction Bilinéaire dans l'espace réduit
        bilinear_interaction = h_right * h_left 
        
        # C. Combinaison (Rank -> Out)
        y = self.eigen_weights(bilinear_interaction)
        
        return y

class SpectralBillinearNet(nn.Module):
        def __init__(
            self, 
            layers_dim: List[int],          
            rank_factor: float = 1.0, # Facteur de réduction du rang (ex: 0.5)
            ortho_mode: str = None, 
            use_final_linear: bool = False, 
            bias: bool = True, 
            use_layernorm: bool = False
        ):
            super().__init__()
            self.layers = nn.ModuleList()
            self.use_layernorm = use_layernorm

            assert len(layers_dim) >= 2, "Il faut au moins une dimension d'entrée et de sortie"

            num_layers = len(layers_dim) - 1
            
            for i in range(num_layers):
                is_last_layer = (i == num_layers - 1)
                in_d = layers_dim[i]
                out_d = layers_dim[i+1]
                
                if is_last_layer and use_final_linear:
                    self.layers.append(nn.Linear(in_d, out_d))
                else:
                    # Calcul du rang basé sur le facteur
                    layer_rank = int(in_d * rank_factor) if rank_factor > 0 else in_d
                    # Sécurité pour éviter rank=0
                    layer_rank = max(1, layer_rank)

                    self.layers.append(
                        SpectralBilinearLayer(
                            in_d, 
                            out_d, 
                            rank=layer_rank, # Passage du rank
                            ortho_mode=ortho_mode, 
                            bias=bias
                        )
                    )
                    
                    if not is_last_layer and self.use_layernorm:
                        self.layers.append(nn.LayerNorm(out_d))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for layer in self.layers:
                x = layer(x)
            return x

        def get_total_ortho_loss(self, loss_fn: Callable) -> torch.Tensor:
            device = next(self.parameters()).device
            total_loss = torch.tensor(0.0, device=device)
            for layer in self.layers:
                if isinstance(layer, SpectralBilinearLayer):
                    total_loss += layer.get_ortho_loss(loss_fn)
            return total_loss

        @property
        def num_parameters(self) -> int:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

class MultiBasisBilinearLayer(nn.Module):
    """
    Couche Multi-Bases avec support Low-Rank.
    H bases projettent x (In) vers H espaces de dimension (Rank).
    """
    def __init__(
        self, 
        in_features: int,          
        out_features: int, 
        num_bases: int = 4,
        rank: Optional[int] = None, # Nouveau
        ortho_mode: str = None, 
        bias: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_bases = num_bases
        self.rank = rank if rank is not None else in_features
        self.ortho_mode = ortho_mode

        # 1. Bases Low-Rank : In -> Rank
        self.right_bases = nn.ModuleList([
            nn.Linear(in_features, self.rank, bias=bias) for _ in range(num_bases)
        ])
        self.left_bases = nn.ModuleList([
            nn.Linear(in_features, self.rank, bias=bias) for _ in range(num_bases)
        ])
        
        self._apply_ortho_constraints()
        
        if self.ortho_mode == 'soft':
            # Cible pour (Rank, Rank)
            self.register_buffer('eye_target', torch.eye(self.rank), persistent=False)

        # 2. Mélange : Entrée = Rank * Num_Bases
        total_hidden_dim = self.rank * num_bases
        self.eigen_weights = nn.Linear(total_hidden_dim, out_features, bias=True)
        
        self._init_parameters()

    def _apply_ortho_constraints(self):
        for bases_list in [self.right_bases, self.left_bases]:
            for linear_layer in bases_list:
                if self.ortho_mode == 'hard':
                    torch.nn.utils.parametrizations.orthogonal(linear_layer, "weight")
                elif self.ortho_mode == 'cayley':
                    torch.nn.utils.parametrizations.orthogonal(linear_layer, "weight", orthogonal_map='cayley')
                elif self.ortho_mode == 'soft':
                    init.orthogonal_(linear_layer.weight)

    def _init_parameters(self) -> None:
        for bases_list in [self.right_bases, self.left_bases]:
            for base in bases_list:
                if self.ortho_mode not in ['hard', 'cayley']: 
                    init.orthogonal_(base.weight)
                if base.bias is not None: 
                    init.zeros_(base.bias)
        
        nn.init.xavier_uniform_(self.eigen_weights.weight, gain=1.0)
        init.zeros_(self.eigen_weights.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch, In_Features)
        all_interactions = []
        
        for i in range(self.num_bases):
            # Projection : (Batch, Rank)
            right_proj = self.right_bases[i](x) 
            left_proj = self.left_bases[i](x)   
            
            # Interaction : (Batch, Rank)
            interaction = right_proj * left_proj
            all_interactions.append(interaction)
            
        # Concaténation : (Batch, Rank * Num_Bases)
        combined_interactions = torch.cat(all_interactions, dim=-1)
        
        y = self.eigen_weights(combined_interactions)
        return y

    def get_ortho_loss(self, loss_fn: Callable) -> torch.Tensor:
        if self.ortho_mode != 'soft':
            return torch.tensor(0.0, device=self.eigen_weights.weight.device)
            
        total_loss = torch.tensor(0.0, device=self.eigen_weights.weight.device)
        
        for bases_list in [self.right_bases, self.left_bases]:
            for base in bases_list:
                w = base.weight # (Rank, In)
                # Gram matrix sur la dimension Rank : (Rank, Rank)
                gram = torch.mm(w, w.t())
                total_loss += loss_fn(gram, self.eye_target)
            
        return total_loss


class DeepMultiBasisBilinearNet(nn.Module):
    def __init__(
        self, 
        layers_dim: List[int],
        num_bases: int = 4,
        rank_factor: float = 1.0, # Facteur Low-Rank
        ortho_mode: str = 'hard',
        use_final_linear: bool = True,
        use_layernorm: bool = True,
        use_residual: bool = True,
        bias: bool = True
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.use_layernorm = use_layernorm
        self.use_residual = use_residual
        self.use_final_linear = use_final_linear

        assert len(layers_dim) >= 2, "Il faut au moins une dimension d'entrée et de sortie"
        
        num_layers = len(layers_dim) - 1
        
        for i in range(num_layers):
            is_last_layer = (i == num_layers - 1)
            in_d = layers_dim[i]
            out_d = layers_dim[i+1]
            
            if is_last_layer and use_final_linear:
                self.layers.append(nn.Linear(in_d, out_d, bias=bias))
            else:
                # Calcul du rank
                layer_rank = int(in_d * rank_factor) if rank_factor > 0 else in_d
                layer_rank = max(1, layer_rank)

                self.layers.append(
                    MultiBasisBilinearLayer(
                        in_features=in_d, 
                        out_features=out_d, 
                        num_bases=num_bases,
                        rank=layer_rank, 
                        ortho_mode=ortho_mode,
                        bias=bias
                    )
                )
                
                if not is_last_layer and self.use_layernorm:
                    self.layers.append(nn.LayerNorm(out_d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            if isinstance(layer, MultiBasisBilinearLayer):
                out = layer(x)
                if self.use_residual and x.shape == out.shape:
                    x = x + out
                else:
                    x = out
            else:
                x = layer(x)      
        return x

    def get_total_ortho_loss(self, loss_fn: Callable) -> torch.Tensor:
        loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.layers:
            if isinstance(layer, MultiBasisBilinearLayer):
                loss += layer.get_ortho_loss(loss_fn)
        return loss
    
    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)