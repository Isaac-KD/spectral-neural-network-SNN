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


class MultiHeadSpectralLayer(nn.Module):
    """
    Couche Spectrale Multi-Têtes (Pure).
    
    Principe :
    1. L'entrée est envoyée à N têtes parallèles.
    2. Chaque tête est une 'SpectralQuadraticLayer' indépendante.
       Elle observe tout le contexte (in_features) mais projette vers une dimension réduite (head_dim).
    3. Les sorties sont concaténées pour reconstituer la dimension d'origine.
    
    Aucun MLP, aucune autre transformation n'est ajoutée.
    """
    def __init__(
        self, 
        in_features: int, 
        num_heads: int, 
        ortho_mode: str = 'hard',
        bias: bool = True
    ):
        super().__init__()
        
        # Vérification des dimensions
        if in_features % num_heads != 0:
            raise ValueError(f"in_features ({in_features}) doit être divisible par num_heads ({num_heads})")
        
        self.in_features = in_features
        self.num_heads = num_heads
        self.head_dim = in_features // num_heads
        
        # Création des têtes spectrales parallèles
        # Chaque tête : Input (Full) -> Projection Orthogonale -> Carré -> Output (Partiel)
        self.heads = nn.ModuleList([
            SpectralQuadraticLayer(
                in_features=in_features, 
                out_features=self.head_dim, 
                ortho_mode=ortho_mode, 
                bias=bias
            ) 
            for _ in range(num_heads)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch, ..., In_Features)
        
        # 1. Application parallèle des têtes
        # Chaque tête renvoie un tenseur de forme (Batch, ..., Head_Dim)
        head_outputs = [head(x) for head in self.heads]
        
        # 2. Concaténation pure et simple
        # On reconstitue le vecteur complet : (Batch, ..., In_Features)
        output = torch.cat(head_outputs, dim=-1)
        
        return output

    def get_ortho_loss(self, loss_fn: Callable) -> torch.Tensor:
        """Somme des pertes d'orthogonalité de chaque tête."""
        total_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for head in self.heads:
            total_loss += head.get_ortho_loss(loss_fn)
        return total_loss


class DeepMultiHeadSpectralNet(nn.Module):
    """
    Réseau Profond 'Pure DSN'.
    Empilement de couches Multi-Head Spectrales sans MLP intermédiaires.
    
    Structure par bloc :
    Input -> [MultiHeadSpectralLayer] -> (Optionnel: Norm/Residual) -> Output
    """
    def __init__(
        self, 
        dim: int,
        num_layers: int,
        num_heads: int,
        ortho_mode: str = 'hard',
        use_residuals: bool = True,
        use_layernorm: bool = True
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.use_residuals = use_residuals
        self.use_layernorm = use_layernorm
        
        for _ in range(num_layers):
            self.layers.append(
                MultiHeadSpectralLayer(
                    in_features=dim, 
                    num_heads=num_heads, 
                    ortho_mode=ortho_mode
                )
            )
            # Normalisation optionnelle pour éviter l'explosion des gradients 
            # (les réseaux polynomiaux sont sensibles à la profondeur)
            if use_layernorm:
                self.layers.append(nn.LayerNorm(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            if isinstance(layer, MultiHeadSpectralLayer):
                # Connexion résiduelle : x = x + Layer(x)
                # C'est la seule opération "extra" permise pour aider la convergence
                if self.use_residuals:
                    x = x + layer(x)
                else:
                    x = layer(x)
            else:
                # C'est une LayerNorm ou autre utilitaire
                x = layer(x)
        return x

    def get_total_ortho_loss(self, loss_fn: Callable) -> torch.Tensor:
        loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.layers:
            if isinstance(layer, MultiHeadSpectralLayer):
                loss += layer.get_ortho_loss(loss_fn)
        return loss

class MultiBasisSpectralLayer(nn.Module):
    """
    Couche Spectrale Multi-Bases.
    
    Contrairement à l'approche "Multi-Head" classique qui divise la dimension,
    cette couche :
    1. Apprend H bases orthogonales complètes (H matrices de rotation de taille N*N).
    2. Projette l'entrée sur ces H bases.
    3. Élève au carré pour obtenir H * N énergies spectrales.
    4. Chaque neurone de sortie apprend H * N valeurs propres (poids de mélange).
    """
    def __init__(
        self, 
        in_features: int,          
        out_features: int, 
        num_bases: int = 4,   # H bases
        ortho_mode: str = 'hard', 
        bias: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_bases = num_bases
        self.ortho_mode = ortho_mode

        # 1. Les H Bases Partagées (Liste de matrices de rotation)
        # Chaque base est une transformation linéaire complète : In -> In
        self.bases = nn.ModuleList([
            nn.Linear(in_features, in_features, bias=bias) 
            for _ in range(num_bases)
        ])
        
        # Application des contraintes d'orthogonalité sur chaque base
        for linear_layer in self.bases:
            if self.ortho_mode == 'hard':
                torch.nn.utils.parametrizations.orthogonal(linear_layer, "weight")
            elif self.ortho_mode == 'cayley':
                torch.nn.utils.parametrizations.orthogonal(linear_layer, "weight", orthogonal_map='cayley')
            elif self.ortho_mode == 'soft':
                init.orthogonal_(linear_layer.weight)
        
        if self.ortho_mode == 'soft':
            self.register_buffer('eye_target', torch.eye(in_features), persistent=False)

        # 2. Les Valeurs Propres (H * K valeurs par neurone de sortie)
        # L'entrée de cette couche est la concaténation des énergies de toutes les bases.
        # Taille entrée : in_features * num_bases
        # Taille sortie : out_features
        self.eigen_weights = nn.Linear(in_features * num_bases, out_features, bias=True)
        
        self._init_parameters()

    def _init_parameters(self) -> None:
        # Initialisation pour favoriser la convergence
        for base in self.bases:
            # On initialise les bases pour qu'elles couvrent bien l'espace
            if self.ortho_mode not in ['hard', 'cayley']: 
                init.orthogonal_(base.weight)
            if base.bias is not None: 
                init.zeros_(base.bias)
        
        # Initialisation Xavier pour les valeurs propres
        nn.init.xavier_uniform_(self.eigen_weights.weight, gain=1.0)
        init.zeros_(self.eigen_weights.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch, In_Features)
        
        all_energies = []
        
        # 1. Calcul des énergies pour chaque base
        for base in self.bases:
            # Projection : h = W_i * x
            h = base(x)
            # Activation quadratique : e = h^2
            energy = h * h
            all_energies.append(energy)
            
        # 2. Concaténation de toutes les énergies
        # Shape: (Batch, In_Features * Num_Bases)
        combined_energy = torch.cat(all_energies, dim=-1)
        
        # 3. Combinaison linéaire (Valeurs propres)
        # y = Sum( lambda_ijk * energy_jk )
        y = self.eigen_weights(combined_energy)
        
        return y

    def get_ortho_loss(self, loss_fn: Callable) -> torch.Tensor:
        """Calcule la perte d'orthogonalité cumulée sur les H bases."""
        if self.ortho_mode != 'soft':
            return torch.tensor(0.0, device=self.eigen_weights.weight.device)
            
        total_loss = torch.tensor(0.0, device=self.eigen_weights.weight.device)
        
        for base in self.bases:
            w = base.weight
            gram = torch.mm(w, w.t())
            total_loss += loss_fn(gram, self.eye_target)
            
        return total_loss

class DeepMultiBasisNet(nn.Module):
    """
    Réseau Profond utilisant l'architecture Multi-Bases avec dimensions variables.
    
    Structure flexible définie par `layers_dim`, similaire à un MLP classique
    mais utilisant des projections spectrales multi-bases.
    """
    def __init__(
        self, 
        layers_dim: List[int],
        num_bases: int = 4,
        ortho_mode: str = 'hard',
        use_final_linear: bool = False,
        use_layernorm: bool = True,
        use_residual: bool = False,
        bias: bool = True
    ):
        """
        Args:
            layers_dim (list): Liste des dimensions [in, hidden, ..., out].
            num_bases (int): Nombre de bases orthogonales par couche (H).
            ortho_mode (str): 'hard', 'cayley', 'soft', ou None.
            use_final_linear (bool): Si True, la dernière couche est un nn.Linear standard 
                                     (utile pour les logits de classification).
            use_layernorm (bool): Ajoute une LayerNorm après chaque couche spectrale.
            use_residual (bool): Active les connexions résiduelles (seulement si in_dim == out_dim).
            bias (bool): Active le biais dans les couches.
        """
        super().__init__()
        self.layers = nn.ModuleList()
        self.use_layernorm = use_layernorm
        self.use_residual = use_residual
        self.use_final_linear = use_final_linear

        assert len(layers_dim) >= 2, "Il faut au moins une dimension d'entrée et de sortie (len >= 2)"
        
        num_layers = len(layers_dim) - 1
        
        for i in range(num_layers):
            is_last_layer = (i == num_layers - 1)
            in_d = layers_dim[i]
            out_d = layers_dim[i+1]
            
            # --- Cas A : Dernière Couche Linéaire Standard ---
            # Souvent préférable pour projeter vers des logits (valeurs non bornées/négatives)
            if is_last_layer and use_final_linear:
                self.layers.append(nn.Linear(in_d, out_d, bias=bias))
            
            # --- Cas B : Couche Spectrale Multi-Bases ---
            else:
                self.layers.append(
                    MultiBasisSpectralLayer(
                        in_features=in_d, 
                        out_features=out_d, 
                        num_bases=num_bases, 
                        ortho_mode=ortho_mode,
                        bias=bias
                    )
                )
                
                # Normalisation (Sauf après la toute dernière couche pour ne pas contraindre la sortie)
                if not is_last_layer and self.use_layernorm:
                    self.layers.append(nn.LayerNorm(out_d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # On itère sur les couches.
        # Attention : self.layers contient un mélange de SpectralLayer et de LayerNorm.
        
        for layer in self.layers:
            # Gestion des connexions résiduelles
            if isinstance(layer, MultiBasisSpectralLayer):
                out = layer(x)
                
                # On applique le résiduel SEULEMENT si demandé ET si les dimensions correspondent
                # (Impossible d'additionner x et out si in_features != out_features)
                if self.use_residual and x.shape == out.shape:
                    x = x + out
                else:
                    x = out
            else:
                # C'est soit une LayerNorm, soit un Linear final
                x = layer(x)
                
        return x

    def get_total_ortho_loss(self, loss_fn: Callable) -> torch.Tensor:
        """Récupère la loss d'orthogonalité de toutes les couches MultiBasis."""
        device = next(self.parameters()).device
        loss = torch.tensor(0.0, device=device)
        
        for layer in self.layers:
            if isinstance(layer, MultiBasisSpectralLayer):
                loss += layer.get_ortho_loss(loss_fn)
        return loss
    
    @property
    def num_parameters(self) -> int:
        """Retourne le nombre total de paramètres entraînables."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    