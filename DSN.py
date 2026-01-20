import torch
import torch.nn as nn
import torch.nn.init as init
    

class SpectralQuadraticLayer(nn.Module):
    def __init__(self, in_features, out_features, ortho_mode='hard',bias=True):
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
            # OPTIMISATION 1 : On pré-calcule la matrice identité pour éviter de la 
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

    def _init_parameters(self):
        # Initialisation spéciale pour les lambdas pour éviter l'explosion initiale
        # On initialise proche de zéro ou avec une petite variance
        init.normal_(self.eigen_weights.weight, mean=0.0, std=0.01)
        init.zeros_(self.eigen_weights.bias)
        if self.base_change.bias is not None: init.zeros_(self.base_change.bias)

    def get_ortho_loss(self, loss_fn):
        """
        Calcule la perte d'orthogonalité pour le mode 'soft'.
        
        Args:
            loss_fn (callable): Fonction de perte à utiliser (ex: F.mse_loss, F.l1_loss).
                                Par défaut : Mean Squared Error.
        """
        if self.ortho_mode != 'soft':
            return 0.0
            
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
    
    def get_param_count(self):
        """
        Retourne le nombre de paramètres entraînables dans cette couche.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
class DeepSpectralNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, depth=3):
        super().__init__()
        layers = []
        
        # Entrée -> Hidden
        layers.append(SpectralQuadraticLayer(input_dim, hidden_dim))
        
        # Hidden -> Hidden (Profondeur)
        for _ in range(depth):
            layers.append(SpectralQuadraticLayer(hidden_dim, hidden_dim))
            # Note: On peut ajouter une Normalisation ici pour stabiliser les carrés successifs
            layers.append(nn.LayerNorm(hidden_dim)) 
            
        # Hidden -> Output
        layers.append(SpectralQuadraticLayer(hidden_dim, output_dim)) # Couche finale linéaire standard ou quadratique
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class DeepSpectralNet(nn.Module):
    def __init__(self, layers_dim, ortho_mode='cayley', use_final_linear=False, bias=True,use_layernorm=False):
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

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def get_total_ortho_loss(self, loss_fn):
        """Récupère la loss d'orthogonalité totale du réseau."""
        total_loss = 0.0
        for layer in self.layers:
            if isinstance(layer, SpectralQuadraticLayer):
                total_loss += layer.get_ortho_loss(loss_fn)
        return total_loss

    def get_param_count(self):
        """Retourne le nombre total de paramètres du réseau."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
