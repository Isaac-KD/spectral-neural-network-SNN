import torch
import torch.nn as nn
import torch.nn.init as init
    
class SpectralQuadraticLayer(nn.Module):
    def __init__(self, in_features, out_features, device='cuda'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 1. La Base Partagée (U)
        # On crée une couche linéaire qui servira de matrice de rotation
        # in_features -> in_features (c'est un changement de base interne)
        self.base_change = nn.Linear(in_features, in_features, bias=True)
        
        # 2. Contrainte d'Orthogonalité (Hard Constraint)
        # On force la matrice de poids de self.base_change à rester orthogonale
        # Cela garantit que le signal est tourné mais jamais écrasé ou explosé à cette étape.
        torch.nn.utils.parametrizations.orthogonal(self.base_change, "weight")
        
        # 3. Les Valeurs Propres (Lambdas)
        # C'est la couche qui combine les énergies.
        # Elle prend les in_features au carré et sort out_features.
        # W_lambda représente les coefficients lambda_i pour chaque sortie.
        self.eigen_weights = nn.Linear(in_features, out_features, bias=True)
        
        # Initialisation spéciale pour les lambdas pour éviter l'explosion initiale
        # On initialise proche de zéro ou avec une petite variance
        nn.init.zeros_(self.base_change.bias)
        nn.init.normal_(self.eigen_weights.weight, mean=0.0, std=0.1)

    def forward(self, x):
        # x shape: (Batch, In_Features)
        
        # Étape A : Changement de base (Projection sur les vecteurs propres)
        # h = x @ U.T
        h = self.base_change(x)
        
        # Étape B : Activation Quadratique (Calcul de l'énergie)
        # energy = h^2
        energy = torch.pow(h, 2)
        
        # Étape C : Pondération par les valeurs propres (Combinaison linéaire des énergies)
        # y = energy @ Lambdas.T + bias
        y = self.eigen_weights(energy)
        
        return y

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


