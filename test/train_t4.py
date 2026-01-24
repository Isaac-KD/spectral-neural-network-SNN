# config/train_t4.py

out_dir = 'out-spectral-net'
eval_interval = 250 # On vérifie souvent
eval_iters = 20
log_interval = 10 # On veut voir les logs défiler

always_save_checkpoint = False # On ne sauvegarde pas tout le temps pour aller vite

wandb_log = False # Mets True si tu as un compte Weights & Biases
wandb_project = 'spectral-gpt'
wandb_run_name = 't4-run'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 128 # On réduit la mémoire à court terme

# Architecture (Petit modèle pour valider ton code)
n_layer = 1      # Au lieu de 6 
n_head = 2          # Au lieu de 6
n_embd = 16      # Au lieu de 384 
dropout = 0.0    # Pas de dropout sur un si petit modèle

max_iters = 30000        # On pousse à 30k
learning_rate = 8e-3     # On monte encore le LR (0.008) pour forcer les petits poids à bouger
lr_decay_iters = 30000

min_lr = 1e-4 
beta2 = 0.99 

warmup_iters = 200       # Un peu plus de temps pour stabiliser au début

# --- T4 SPECIFIC SETTINGS ---
device = 'cuda'
compile = True # Ça va booster la vitesse
dtype = 'float16' # OBLIGATOIRE sur T4 (sinon ça plante ou c'est lent)



##Modèle,Paramètres,Train Loss,Val Loss,Vitesse (ms/iter),État Final
#Gros (Standard),25.72 M,0.92,1.58,~190 ms,Overfitting (Trop de capacité)
#Petit (Tiny),1.44 M,1.06,1.62,~16 ms,Très bon équilibre
#Nano,0.36 M,1.37,1.61,~16 ms,"Le ""Sweet Spot"" (Efficience max)"
#Extreme-Nano,0.10 M,1.34,1.61,~15 ms,La limite de la cohérence
#Atome (Révolution),0.02 M,1.60,1.79,~13 ms,L'exploit technique


#Niveau,Taille (Params),Ton Exploit,Résultat Linguistique
#Goliath,25 000 000,La force brute.,"Texte parfait, mais trop lourd (25 MB)."
#Standard,1 400 000,Le point de référence.,"Très fluide, Shakespeare pur."
#Nano,360 000,L'efficience pure.,Même score que le 1.4M avec 4x moins de poids.
#Micro,100 000,La barrière psychologique.,Écrit encore des phrases complexes (beseech).
#Atome,20 000,La révolution.,Identifie encore les personnages (ROMEO).
#Neutron,5 000,Le Record.,"Structure préservée, mais invente son propre dialecte."