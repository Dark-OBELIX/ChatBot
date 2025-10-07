# 🤖 ChatBot

Ce projet utilise un **PC "serveur"** équipé d’un **GPU** sous **Windows 11** avec **WSL2**, ainsi qu’un **PC portable "client"** pour le développement.

- Le **serveur** exécute PyTorch et CUDA.  
- Le **client** utilise **PyCharm** pour le développement.  
- Les deux machines sont reliées via **SSH** avec **Tailscale**, car elles ne sont pas sur le même réseau.

## ⚙️ Commandes utiles

Vérifier la détection de CUDA et du GPU :
```bash
python -c "import torch; print('CUDA disponible:', torch.cuda.is_available(), 'Nom du GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Aucun GPU')"
```

Activer l’environnement sur la tour
```bash
source ~/pytorch_env/bin/activate
```

Surveiller l’utilisation du GPU:
```bash
watch -n 1 gpustat -cp
```

Lancer le script : dans le repo : 
```bash
run_remote_script main.py
```

Alias :
```bash
nano ~/.bashrc

alias run_remote_script='f() { \
  rsync -avz "/mnt/c/Users/hugol/OneDrive/Documents/0_Perso/0_2_Projets_perso/ChatBot/scripts/" sidiouslinux@100.88.42.33:~/projects/pytorch_projects/scrip>  ssh sidiouslinux@100.88.42.33 "cd ~/projects/pytorch_projects && ./run.sh scripts/$1"; \
}; f'

source ~/.bashrc
```
