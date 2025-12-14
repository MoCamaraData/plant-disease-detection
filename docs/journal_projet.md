#  Journal de Projet – API FastAPI pour la Détection de Maladies des Plantes

##  Objectif du projet
Le but du projet est de développer une API capable de recevoir une image de feuille et d’identifier automatiquement la maladie correspondante.  
L’API sert d’interface entre un modèle IA (ResNet9) et un utilisateur ou une application externe.

Elle permet :
- de charger automatiquement le modèle sauvegardé,
- de prétraiter des images,
- d’effectuer une prédiction au format JSON,
- d’être déployée facilement grâce à Docker.

---

##  Technologies utilisées
- FastAPI (framework API)
- Uvicorn (serveur ASGI)
- PyTorch & Torchvision (chargement du modèle + preprocessing)
- Pillow (lecture des images)
- python-multipart (upload multipart)
- Docker (conteneurisation)
- Python 3.12
##  Architecture de l’API

L’API contient trois endpoints principaux :

| Endpoint | Méthode | Description |
|---------|----------|-------------|
| `/` | GET | Page d’accueil |
| `/health` | GET | Vérifie l’état du modèle et du serveur |
| `/predict` | POST | Reçoit une image et retourne la classe prédite |

Le modèle est chargé **une seule fois** au démarrage grâce au décorateur `@app.on_event("startup")`.

---

## Chargement du modèle

Le modèle sauvegardé (`plant-disease-model.pth`) contient un **state_dict**.  
Pour pouvoir le charger correctement, l’API doit :

1. reconstruire l’architecture ResNet9,
2. charger les poids avec `load_state_dict`,
3. mettre le modèle en mode évaluation avec `model.eval()`.

Le chemin du modèle est :

```python
MODEL_PATH = "./plant-disease-model.pth"
##  Dockerisation de l’API

L’objectif de la dockerisation est de rendre l’API :
- portable (même comportement sur n’importe quelle machine),
- indépendante de l’environnement local (version de Python, packages, etc.),
- simple à lancer avec une seule commande Docker.

L’API et le modèle sont intégrés dans une image Docker basée sur Python 3.12.

###  Image Docker

Une image personnalisée `plant-api` est construite à partir d’un `Dockerfile` qui :

- utilise une image de base `python:3.12-slim`,
- installe les dépendances système nécessaires (Pillow, etc),
- installe PyTorch en version CPU,
- installe les dépendances de l’API via `requirements.txt`,
- copie le fichier du modèle `plant-disease-model.pth`,
- expose le port `8000`,
- lance le serveur Uvicorn avec FastAPI.

La commande de construction est :

```bash
docker build -t plant-api .
