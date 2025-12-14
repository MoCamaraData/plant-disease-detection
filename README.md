 Plant Disease Detection API – README
 1. Description du projet

Ce projet propose une API capable de prédire la maladie présente sur une feuille de plante à partir d’une image.
L’API est développée avec FastAPI, conteneurisée avec Docker, puis déployée sur Google Cloud Run pour une mise en production simple, scalable et professionnelle.

 2. Exécution de l’API localement
 Prérequis

Python 3.12

pip

Virtualenv (recommandé)

Étapes d’installation

Se placer dans le dossier du projet

cd plant_project


Créer et activer un environnement virtuel

python -m venv venv
venv\Scripts\activate


Installer les dépendances

pip install -r requirements.txt


Lancer l’API

uvicorn api:app --reload


Accéder à l’API

http://localhost:8000

http://localhost:8000/docs

http://localhost:8000/health

3. Construction et exécution avec Docker
Construire l’image Docker
docker build -t plant-api .

Lancer le conteneur
docker run -p 8000:8000 plant-api

Accès

http://localhost:8000

http://localhost:8000/docs

4. Manuel d’utilisation de l’API
Objectif

Ce manuel explique comment utiliser l’API pour obtenir une prédiction de maladie à partir d’une image de feuille, en local, via Docker ou via la version déployée sur Google Cloud Run.

 Vérification de l’état de l’API

Avant toute prédiction, il est recommandé de vérifier que le modèle est bien chargé.

Endpoint
GET /health

Exemple
curl http://localhost:8000/health

Réponse attendue
{
  "status": "ok",
  "model_loaded": true,
  "model_path": "./plant-disease-model.pth"
}


Si model_loaded vaut true, l’API est prête à être utilisée.

 Effectuer une prédiction

L’endpoint principal /predict permet d’envoyer une image et de recevoir la maladie prédite.

Endpoint
POST /predict

Format requis

Type : multipart/form-data

Champ : file

Valeur : image (.jpg, .jpeg, .png)

 Utilisation via Swagger UI

Ouvrir la documentation interactive :

Local : http://localhost:8000/docs

Cloud Run : https://<votre-url-cloud-run>/docs

Sélectionner POST /predict

Cliquer sur Try it out

Télécharger une image de feuille

Cliquer sur Execute

 Utilisation via la ligne de commande (curl)

Local :

curl -X POST -F "file=@leaf.jpg" http://localhost:8000/predict


Cloud Run :

curl -X POST -F "file=@leaf.jpg" https://<votre-url-cloud-run>/predict

🔹 Exemple de réponse
{
  "status": "success",
  "prediction": {
    "class": 20,
    "index": 20,
    "probability": 0.99
  },
  "top3": [
    { "class": "20", "probability": 0.99 },
    { "class": "29", "probability": 0.000000002 },
    { "class": "30", "probability": 0.000000001 }
  ]
}

 5. Structure du projet
plant_project/
│── api.py
│── plant-disease-model.pth
│── requirements.txt
│── Dockerfile
│── .dockerignore
│── journal_de_projet.md
│── README.md

 6. Déploiement sur Google Cloud Run
Build et push
gcloud builds submit --tag northamerica-northeast1-docker.pkg.dev/plant-disease-detection-480204/plant-repo/plant-api

Déploiement
gcloud run deploy plant-api --image northamerica-northeast1-docker.pkg.dev/plant-disease-detection-480204/plant-repo/plant-api --platform managed --region northamerica-northeast1 --allow-unauthenticated

Accès en ligne

L’API est accessible via l’URL Cloud Run fournie après le déploiement :

https://plant-api-xxxxx.run.app](https://plant-api-53813388828.northamerica-northeast1.run.app

7. Problèmes rencontrés et ajustements

Erreur PORT=8080 : corrigée en utilisant la variable d’environnement $PORT dans le Dockerfile.

Service Unavailable (Cloud Run) : causé par un compte de service inexistant, corrigé en assignant le compte par défaut Compute Engine.

Problèmes de dépendances locales : utilisation de Python 3.12 pour compatibilité avec Pillow.

 8. Conclusion

Cette API fournit une solution complète et déployée pour la détection de maladies des plantes.
Elle fonctionne en local, dans Docker et en production sur Google Cloud Run, et peut facilement être connectée à une interface utilisateur comme Streamlit.
