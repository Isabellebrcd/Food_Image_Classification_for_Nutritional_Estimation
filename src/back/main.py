import tensorflow as tf

print(f"TensorFlow version: {tf.__version__}")

try:
    from tensorflow import keras

    print("Import TensorFlow Keras réussi ✅")
except ImportError as e:
    print(f"Erreur d'import TensorFlow Keras: {e}")
from tensorflow import keras
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

import numpy as np
import os
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
import logging
import traceback
import matplotlib.pyplot as plt
from typing import Dict, Any, Union, Literal
import uuid
import time
import json
from pathlib import Path

# Configuration du logging
logging.basicConfig(
    level=logging.DEBUG,  # Niveau DEBUG pour plus de détails
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api_debug.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Dossier pour sauvegarder les images reçues (pour débogage)
DEBUG_DIR = "debug_images"
os.makedirs(DEBUG_DIR, exist_ok=True)

# Configuration des modèles disponibles
MODEL_CONFIGS = {
    "EfficientNetV2B2": {
        "file": "food101_EfficientNetV2B2_model_whole.h5",
        "input_size": (260, 260),
        "preprocessing": "EfficientNetV2B2",  # Utilise preprocess_input
    },
    "CNN": {
        "file": "cnn_food101_allclasses__final.h5",  # Remplacez par le nom réel de votre deuxième modèle
        "input_size": (224, 224),  # Ajustez selon les besoins de votre modèle
        "preprocessing": "standard",  # Utilise simple normalisation
    },
    "InceptionV3": {
        "file": "food101_Inception_model_finetuning_wholeds_.h5",  # Remplacez par le nom réel de votre troisième modèle
        "input_size": (299, 299),  # Ajustez selon les besoins de votre modèle
        "preprocessing": "standard",  # Utilise simple normalisation
    }
}

# Modèle actuellement chargé
current_model = None
current_model_name = None

# Classes du dataset Food-101
class_names = [
    'apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare', 'beet_salad', 'beignets',
    'bibimbap', 'bread_pudding', 'breakfast_burrito', 'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad',
    'carrot_cake', 'ceviche', 'cheesecake', 'cheese_plate', 'chicken_curry', 'chicken_quesadilla', 'chicken_wings',
    'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder', 'club_sandwich', 'crab_cakes', 'creme_brulee',
    'croque_madame', 'cup_cakes', 'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict', 'escargots',
    'falafel', 'filet_mignon', 'fish_and_chips', 'foie_gras', 'french_fries', 'french_onion_soup', 'french_toast',
    'fried_calamari', 'fried_rice', 'frozen_yogurt', 'garlic_bread', 'gnocchi', 'greek_salad',
    'grilled_cheese_sandwich',
    'grilled_salmon', 'guacamole', 'gyoza', 'hamburger', 'hot_and_sour_soup', 'hot_dog', 'huevos_rancheros', 'hummus',
    'ice_cream', 'lasagna', 'lobster_bisque', 'lobster_roll_sandwich', 'macaroni_and_cheese', 'macarons', 'miso_soup',
    'mussels', 'nachos', 'omelette', 'onion_rings', 'oysters', 'pad_thai', 'paella', 'pancakes', 'panna_cotta',
    'peking_duck',
    'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib', 'pulled_pork_sandwich', 'ramen', 'ravioli', 'red_velvet_cake',
    'risotto', 'samosa', 'sashimi', 'scallops', 'seaweed_salad', 'shrimp_and_grits', 'spaghetti_bolognese',
    'spaghetti_carbonara', 'spring_rolls', 'steak', 'strawberry_shortcake', 'sushi', 'tacos', 'takoyaki', 'tiramisu',
    'tuna_tartare', 'waffles'
]


def load_nutrition_data():
    """Charge les données nutritionnelles depuis le fichier JSON"""
    try:
        nutrition_file = Path("nutrition.json")
        if not nutrition_file.exists():
            logger.warning(f"Fichier de nutrition non trouvé: {nutrition_file}")
            return {}

        with open(nutrition_file, "r") as f:
            nutrition_data = json.load(f)

        logger.info(f"Données nutritionnelles chargées pour {len(nutrition_data)} plats")
        return nutrition_data
    except Exception as e:
        logger.error(f"Erreur lors du chargement des données nutritionnelles: {str(e)}")
        return {}

nutrition_data = load_nutrition_data()

def load_model(model_name: str):
    """Charge le modèle spécifié"""
    global current_model, current_model_name

    if current_model_name == model_name:
        logger.info(f"Modèle '{model_name}' déjà chargé")
        return current_model

    if model_name not in MODEL_CONFIGS:
        error_msg = f"Modèle '{model_name}' non trouvé dans la configuration"
        logger.error(error_msg)
        raise ValueError(error_msg)

    model_config = MODEL_CONFIGS[model_name]
    model_path = model_config["file"]

    # Vérifier l'existence du modèle
    if not os.path.exists(model_path):
        error_msg = f"Le fichier du modèle n'existe pas: {model_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    # Charger le modèle
    try:
        logger.info(f"Chargement du modèle '{model_name}' depuis: {model_path}")
        model = tf.keras.models.load_model(model_path)
        logger.info(f"Modèle '{model_name}' chargé avec succès")

        # Vérifier la structure du modèle
        input_shape = model.input_shape
        output_shape = model.output_shape
        logger.info(f"Forme d'entrée du modèle: {input_shape}")
        logger.info(f"Forme de sortie du modèle: {output_shape}")

        # Mettre à jour les variables globales
        current_model = model
        current_model_name = model_name

        return model
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle '{model_name}': {str(e)}")
        logger.error(traceback.format_exc())
        raise


# Charger le modèle par défaut au démarrage
try:
    default_model = "EfficientNetV2B2"
    load_model(default_model)
except Exception as e:
    logger.error(f"Erreur lors du chargement du modèle par défaut: {str(e)}")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"message": "Food-101 Classifier API", "status": "running", "current_model": current_model_name}


@app.get("/models")
def get_models():
    """Retourne la liste des modèles disponibles"""
    return {
        "models": list(MODEL_CONFIGS.keys()),
        "current_model": current_model_name,
        "details": {
            name: {
                "input_size": config["input_size"],
                "preprocessing": config["preprocessing"]
            }
            for name, config in MODEL_CONFIGS.items()
        }
    }


@app.get("/test-model")
def test_model(model_name: str = Query(None)):
    """Endpoint pour tester le modèle avec une image de test aléatoire"""
    if model_name:
        try:
            load_model(model_name)
        except Exception as e:
            return {"test_successful": False, "error": str(e)}

    try:
        # Créer une image test aléatoire
        input_size = MODEL_CONFIGS[current_model_name]["input_size"]
        test_img = np.random.rand(input_size[0], input_size[1], 3)
        test_img_batch = np.expand_dims(test_img, axis=0)

        # Prédiction
        start_time = time.time()
        pred = current_model.predict(test_img_batch)
        elapsed_time = time.time() - start_time

        # Analyser la prédiction
        pred_index = np.argmax(pred[0])
        pred_value = float(pred[0][pred_index])

        # Vérifier la distribution des prédictions
        is_uniform = np.allclose(pred[0], pred[0][0], rtol=1e-3)
        std_dev = np.std(pred[0])

        return {
            "test_successful": True,
            "model_name": current_model_name,
            "prediction_index": int(pred_index),
            "prediction_class": class_names[pred_index],
            "prediction_value": pred_value,
            "time_taken_ms": elapsed_time * 1000,
            "distribution_uniform": is_uniform,
            "standard_deviation": float(std_dev),
            "min_value": float(np.min(pred[0])),
            "max_value": float(np.max(pred[0])),
            "output_shape": pred.shape
        }
    except Exception as e:
        logger.error(f"Erreur lors du test du modèle: {str(e)}")
        logger.error(traceback.format_exc())
        return {"test_successful": False, "error": str(e)}


def save_debug_image(image, stage="original", suffix=""):
    """Sauvegarde l'image pour le débogage"""
    try:
        if isinstance(image, np.ndarray):
            # Convertir l'array numpy en image PIL
            if image.ndim == 3 and image.shape[2] == 3:
                # Normaliser si nécessaire
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                debug_img = Image.fromarray(image)
            else:
                logger.warning(f"Format d'image inattendu: {image.shape}")
                return
        else:
            debug_img = image

        # Générer un nom de fichier unique
        filename = f"{stage}_{uuid.uuid4().hex[:8]}{suffix}.png"
        filepath = os.path.join(DEBUG_DIR, filename)

        # Sauvegarder l'image
        debug_img.save(filepath)
        logger.info(f"Image de débogage sauvegardée: {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde de l'image de débogage: {str(e)}")
        return None


def preprocess_image(image: Image.Image, model_name: str) -> np.ndarray:
    """
    Prétraite une image pour la prédiction avec le modèle spécifié.
    """
    try:
        model_config = MODEL_CONFIGS[model_name]
        target_size = model_config["input_size"]
        preprocessing_method = model_config["preprocessing"]

        # Sauvegarder l'image originale pour débogage
        save_debug_image(image, "original")

        # Vérifier le mode de l'image
        if image.mode != "RGB":
            logger.info(f"Conversion de l'image du mode {image.mode} à RGB")
            image = image.convert("RGB")
            save_debug_image(image, "converted_rgb")

        # Redimensionner l'image selon les besoins du modèle
        logger.info(f"Redimensionnement de l'image de {image.size} à {target_size}")
        image = image.resize(target_size)
        save_debug_image(image, "resized")

        # Convertir en tableau numpy
        img_array = np.array(image)
        logger.info(
            f"Forme de l'array avant normalisation: {img_array.shape}, type: {img_array.dtype}, min: {img_array.min()}, max: {img_array.max()}")

        # Normaliser selon la méthode de prétraitement configurée
        if preprocessing_method == "EfficientNetV2B2":
            img_array = preprocess_input(img_array)
            logger.info("Utilisation du prétraitement EfficientNet")
        else:  # standard
            img_array = img_array.astype(np.float32) / 255.0
            logger.info("Utilisation du prétraitement standard (division par 255)")

        logger.info(
            f"Forme de l'array après normalisation: {img_array.shape}, type: {img_array.dtype}, min: {img_array.min()}, max: {img_array.max()}")

        # Sauvegarder l'image normalisée pour débogage
        if preprocessing_method == "standard":
            save_debug_image((img_array * 255).astype(np.uint8), "normalized")
        else:
            # Pour les prétraitements plus complexes, l'image peut ne pas être facilement visualisable
            # mais nous essayons quand même de la sauvegarder
            try:
                normalized_vis = (img_array - img_array.min()) / (img_array.max() - img_array.min())
                save_debug_image((normalized_vis * 255).astype(np.uint8), "normalized")
            except:
                logger.warning("Impossible de sauvegarder l'image normalisée pour visualisation")

        # Ajouter la dimension du batch
        img_array = np.expand_dims(img_array, axis=0)
        logger.info(f"Forme finale de l'image: {img_array.shape}")

        return img_array
    except Exception as e:
        logger.error(f"Erreur lors du prétraitement de l'image: {str(e)}")
        logger.error(traceback.format_exc())
        raise


@app.post("/predict")
async def predict(
        file: UploadFile = File(...),
        model_name: str = Query(None, description="Nom du modèle à utiliser")
) -> Dict[str, Any]:
    """
    Endpoint pour prédire la classe d'une image alimentaire.
    """
    logger.info(f"Requête de prédiction reçue pour le fichier: {file.filename}, type: {file.content_type}")
    logger.info(f"Modèle demandé: {model_name if model_name else 'par défaut'}")

    if not file.content_type.startswith("image/"):
        error_msg = f"Type de fichier non supporté: {file.content_type}"
        logger.error(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)

    # Charger le modèle spécifié ou utiliser le modèle actuel
    if model_name:
        try:
            load_model(model_name)
        except Exception as e:
            error_msg = f"Erreur lors du chargement du modèle '{model_name}': {str(e)}"
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)

    try:
        # Lire le contenu du fichier
        contents = await file.read()
        logger.info(f"Fichier lu, taille: {len(contents)} bytes")

        # Sauvegarder le fichier reçu pour débogage
        request_id = uuid.uuid4().hex[:8]
        debug_file = os.path.join(DEBUG_DIR, f"request_{request_id}_{file.filename}")
        with open(debug_file, "wb") as f:
            f.write(contents)
        logger.info(f"Fichier d'origine sauvegardé pour débogage: {debug_file}")

        # Ouvrir l'image
        image = Image.open(io.BytesIO(contents))
        logger.info(f"Image ouverte: {image.size} pixels, mode: {image.mode}")

        # Prétraiter l'image avec la configuration du modèle actuel
        processed_image = preprocess_image(image, current_model_name)

        # Faire la prédiction
        logger.info(f"Exécution de la prédiction avec le modèle '{current_model_name}'...")
        start_time = time.time()
        prediction = current_model.predict(processed_image)
        elapsed_time = time.time() - start_time
        logger.info(f"Prédiction terminée en {elapsed_time:.2f} secondes")

        # Vérifier si la prédiction est de la bonne forme
        if prediction.shape[1] != len(class_names):
            error_msg = f"La sortie du modèle a une forme inattendue: {prediction.shape}, attendu: {(1, len(class_names))}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Journaliser les détails de la prédiction pour le débogage
        logger.info(
            f"Statistiques de la prédiction - Min: {np.min(prediction)}, Max: {np.max(prediction)}, Moyenne: {np.mean(prediction)}, Écart-type: {np.std(prediction)}")

        # Journaliser les 5 meilleures prédictions
        top_indices = np.argsort(prediction[0])[-5:][::-1]
        for i, idx in enumerate(top_indices):
            logger.info(f"Top {i + 1}: {class_names[idx]} ({prediction[0][idx]:.4f})")

        # Obtenir l'indice de la classe prédite
        class_index = np.argmax(prediction[0])
        confidence = float(prediction[0][class_index])
        logger.info(f"Indice de classe prédit: {class_index}, confiance: {confidence:.4f}")

        # Vérifier que l'indice est valide
        if 0 <= class_index < len(class_names):
            class_name = class_names[class_index]
            logger.info(f"Classe prédite: {class_name}")

            # Générer un graphique de la distribution des prédictions pour le débogage
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(class_names)), prediction[0])
            plt.xlabel('Classes')
            plt.ylabel('Probabilités')
            plt.title(f'Distribution des prédictions (top: {class_name})')
            plt.tight_layout()
            plt.savefig(os.path.join(DEBUG_DIR, f"prediction_dist_{request_id}.png"))
            plt.close()

            class_name = class_names[class_index]
            nutrition_info = nutrition_data.get(class_name, {})

            return {
                "prediction": class_name,
                "confidence": confidence,
                "prediction_time_ms": elapsed_time * 1000,
                "model_used": current_model_name,
                "top_predictions": [
                    {"class": class_names[i], "confidence": float(prediction[0][i])}
                    for i in top_indices
                ],
                "nutrition": nutrition_info,
                "request_id": request_id
            }
        else:
            error_msg = f"Indice de classe invalide: {class_index}, max attendu: {len(class_names) - 1}"
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)

    except Exception as e:
        error_msg = f"Erreur lors du traitement: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Gestionnaire global d'exceptions pour l'API."""
    logger.error(f"Exception globale: {str(exc)}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "type": type(exc).__name__}
    )


if __name__ == "__main__":
    import uvicorn

    logger.info("Démarrage du serveur FastAPI...")
    uvicorn.run(app, host="0.0.0.0", port=5174)