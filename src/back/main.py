import tensorflow as tf

print(f"TensorFlow version: {tf.__version__}")

try:
    from tensorflow import keras

    print("TensorFlow Keras import successful")
except ImportError as e:
    print(f"TensorFlow Keras import error:  {e}")
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


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api_debug.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


DEBUG_DIR = "debug_images"
os.makedirs(DEBUG_DIR, exist_ok=True)


MODEL_CONFIGS = {
    "EfficientNetV2B2": {
        "file": "food101_EfficientNetV2B2_model_whole.h5",
        "input_size": (260, 260),
        "preprocessing": "EfficientNetV2B2",
    },
    "CNN": {
        "file": "cnn_food101_allclasses__final.h5",
        "input_size": (224, 224),
        "preprocessing": "standard",
    },
    "InceptionV3": {
        "file": "v3_food101_Inception_model_finetuning_wholeds_.h5",
        "input_size": (299, 299),
        "preprocessing": "standard",
    }
}


current_model = None
current_model_name = None


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
    """Loads nutritional data from the JSON file"""
    try:
        nutrition_file = Path("nutrition.json")
        if not nutrition_file.exists():
            logger.warning(f"Nutrition file not found: {nutrition_file}")
            return {}

        with open(nutrition_file, "r") as f:
            nutrition_data = json.load(f)

        logger.info(f"Nutritional data loaded for {len(nutrition_data)} meals")
        return nutrition_data
    except Exception as e:
        logger.error(f"Error loading nutritional data: {str(e)}")
        return {}

nutrition_data = load_nutrition_data()

def load_model(model_name: str):
    """Load the specified model"""
    global current_model, current_model_name

    if current_model_name == model_name:
        logger.info(f"Model '{model_name}' already loaded")
        return current_model

    if model_name not in MODEL_CONFIGS:
        error_msg = f"Model '{model_name}' not found in configuration"
        logger.error(error_msg)
        raise ValueError(error_msg)

    model_config = MODEL_CONFIGS[model_name]
    model_path = model_config["file"]


    if not os.path.exists(model_path):
        error_msg = f"The model file does not exist: {model_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)


    try:
        logger.info(f"Loading the model '{model_name}' from: {model_path}")
        model = tf.keras.models.load_model(model_path)
        logger.info(f"Model '{model_name}' loaded with success")


        input_shape = model.input_shape
        output_shape = model.output_shape
        logger.info(f"Model input form: {input_shape}")
        logger.info(f"Model output form: {output_shape}")


        current_model = model
        current_model_name = model_name

        return model
    except Exception as e:
        logger.error(f"Error when loading the model '{model_name}': {str(e)}")
        logger.error(traceback.format_exc())
        raise



try:
    default_model = "EfficientNetV2B2"
    load_model(default_model)
except Exception as e:
    logger.error(f"Error loading the default model: {str(e)}")

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
    """Returns the list of available models"""
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
    """Endpoint to test the model with a random test image"""
    if model_name:
        try:
            load_model(model_name)
        except Exception as e:
            return {"test_successful": False, "error": str(e)}

    try:

        input_size = MODEL_CONFIGS[current_model_name]["input_size"]
        test_img = np.random.rand(input_size[0], input_size[1], 3)
        test_img_batch = np.expand_dims(test_img, axis=0)


        start_time = time.time()
        pred = current_model.predict(test_img_batch)
        elapsed_time = time.time() - start_time


        pred_index = np.argmax(pred[0])
        pred_value = float(pred[0][pred_index])


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
        logger.error(f"Error when testing the model: {str(e)}")
        logger.error(traceback.format_exc())
        return {"test_successful": False, "error": str(e)}


def save_debug_image(image, stage="original", suffix=""):
    """Saves the image for debugging purposes"""
    try:
        if isinstance(image, np.ndarray):

            if image.ndim == 3 and image.shape[2] == 3:

                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                debug_img = Image.fromarray(image)
            else:
                logger.warning(f"Unexpected image format: {image.shape}")
                return
        else:
            debug_img = image


        filename = f"{stage}_{uuid.uuid4().hex[:8]}{suffix}.png"
        filepath = os.path.join(DEBUG_DIR, filename)


        debug_img.save(filepath)
        logger.info(f"Debug image saved: {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Error saving the debug image: {str(e)}")
        return None


def preprocess_image(image: Image.Image, model_name: str) -> np.ndarray:
    """
    Pre-processes an image for prediction using the specified model.
    """
    try:
        model_config = MODEL_CONFIGS[model_name]
        target_size = model_config["input_size"]
        preprocessing_method = model_config["preprocessing"]


        save_debug_image(image, "original")


        if image.mode != "RGB":
            logger.info(f"Converting the image from mode {image.mode} to RGB")
            image = image.convert("RGB")
            save_debug_image(image, "converted_rgb")


        logger.info(f"Resize image from {image.size} to {target_size}")
        image = image.resize(target_size)
        save_debug_image(image, "resized")


        img_array = np.array(image)
        logger.info(
            f"Shape of the array before normalisation: {img_array.shape}, type: {img_array.dtype}, min: {img_array.min()}, max: {img_array.max()}")


        if preprocessing_method == "EfficientNetV2B2":
            img_array = preprocess_input(img_array)
            logger.info("Using EfficientNet pre-processing")
        else:
            img_array = img_array.astype(np.float32) / 255.0
            logger.info("Use of standard pre-processing (division by 255)")

        logger.info(
            f"Shape of the array after normalisation: {img_array.shape}, type: {img_array.dtype}, min: {img_array.min()}, max: {img_array.max()}")


        if preprocessing_method == "standard":
            save_debug_image((img_array * 255).astype(np.uint8), "normalized")
        else:

            try:
                normalized_vis = (img_array - img_array.min()) / (img_array.max() - img_array.min())
                save_debug_image((normalized_vis * 255).astype(np.uint8), "normalized")
            except:
                logger.warning("Unable to save normalized image for viewing")


        img_array = np.expand_dims(img_array, axis=0)
        logger.info(f"Final image shape: {img_array.shape}")

        return img_array
    except Exception as e:
        logger.error(f"Error during image pre-processing: {str(e)}")
        logger.error(traceback.format_exc())
        raise


@app.post("/predict")
async def predict(
        file: UploadFile = File(...),
        model_name: str = Query(None, description="Name of model to be used")
) -> Dict[str, Any]:
    """
    Endpoint to predict the class of a food image.
    """
    logger.info(f"Prediction request received for the file: {file.filename}, type: {file.content_type}")
    logger.info(f"Model asked: {model_name if model_name else 'by default'}")

    if not file.content_type.startswith("image/"):
        error_msg = f"File type not supported: {file.content_type}"
        logger.error(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)


    if model_name:
        try:
            load_model(model_name)
        except Exception as e:
            error_msg = f"Error when loading the model '{model_name}': {str(e)}"
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)

    try:

        contents = await file.read()
        logger.info(f"File read, size: {len(contents)} bytes")


        request_id = uuid.uuid4().hex[:8]
        debug_file = os.path.join(DEBUG_DIR, f"request_{request_id}_{file.filename}")
        with open(debug_file, "wb") as f:
            f.write(contents)
        logger.info(f"Original file saved for debugging purposes: {debug_file}")


        image = Image.open(io.BytesIO(contents))
        logger.info(f"Open image: {image.size} pixels, mode: {image.mode}")


        processed_image = preprocess_image(image, current_model_name)


        logger.info(f"Performing prediction with the model '{current_model_name}'...")
        start_time = time.time()
        prediction = current_model.predict(processed_image)
        elapsed_time = time.time() - start_time
        logger.info(f"Prediction completed in {elapsed_time:.2f} seconds")


        if prediction.shape[1] != len(class_names):
            error_msg = f"The model's output takes an unexpected form: {prediction.shape}, expected: {(1, len(class_names))}"
            logger.error(error_msg)
            raise ValueError(error_msg)


        logger.info(
            f"Prediction statistics - Min: {np.min(prediction)}, Max: {np.max(prediction)}, Mean: {np.mean(prediction)}, Standard deviation: {np.std(prediction)}")


        top_indices = np.argsort(prediction[0])[-5:][::-1]
        for i, idx in enumerate(top_indices):
            logger.info(f"Top {i + 1}: {class_names[idx]} ({prediction[0][idx]:.4f})")


        class_index = np.argmax(prediction[0])
        confidence = float(prediction[0][class_index])
        logger.info(f"Predicted class index: {class_index}, accuracy: {confidence:.4f}")


        if 0 <= class_index < len(class_names):
            class_name = class_names[class_index]
            logger.info(f"Predicted class: {class_name}")


            plt.figure(figsize=(10, 6))
            plt.bar(range(len(class_names)), prediction[0])
            plt.xlabel('Classes')
            plt.ylabel('Probabilities')
            plt.title(f'Distribution of predictions (top: {class_name})')
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
            error_msg = f"Invalid class index: {class_index}, max expected: {len(class_names) - 1}"
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)

    except Exception as e:
        error_msg = f"Processing error: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for the API."""
    logger.error(f"Global exception: {str(exc)}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "type": type(exc).__name__}
    )


if __name__ == "__main__":
    import uvicorn

    logger.info("Launching the FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=5174)