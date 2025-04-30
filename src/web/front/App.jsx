import React, {useState, useEffect} from "react";
import axios from "axios";
import "./styles.css";

function App() {
    const [file, setFile] = useState(null);
    const [imagePreview, setImagePreview] = useState(null);
    const [prediction, setPrediction] = useState("");
    const [confidence, setConfidence] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");
    const [currentSection, setCurrentSection] = useState("home");
    const [progress, setProgress] = useState(0);
    const [nutritionInfo, setNutritionInfo] = useState(null);


    const [availableModels, setAvailableModels] = useState([]);
    const [selectedModel, setSelectedModel] = useState("EfficientNetV2B2");



    useEffect(() => {
        const fetchModels = async () => {
            try {
                const response = await axios.get("http://127.0.0.1:5174/models");
                if (response.data && response.data.models) {
                    setAvailableModels(response.data.models);

                    if (response.data.current_model) {
                        setSelectedModel(response.data.current_model);
                    }
                }
            } catch (error) {
                console.error("Error while retrieving models:", error);
                setError("Unable to retrieve the list of models. Check that the backend is running.");
            }
        };

        fetchModels();
    }, []);

    useEffect(() => {
        if (loading) {

            setProgress(0);


            const totalDuration = 350;
            const framesPerSecond = 60;
            const totalFrames = Math.ceil((totalDuration / 1000) * framesPerSecond);
            const incrementPerFrame = 100 / totalFrames;

            let frame = 0;

            const animate = () => {
                if (frame < totalFrames) {
                    frame++;
                    setProgress(Math.min(frame * incrementPerFrame, 100));
                    requestAnimationFrame(animate);
                }
            };

            requestAnimationFrame(animate);
        } else {

            setProgress(0);
        }
    }, [loading]);

    const handleFileChange = (e) => {
        const selectedFile = e.target.files[0];
        setFile(selectedFile);
        setError("");

        if (selectedFile) {
            const reader = new FileReader();
            reader.onloadend = () => {
                setImagePreview(reader.result);
            };
            reader.readAsDataURL(selectedFile);
        }
    };

    const handleModelChange = (modelName) => {
        setSelectedModel(modelName);

        setPrediction("");
        setConfidence(null);
    };

    const handleUpload = async () => {
        if (!file) {
            setError("Please select an image");
            return;
        }

        setLoading(true);
        setPrediction("");
        setNutritionInfo(null);
        setError("");

        const formData = new FormData();
        formData.append("file", file);

        try {
            console.log(`Sending the request to the API with the model ${selectedModel}...`);
            const response = await axios.post(`http://127.0.0.1:5174/predict?model_name=${selectedModel}`, formData, {
                headers: {"Content-Type": "multipart/form-data"},
            });

            console.log("Response received:", response.data);
            if (response.data && response.data.prediction) {
                setPrediction(response.data.prediction);
                setConfidence(response.data.confidence);

                if (response.data.nutrition) {
                    setNutritionInfo(response.data.nutrition);
                }
            } else {
                setError("Incorrect response format");
            }
        } catch (error) {
            console.error("Error during prediction:", error);

            if (error.response) {
                console.error("Response data:", error.response.data);
                console.error("Status:", error.response.status);
                setError(`Error ${error.response.status}: ${JSON.stringify(error.response.data)}`);
            } else if (error.request) {
                console.error("No response received");
                setError("The server is not responding. Check that the backend is running.");
            } else {
                console.error("Configuration error:", error.message);
                setError(`Error: ${error.message}`);
            }
        } finally {
            setLoading(false);
        }
    };

    const ProgressBar = ({progress}) => {
        return (
            <div className="progress-container">
                <progress className="progress-bar" max="100" value={progress}></progress>
                <img
                    src="src/web/assets/avocado.png"
                    alt="Avocado"
                    className="avocado"
                    style={{
                        left: `calc(${progress}% - 15px)`,
                        transform: 'rotate(15deg) translateY(-17px)',
                    }}
                />
            </div>
        );
    };

    const BananaIcon = ({style}) => (
        <svg width="60px" height="60px" viewBox="0 -11.18 437.91 437.91" style={style}>
            <path className="food-icon-path"
                  d="M437.35,213c-2.58-11.66-13-18.29-16.36-20.16-4.54-3.51-23.45-15.57-52.19.18a206.84,206.84,0,0,1-64.41,22.14c-33.63,5.48-65.54,1.46-94.85-12-54.45-24.93-99-100.9-117.67-161.63.36-3,.78-5.87,1.25-8.43,2.15-11.72-2-18.87-5.84-22.8C79,1.91,64.78-2,52.69,1,31.21,6.17,24.4,28.29,24.12,29.23a8.46,8.46,0,0,0,.68,6.48C39,61.82,48.63,95.09,39.8,101.08c-7.75,5.26-17.17,19.23-24.59,36.46A188.78,188.78,0,0,0,.11,204.92C-1.3,245.2,10.38,303.72,74,353.78c52.09,41,108.79,61.77,168.54,61.77h.37c48.58-.07,99.82-13.45,156.65-40.89,12.76-6.16,16.44-14.38,17.28-20.2,1.8-12.48-7.68-22.93-10.68-25.88a40.59,40.59,0,0,0-8.43-8.21c6.43-4.68,12.94-9.67,19.46-14.92,10.38-8.37,12.31-17,12.1-22.78-.45-12.23-10.35-20.74-13.58-23.2a30.19,30.19,0,0,0-2.26-2.13c5.65-6.3,11.32-13,16.95-20C438.38,227.31,438.58,218.6,437.35,213Z"/>
        </svg>
    );

    const AvocadoIcon = ({style}) => (
        <svg width="60px" height="60px" viewBox="-92.47 0 507.19 507.19" style={style}>
            <path className="food-icon-path"
                  d="M302.93,269.49,247.87,111.83a88,88,0,0,0-82.72-73.07C167.73,18.1,179,6.06,179.1,5.93a3.5,3.5,0,0,0-5-4.86c-.6.63-13.35,14.14-16,37.66a88,88,0,0,0-83.72,73.1L19.32,269.49A161.37,161.37,0,0,0,0,346.06c0,88.85,72.28,161.13,161.13,161.13s161.13-72.28,161.13-161.13A161.27,161.27,0,0,0,302.93,269.49Z"/>
            <path className="food-icon-path"
                  d="M161.13,264.43a88.65,88.65,0,1,0,88.64,88.64A88.74,88.74,0,0,0,161.13,264.43Z"/>
        </svg>
    );

    const BurgerIcon = ({style}) => (
        <svg width="60px" height="60px" viewBox="-7.53 0 455.26 455.26" style={style}>
            <path className="food-icon-path"
                  d="M408.16,378.33H401c7.47-7.78,14-17.26,14-32.52a52.84,52.84,0,0,0-.69-8.55c16.23-1.78,25.82-18.44,25.82-34.59,0-18.18-11.81-30.87-28.71-30.87h-6.15V253.53A37.2,37.2,0,0,0,420.19,243a36.58,36.58,0,0,0,8.28-29.62C418,149.13,377.58,92.61,320.15,61.63L326.56,44,392,68.49a3.49,3.49,0,0,0,4.4-4.74L386.29,41.83l21.74-10a3.5,3.5,0,0,0-.27-6.46L338.78.21a3.51,3.51,0,0,0-2.68.12,3.45,3.45,0,0,0-1.8,2l-13.1,36v0l-7.31,20.07a211.18,211.18,0,0,0-302.15,155A36.55,36.55,0,0,0,20,243a37.17,37.17,0,0,0,14.84,10.57V271.8H32.11c-13.36,0-22.73,4.32-27.86,12.85C0,291.72,0,299.94,0,304.85c0,18.06,9.55,29.67,25.89,32.1a53.33,53.33,0,0,0-.75,8.86,47.85,47.85,0,0,0,12.38,32.52H32A23.1,23.1,0,0,0,9,401.4v7.43a46.49,46.49,0,0,0,46.44,46.43H384.8a46.48,46.48,0,0,0,46.43-46.43V401.4A23.1,23.1,0,0,0,408.16,378.33Z"/>
        </svg>
    );


    const NutritionInfo = ({nutrition}) => {
        if (!nutrition) return null;

        return (
            <div className="nutrition-container">
                <h3 className="nutrition-title">Nutritional information</h3>
                <div className="nutrition-grid">
                    <div className="nutrition-item">
                        <div className="nutrition-value">{nutrition.calories || "N/A"}</div>
                        <div className="nutrition-label">Calories</div>
                    </div>
                    <div className="nutrition-item">
                        <div className="nutrition-value">{nutrition.fat || "N/A"}g</div>
                        <div className="nutrition-label">Fat</div>
                    </div>
                    <div className="nutrition-item">
                        <div className="nutrition-value">{nutrition.carbs || "N/A"}g</div>
                        <div className="nutrition-label">Carbs</div>
                    </div>
                    <div className="nutrition-item">
                        <div className="nutrition-value">{nutrition.protein || "N/A"}g</div>
                        <div className="nutrition-label">Protein</div>
                    </div>
                </div>
            </div>
        );
    };

    const generateFoodBackground = () => {
        const icons = [];
        const foodTypes = ['banana', 'avocado', 'burger'];
        const gridSize = {cols: 6, rows: 6};
        const totalCells = gridSize.cols * gridSize.rows;

        for (let i = 0; i < totalCells; i++) {
            const row = Math.floor(i / gridSize.cols);
            const col = i % gridSize.cols;

            const foodTypeIndex = i % foodTypes.length;
            const foodType = foodTypes[foodTypeIndex];


            const rotation = (i * 30) % 360;


            const style = {
                gridRow: `${row + 1}`,
                gridColumn: `${col + 1}`,
                transform: `rotate(${rotation}deg)`,
                padding: '10px',
                opacity: 0.3,
            };

            let icon;
            if (foodType === 'banana') {
                icon = <BananaIcon key={`food-${i}`} style={style}/>;
            } else if (foodType === 'avocado') {
                icon = <AvocadoIcon key={`food-${i}`} style={style}/>;
            } else {
                icon = <BurgerIcon key={`food-${i}`} style={style}/>;
            }

            icons.push(icon);
        }

        return icons;
    };

    const TitleBackgroundSVG = () => (
        <svg width="617" height="441" viewBox="0 0 517 341" className="title-background-svg" fill="none"
             xmlns="http://www.w3.org/2000/svg">
            <path
                d="M19.4488 243.778C48.9488 261.778 186.949 280.778 179.949 296.778C172.949 312.778 143.949 345.278 191.949 337.778C239.949 330.278 310.949 363.278 332.449 308.778C353.949 254.278 519.449 325.278 458.949 266.778C398.449 208.278 478.449 225.778 507.449 171.278C536.449 116.778 489.449 -0.221638 410.949 55.2784C332.449 110.778 193.949 -66.7216 186.949 28.7784C179.949 124.278 105.949 66.7784 47.9488 88.7784C-10.0512 110.778 -10.0512 225.778 19.4488 243.778Z"
                fill="#B0CFB5" fillOpacity="0.42"/>
        </svg>
    );


    const goToClassifier = () => {
        setCurrentSection("classifier");
        document.getElementById("classifier-section").scrollIntoView({behavior: "smooth"});
    };


    const renderHomeSection = () => (
        <div className="home-section">
            <div className="title-container">
                <TitleBackgroundSVG className="title-background-svg"/>
                <h1 className="main-title">F<span className="accent-letter">OO</span>D101</h1>
                <p className="subtitle">Find out what you eat in just one click</p>
            </div>

            <div className="infinite-scroll-container">
                <div className="scroll-content">
                    <div className="category-card">
                        <div className="category-image-container">
                            <img src="/src/web/assets/pad_thai.png" alt="Pad thai" className="category-image"/>
                        </div>
                        <h3 className="category-title">Pad Thaï</h3>
                        <p className="category-description">An energy-packed Thai dish combining fried noodles, protein and peanuts.</p>
                    </div>

                    <div className="category-card">
                        <div className="category-image-container">
                            <img src="/src/web/assets/oysters.png" alt="Oyesters" className="category-image"/>
                        </div>
                        <h3 className="category-title">Oysters</h3>
                        <p className="category-description">Seafood rich in zinc, omega-3 and protein, renowned for its immune benefits and iodine texture.</p>
                    </div>

                    <div className="category-card">
                        <div className="category-image-container">
                            <img src="/src/web/assets/sushis.png" alt="Sushis" className="category-image"/>
                        </div>
                        <h3 className="category-title">Sushis</h3>
                        <p className="category-description">Balanced Japanese bites, combining vinegared rice, raw fish and seaweed, rich in omega-3 and protein.</p>
                    </div>

                    <div className="category-card">
                        <div className="category-image-container">
                            <img src="/src/web/assets/beef_tartare.png" alt="Beef tartare" className="category-image"/>
                        </div>
                        <h3 className="category-title">Beef Tartare</h3>
                        <p className="category-description">A raw dish rich in protein and iron, appreciated for its tenderness and spicy seasonings.</p>
                    </div>

                    <div className="category-card">
                        <div className="category-image-container">
                            <img src="/src/web/assets/cupcakes.png" alt="Cupcakes" className="category-image"/>
                        </div>
                        <h3 className="category-title">Cupcakes</h3>
                        <p className="category-description">Soft little cakes, rich in sugars and fats, recognisable by their colourful, decorative icing.</p>
                    </div>

                    <div className="category-card">
                        <div className="category-image-container">
                            <img src="/src/web/assets/waffles.png" alt="Waffles" className="category-image"/>
                        </div>
                        <h3 className="category-title">Waffles</h3>
                        <p className="category-description">Crunchy, moist waffles, rich in carbohydrates, often topped with sugar, fruit or chocolate.</p>
                    </div>

                    <div className="category-card">
                        <div className="category-image-container">
                            <img src="/src/web/assets/paella.png" alt="Paella" className="category-image"/>
                        </div>
                        <h3 className="category-title">Paëlla</h3>
                        <p className="category-description">A Spanish dish rich in rice, proteins (seafood, chicken) and vegetables, with a tasty, colourful flavour.</p>
                    </div>

                    <div className="category-card">
                        <div className="category-image-container">
                            <img src="/src/web/assets/pad_thai.png" alt="Pad thai" className="category-image"/>
                        </div>
                        <h3 className="category-title">Pad Thaï</h3>
                        <p className="category-description">An energy-packed Thai dish combining fried noodles, protein and peanuts.</p>
                    </div>

                    <div className="category-card">
                        <div className="category-image-container">
                            <img src="/src/web/assets/oysters.png" alt="Oysters" className="category-image"/>
                        </div>
                        <h3 className="category-title">Oysters</h3>
                        <p className="category-description">Seafood rich in zinc, omega-3 and protein, renowned for its immune benefits and iodine texture.</p>
                    </div>

                    <div className="category-card">
                        <div className="category-image-container">
                            <img src="/src/web/assets/sushis.png" alt="Sushis" className="category-image"/>
                        </div>
                        <h3 className="category-title">Sushis</h3>
                        <p className="category-description">Balanced Japanese bites, combining vinegared rice, raw fish and seaweed, rich in omega-3 and protein.</p>
                    </div>

                    <div className="category-card">
                        <div className="category-image-container">
                            <img src="/src/web/assets/beef_tartare.png" alt="Beef tartare" className="category-image"/>
                        </div>
                        <h3 className="category-title">Beef Tartare</h3>
                        <p className="category-description">A raw dish rich in protein and iron, appreciated for its tenderness and spicy seasonings.</p>
                    </div>

                    <div className="category-card">
                        <div className="category-image-container">
                            <img src="/src/web/assets/cupcakes.png" alt="Cupcakes" className="category-image"/>
                        </div>
                        <h3 className="category-title">Cupcakes</h3>
                        <p className="category-description">Soft little cakes, rich in sugars and fats, recognisable by their colourful, decorative icing.</p>
                    </div>

                    <div className="category-card">
                        <div className="category-image-container">
                            <img src="/src/web/assets/waffles.png" alt="Waffles" className="category-image"/>
                        </div>
                        <h3 className="category-title">Waffles</h3>
                        <p className="category-description">Crunchy, moist waffles, rich in carbohydrates, often topped with sugar, fruit or chocolate.</p>
                    </div>

                    <div className="category-card">
                        <div className="category-image-container">
                            <img src="/src/web/assets/paella.png" alt="Paella" className="category-image"/>
                        </div>
                        <h3 className="category-title">Paëlla</h3>
                        <p className="category-description">A Spanish dish rich in rice, proteins (seafood, chicken) and vegetables, with a tasty, colourful flavour.</p>
                    </div>
                </div>
            </div>

            <button className="start-button" onClick={goToClassifier}>
                Start to classify
            </button>
        </div>
    );


    const renderClassifierSection = () => {
        const handleDrop = (e) => {
            e.preventDefault();
            e.stopPropagation();

            if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                const droppedFile = e.dataTransfer.files[0];
                setFile(droppedFile);
                setError("");

                if (droppedFile) {
                    const reader = new FileReader();
                    reader.onloadend = () => {
                        setImagePreview(reader.result);
                    };
                    reader.readAsDataURL(droppedFile);
                }
            }
        };

        const handleDragOver = (e) => {
            e.preventDefault();
            e.stopPropagation();
        };

        const handleClick = () => {
            document.getElementById('file-input').click();
        };

        return (
            <div id="classifier-section" className="classifier-section">
                <h2 className="section-title">Classify your image</h2>
                <p className="section-description">Upload an image of a food to classify it.</p>

                <div className="classifier-content">
                    <div className="classifier-left-column">
                        <div className="model-selection-container">
                            <h3 className="model-selection-title">Select a model</h3>
                            <div className="model-buttons">
                                {availableModels.map((model) => (
                                    <button
                                        key={model}
                                        className={`model-button ${selectedModel === model ? 'selected' : ''}`}
                                        onClick={() => handleModelChange(model)}
                                    >
                                        {model}
                                    </button>
                                ))}
                            </div>
                            <p className="model-description">
                                {selectedModel === "EfficientNetV2B2" && "EfficientNetV2B2 - High-performance, high-precision model"}
                                {selectedModel === "CNN" && "CNN - Classical convolutional neural network model"}
                                {selectedModel === "InceptionV3" && "InceptionV3 - Deep model with advanced architecture"}
                            </p>

                            <div
                                className="dropzone"
                                onDrop={handleDrop}
                                onDragOver={handleDragOver}
                                onClick={handleClick}
                            >
                                <input
                                    id="file-input"
                                    type="file"
                                    accept="image/*"
                                    onChange={handleFileChange}
                                    className="file-input-hidden"
                                />
                                <div className="dropzone-content">
                                    <svg className="upload-icon" width="48" height="48" viewBox="0 0 24 24" fill="none"
                                         xmlns="http://www.w3.org/2000/svg">
                                        <path d="M12 16L12 8" stroke="#2c7d43" strokeWidth="2" strokeLinecap="round"
                                              strokeLinejoin="round"/>
                                        <path d="M9 11L12 8 15 11" stroke="#2c7d43" strokeWidth="2"
                                              strokeLinecap="round" strokeLinejoin="round"/>
                                        <path
                                            d="M20 16.7428C21.2215 15.734 22 14.2079 22 12.5C22 9.46243 19.5376 7 16.5 7C16.2815 7 16.0771 7.01241 15.8767 7.03507C15.0326 4.63805 12.7867 3 10.1818 3C6.63636 3 3.77273 5.9467 3.77273 9.5C3.77273 11.1631 4.30568 12.7087 5.22727 13.9263C5.81254 14.7906 6.60954 15.4611 7.52381 15.8239"
                                            stroke="#2c7d43" strokeWidth="2" strokeLinecap="round"
                                            strokeLinejoin="round"/>
                                    </svg>
                                    <p>Drag your image here or click to download</p>
                                </div>
                            </div>

                            <button
                                onClick={handleUpload}
                                disabled={loading || !file}
                                className="predict-button"
                            >
                                {loading ? "Prediction in progress..." : "Predict"}
                            </button>
                        </div>
                    </div>


                    <div className="classifier-right-column">
                        <div className="preview-results-container">
                            {imagePreview ? (
                                <div className="image-preview-container">
                                    <h3 className="preview-title">Image preview</h3>
                                    <div className="preview-image-wrapper">
                                        <img
                                            src={imagePreview}
                                            alt="Preview"
                                            className="preview-image"
                                        />
                                    </div>
                                </div>
                            ) : (
                                <div className="no-preview-placeholder">
                                    <svg width="64" height="64" viewBox="0 0 24 24" fill="none"
                                         xmlns="http://www.w3.org/2000/svg">
                                        <rect x="3" y="3" width="18" height="18" rx="2" stroke="#2c7d43"
                                              strokeWidth="2"/>
                                        <circle cx="8.5" cy="8.5" r="1.5" fill="#2c7d43"/>
                                        <path d="M3 15L7 13L10 15L15 11L21 15" stroke="#2c7d43" strokeWidth="2"
                                              strokeLinecap="round" strokeLinejoin="round"/>
                                    </svg>
                                    <p>No image selected</p>
                                </div>
                            )}

                            {loading && (
                                <ProgressBar progress={progress}/>
                            )}

                            {error && (
                                <div className="error-container">
                                    <h3 className="error-title">Error :</h3>
                                    <p>{error}</p>
                                </div>
                            )}

                            {prediction && (
                                <div className="result-container">
                                    <h2 className="result-title">Result : {prediction}</h2>
                                    <p className="result-text">
                                        The model <strong>{selectedModel}</strong> has identified your image as: <strong>{prediction.replace(/_/g, ' ')}.</strong>
                                    </p>
                                    <p className="result-confidence">Accuracy : {(confidence * 100).toFixed(2)}%</p>
                                    <NutritionInfo nutrition={nutritionInfo}/>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </div>
        );
    };

    return (
        <div className="app-container">

            <div className="food-background">
                {generateFoodBackground()}
            </div>


            <div className="app-content">
                {currentSection === "home" ? renderHomeSection() : null}
                {renderClassifierSection()}
            </div>
        </div>
    );
}

export default App;