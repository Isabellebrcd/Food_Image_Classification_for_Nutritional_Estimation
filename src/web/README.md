# FOOD101 - Web Application

Web Application based on Food101 Dataset. Used to classify food images using deep learning models.

## Installation and Setup

### Prerequisites

- Python 3.11 for the backend
- Node.js and npm for the frontend
- TensorFlow 2.19.0
- Pre-trained models (h5 files)

### Backend Installation

```bash
# Clone the repository
git clone [link]
cd [file]

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies from requirements.txt
pip install -r requirements.txt

# for mac
pip uninstall tensorflow
pip install tensorflow==2.16.2

# for windows
pip install tensorflow==2.19

# Start the backend server
cd back
python main.py
```

Open your browser and go to <http://127.0.0.1:5174/>

### Frontend Installation

```bash
# Navigate to the frontend directory
cd front

# Install dependencies using package.json
npm install

# Start the development server
npm run dev
```

Open your browser and go to <http://localhost:5173>

## Project Structure

```bash
food101-classifier/
├── src/
│   ├── assets/
│   ├── back/
│   │   ├── models/
│   │   ├── main.py
│   │   ├── nutrition.json
│   │   └── requirements.txt
│   │
│   ├── front/
│   │   ├── App.jsx
│   │   ├── index.jsx
│   │   └── styles.css
│
├── index.html
├── vite.config.js
└── package.json


```
