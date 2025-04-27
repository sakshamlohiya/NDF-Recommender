Neural Collaborative Filtering Recommender System

A deep learning-based recommendation system built using TensorFlow/Keras, powered by Neural Collaborative Filtering (NCF).
It predicts top recommended items for users based on historical interaction data.
🚀 Project Overview

This project implements a Neural Collaborative Filtering (NCF) model to predict user-item interactions and recommend the top items for a given user.

Built with:

TensorFlow/Keras for the NCF model
FastAPI for serving recommendations via API
Docker for containerized deployment
📂 Project Structure

ncf_recommender/
├── app.py                 # FastAPI API server
├── train_ncf.py            # (optional) Training script
├── ncf_model.h5            # Trained NCF model
├── user2user_encoded.pkl   # User encoder
├── item2item_encoded.pkl   # Item encoder
├── item_encoder_to_id.pkl  # Mapping for item decoding
├── Dockerfile              # Container setup
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
📈 Dataset

We used the MovieLens 100k Dataset
(you can replace it with any user-item interaction dataset).

Each row = (user_id, item_id, rating)
Only positive interactions (rating > threshold) were considered.
🛠 How to Run

1. Clone the Repository
git clone https://github.com/yourusername/ncf-recommender.git
cd ncf-recommender
2. Install Dependencies
pip install -r requirements.txt
3. Train the Model (if needed)
python train_ncf.py
(Saves ncf_model.h5 and encoder mappings.)

OR use the already trained model provided.

4. Start the FastAPI Server
uvicorn app:app --reload --host 0.0.0.0 --port 8000
Visit: http://localhost:8000/docs to test the API!

📬 API Endpoint

POST /recommend
Request Body:

{
  "user_id": 10,
  "top_k": 5
}
Response:

{
  "recommended_items": [50, 120, 85, 172, 350]
}
🐳 Docker Deployment

Build Docker Image:

docker build -t ncf-recommender .
Run Container:

docker run -p 8000:8000 ncf-recommender
📈 Model Architecture

The NCF model combines:

Embedding layers for users and items
Multiple Dense (MLP) layers to learn complex interactions
Sigmoid activation for final prediction score
📋 Features

✅ Neural Collaborative Filtering (Pure NCF)
✅ TensorFlow/Keras Implementation
✅ FastAPI-based Inference Server
✅ Docker-Ready Deployment
✅ Easy to extend with new datasets

✨ Future Improvements

Add regular retraining pipeline (auto-retrain with fresh data)
Monitoring (model drift detection)
Full CI/CD deployment (GitHub Actions + AWS/GCP)
🤝 Contributing

Pull requests and feedback are welcome!
Feel free to suggest features or improvements!

📝 License

This project is open source under the MIT License.
