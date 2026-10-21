# Course Recommendation System

This project is a Django-based course recommendation system that utilizes a hybrid approach, Content-Based Filtering (using TF-IDF on course descriptions) and Collaborative Filtering (using FunkSVD on user interactions).

## Features
- **Hybrid Recommendation Engine**: Combines content and collaborative scores for personalized suggestions.
- **Content-Based Filtering**: Analyzes course titles, descriptions, and concepts.
- **Collaborative Filtering**: Learns latent factors from user-course interactions.
- **Evaluation Scripts**: Tools to measure RMSE, MAE, Precision, Recall and F1 score of the models.

## Prerequisites
- Python 3.8+
- Data files in `data/` directory:
    - `courses.csv`: Course metadata.
    - `studentVle.csv`: Student interactions (VLE log).

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd p2
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Database Setup

1.  **Apply migrations:**
    ```bash
    python manage.py makemigrations
    python manage.py migrate
    ```

2.  **Load Data:**
    This script ingests real data from the `data/` directory.
    ```bash
    python scripts/load_real_data.py
    ```
    *Note: Ensure `courses.csv` and `studentVle.csv` are present in the `data/` folder.*
    *If the `data` folder is empty, pull the data using DVC:*
    ```bash
    dvc pull
    ```

## Usage

1.  **Run the development server:**
    ```bash
    python manage.py runserver
    ```

2.  **Access the application:**
    Open your browser and navigate to `http://127.0.0.1:8000/`.

## Evaluation

To evaluate the performance of the recommendation models:
```bash
python scripts/evaluate_models.py
```