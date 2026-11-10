# Article Recommendation System

A content-based article recommendation system built with Python and machine learning. The project analyzes article content and users' implicit interactions to create personalized article recommendations, then exposes the trained recommender through a lightweight Flask REST API.

> **Project type:** Graduation project  
> **Primary approach:** Content-Based Filtering  
> **Dataset:** CI&T DeskDrop article-sharing and user-interaction data  
> **Core technologies:** Python, Pandas, NumPy, SciPy, scikit-learn, NLTK, Flask

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Recommendation Method](#recommendation-method)
- [Model Evaluation](#model-evaluation)
- [REST API](#rest-api)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Running the API](#running-the-api)
- [Example Response](#example-response)
- [Model Artifacts](#model-artifacts)
- [Limitations](#limitations)
- [Possible Improvements](#possible-improvements)
- [Technologies](#technologies)

---

## Overview

The system recommends articles to users based on the textual content of articles they have previously interacted with.

It uses a **content-based recommendation approach**:

1. Article titles and body text are converted into numerical feature vectors using TF-IDF.
2. User interactions are assigned different strengths according to the type of action.
3. A weighted user profile is built from the TF-IDF vectors of articles the user interacted with.
4. Cosine similarity is calculated between the user's profile and the article vectors.
5. Articles are ranked by similarity, and articles the user has already interacted with can be excluded from the recommendations.

The trained artifacts are loaded by a Python recommender class. A Flask endpoint accepts a user ID and returns that user's ranked recommendations.

## Problem Statement

Content-sharing platforms may contain thousands of articles. Users need a way to discover relevant content without manually searching through the entire collection.

The goal of this project is to use the available article text and historical user interactions to estimate each user's interests and recommend articles with similar content.

The project uses **implicit feedback** rather than explicit star ratings. Actions such as viewing, liking, bookmarking, following, and commenting are treated as signals of different strengths.

## Dataset

The project uses the **DeskDrop dataset**, a real-world sample from CI&T's internal communication platform. DeskDrop enables employees to share articles and interact with content.

The dataset covers approximately 12 months, from March 2016 to February 2017. It contains two main CSV files:

| File | Description |
|---|---|
| `shared_articles.csv` | Article metadata, including content ID, title, URL, text, language, sharing event, and related metadata |
| `users_interactions.csv` | User activity records, including user ID, article ID, timestamp, and interaction type |

### Dataset statistics observed in the notebook

| Measure | Value |
|---|---:|
| Article records in the original article table | 3,122 |
| Shared articles retained for recommendation | 3,047 |
| Removed-content records | 75 |
| User interaction records | 72,312 |
| Distinct users in the interaction data | 1,895 |
| Distinct article IDs in the interaction data | Dataset-dependent; recommendations use the 3,047 shared article IDs |
| Users retained after the minimum-activity filter | 1,140 |
| Interaction records from retained users | 69,868 |
| Aggregated user–article pairs after weighting | 39,106 |

The notebook also reports that article text is available in English and Portuguese. The TF-IDF preprocessing uses stop-word lists for both languages.

### Interaction distribution

The original interaction data contains the following event counts:

| Interaction type | Count | Strength used by the model |
|---|---:|---:|
| `VIEW` | 61,086 | 1.0 |
| `LIKE` | 5,745 | 2.0 |
| `BOOKMARK` | 2,463 | 3.0 |
| `COMMENT CREATED` | 1,611 | 5.0 |
| `FOLLOW` | 1,407 | 4.0 |
| **Total** | **72,312** | |

The event strengths encode the assumption that some actions indicate stronger interest than others. A comment, for example, receives a higher weight than a simple view.

> **Dataset note:** The raw CSV files are not included in this repository snapshot. Obtain the DeskDrop dataset separately and place the two files in the expected data directory before rerunning the notebook.

## Project Workflow

### 1. Exploratory Data Analysis

The notebook explores the article and interaction tables before model construction.

The article analysis includes:

- Inspecting the article records and their columns.
- Counting shared and removed content.
- Keeping records with the `CONTENT SHARED` event.
- Examining the article-sharing period.
- Counting distinct shared articles and article sharers.

The interaction analysis includes:

- Inspecting interaction records and event types.
- Counting total interactions and distinct users.
- Examining interaction activity over time.
- Reviewing the number of articles each user interacted with.
- Exploring available contextual information, including country and user activity.

The notebook records 3,047 shared articles and 72,312 user interaction events.

### 2. Interaction Filtering

Users with fewer than five distinct interacted articles are excluded from model training and evaluation.

This reduces the interaction data from:

- 1,895 distinct users to 1,140 selected users.
- 72,312 total interaction records to 69,868 records from selected users.

The filter is based on the number of distinct articles a user interacted with, rather than the raw number of event rows.

### 3. Implicit Feedback Weighting

Each interaction type is mapped to a numerical strength:

```python
event_type_strength = {
    "VIEW": 1.0,
    "LIKE": 2.0,
    "BOOKMARK": 3.0,
    "FOLLOW": 4.0,
    "COMMENT CREATED": 5.0,
}
```

A user may interact with the same article multiple times and may perform different actions on it. The project aggregates these events for each user–article pair by summing their interaction strengths.

To reduce the effect of very large accumulated values, a base-2 logarithmic transformation is applied:

$$
s_{u,i} = \log_2(1 + \sum_{e \in E_{u,i}} w(e))
$$

Where:

- $(u\)$ is a user.
- $(i\)$ is an article.
- $(E_{u,i}\)$ is the set of recorded events for that user and article.
- $(w(e)\)$ is the strength assigned to an event.
- $(s_{u,i}\)$ is the resulting aggregated interaction strength.

After aggregation, the notebook contains **39,106 unique user–article interaction pairs**.

### 4. Article Text Processing and TF-IDF

The model represents each article using its title and body text.

The notebook uses `TfidfVectorizer` with the following configuration:

| Parameter | Value |
|---|---|
| Analyzer | Word |
| N-gram range | `(1, 2)` |
| Minimum document frequency | `0.003` |
| Maximum document frequency | `0.5` |
| Maximum features | `5,000` |
| Stop words | English and Portuguese NLTK stop-word lists |

The article title and text are combined before vectorization. TF-IDF assigns a numerical weight to terms based on their relevance within an article and across the article collection.

The resulting article matrix has the shape:

```text
(3047, 5000)
```

Each row represents an article, and each column represents a TF-IDF feature.

### 5. Train/Test Split

The aggregated user–article interactions are split into training and test sets using a holdout strategy.

```python
train_test_split(
    interactions_full_df,
    stratify=interactions_full_df["personId"],
    test_size=0.20,
    random_state=42
)
```

| Split | User–article pairs |
|---|---:|
| Training | 31,284 |
| Test | 7,822 |
| **Total** | **39,106** |

The split is stratified by user ID so that each user represented in the data is represented across the split.

### 6. User Profile Construction

A user profile is created by combining the TF-IDF vectors of the articles the user interacted with in the training set.

The profile is a weighted average: articles associated with stronger interaction signals contribute more to the user's representation.

Conceptually:

$$
mathbf{p}u =
\frac{\sum{i \in I_u} s_{u,i}\mathbf{v}i}
{\left| \sum{i \in I_u} s_{u,i}\mathbf{v}_i \right|_2}
$$

Where:

- $(\mathbf{p}_u\)$ is the profile vector for user $(u\)$.
- $(I_u\)$ is the set of articles interacted with by user $(u\)$ in the training set.
- $(s_{u,i}\)$ is the aggregated interaction strength.
- $(\mathbf{v}_i\)$ is the TF-IDF vector for article $(i\)$.
- `Normalize` applies vector normalization.

The notebook builds profiles for **1,140 users**. Each profile has 5,000 features, matching the article TF-IDF feature space.

### 7. Content-Based Recommendation

For a given user, the system computes cosine similarity between the user's profile and all article vectors.

$$
mathrm{sim}(u,i) =
\frac{\mathbf{p}_u \cdot \mathbf{v}_i}
{|\mathbf{p}_u|_2 \times |\mathbf{v}_i|_2}
$$

Articles are sorted by descending similarity. The recommender can then exclude article IDs supplied through `items_to_ignore` and return the requested number of results.

The recommendation output contains:

| Column | Description |
|---|---|
| `contentId` | Identifier of the recommended article |
| `recStrength` | Cosine similarity between the user profile and the article vector |

The notebook's recommender supports an optional verbose mode that can join article metadata such as title, URL, and language when an article DataFrame is supplied.

## Model Evaluation

The notebook evaluates the content-based model using a holdout test set and sampled non-interacted articles.

### Evaluation procedure

For each user represented in the test set:

1. Retrieve the articles that user interacted with in the test set.
2. Generate recommendations while excluding the user's training interactions.
3. Sample 100 articles that the user did not interact with in the full interaction data.
4. For each test article, compare its position against the sampled non-interacted articles.
5. Count whether the test article appears within the top 5 and top 10 positions.
6. Aggregate the results into Recall@5 and Recall@10.

The negative-item sample uses a fixed seed derived from the test article ID, making the sampling repeatable for a given item.

### Reported results

The notebook reports the following global metrics:

| Metric | Result |
|---|---:|
| Recall@5 | 0.1631 (16.31%) |
| Recall@10 | 0.2612 (26.12%) |

The evaluation processed **1,140 users**.

### How to interpret the results

- **Recall@5** measures the proportion of held-out interacted articles retrieved within the first five positions of the sampled candidate ranking.
- **Recall@10** measures the corresponding proportion within the first ten positions.

These values come from the notebook's sampled-negative evaluation procedure. They should not be interpreted as results from a full ranking against every possible article, nor as a production performance guarantee.

## REST API

The project includes a small Flask service that exposes recommendations through an HTTP GET endpoint.

### Endpoint

```http
GET /recommend/<userId>
```

Example:

```http
GET http://127.0.0.1:5000/recommend/3609194402293569455
```

The service:

1. Receives a user ID in the URL.
2. Loads the serialized recommender artifacts through `ContentBasedRecommender`.
3. Converts the user ID to an integer.
4. Generates recommendations for that user.
5. Returns the result as JSON Lines using Pandas `to_json(orient="records", lines=True)`.

The current service calls `recommend_items()` with its default settings, which returns up to 15 recommendations.

### Response format

The endpoint returns one JSON object per line. A response has this general structure:

```json
{"contentId": 123456789, "recStrength": 0.42}
{"contentId": 987654321, "recStrength": 0.37}
```

The IDs and similarity values above are illustrative only; actual results depend on the loaded model artifacts and requested user.

## Repository Structure

The following structure describes the files used by the notebook and service. Place the raw dataset files in `data/` or update the notebook's CSV paths to match your local setup.

```text
article-recommendation-system/
│
├── Recommendation systems model.ipynb
│
├── data/
│   ├── shared_articles.csv
│   └── users_interactions.csv
│
├── service/
│   ├── content_based_recommender.py
│   ├── recommender_system_service.py
│   ├── content_based_recommender_model.p
│   ├── user_profiles.p
│   ├── tfidf_matrix.p
│   └── item_ids.p
│
├── requirements.txt
├── Dockerfile
└── README.md
```

The exact folder names may differ in the original working directory. The Python service expects the serialized `.p` files to be located beside `content_based_recommender.py`, because it loads them relative to that module's directory.

## Installation

### Requirements

- Python 3
- pip
- The project dependencies listed in `requirements.txt`
- The serialized model artifacts for serving
- The raw DeskDrop CSV files for rerunning the notebook

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd article-recommendation-system
```

Replace `<YOUR_REPOSITORY_URL>` with the actual repository URL.

### 2. Create and activate a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The supplied requirements file lists:

```text
numpy
scipy
flask
pandas
sklearn
```

For a fresh environment, the package is commonly installed as `scikit-learn` even though the code imports it through the `sklearn` namespace. If installation fails because of the package name in the supplied requirements file, install it with:

```bash
pip install scikit-learn
```

The notebook also uses NLTK and downloads its English and Portuguese stop-word data. If you plan to rerun the notebook, install NLTK and download the required corpus:

```bash
pip install nltk
python -m nltk.downloader stopwords
```

## Running the API

Make sure the four serialized artifacts are in the same directory as `content_based_recommender.py`:

- `user_profiles.p`
- `tfidf_matrix.p`
- `item_ids.p`
- `content_based_recommender_model.p`

Then run the Flask service from the service directory:

```bash
python recommender_system_service.py
```

The service starts on port `5000` and binds to `0.0.0.0` as configured in the source code.

Test it locally:

```bash
curl http://127.0.0.1:5000/recommend/<USER_ID>
```

Replace `<USER_ID>` with a user ID that exists in the saved user profiles.

> **Important:** The API currently has a single recommendation route. It does not include authentication, request validation beyond integer conversion, pagination, health checks, or production deployment configuration.

## Model Artifacts

The notebook serializes the objects required by the serving code:

| Artifact | Purpose |
|---|---|
| `user_profiles.p` | Mapping of user IDs to their profile vectors |
| `tfidf_matrix.p` | TF-IDF article feature matrix |
| `item_ids.p` | Article IDs aligned with the rows of the TF-IDF matrix |
| `content_based_recommender_model.p` | Serialized recommender model object |

The service loads these files with Python `pickle`.

**Security warning:** Only load pickle files from a trusted source. Unpickling an untrusted file can execute arbitrary code. The artifacts must also remain compatible with the Python and library versions used to create them.

## Limitations

The current implementation has several limitations that should be considered:

1. **Cold-start users:** A user without a saved profile cannot receive personalized recommendations from the current implementation.
2. **New articles:** New articles are not automatically vectorized or added to the saved TF-IDF matrix. The model artifacts must be rebuilt to include them.
3. **Text dependence:** Recommendation quality depends on the available article title and body text. The current content representation does not use a learned semantic embedding model.
4. **Language handling:** English and Portuguese stop-word lists are used, but the notebook does not implement a dedicated multilingual language-detection or translation pipeline.
5. **Sampled evaluation:** The reported Recall@K metrics use 100 sampled non-interacted articles per test item rather than ranking against the complete catalogue.
6. **Limited serving features:** The Flask service is a basic inference endpoint and does not provide authentication, monitoring, caching, rate limiting, or production orchestration.
7. **Artifact loading:** The service loads the serialized objects when a `ContentBasedRecommender` instance is created. The current route creates an instance for each request, which can add avoidable loading overhead.
8. **Reproducibility:** The notebook contains local machine-specific CSV paths. These paths need to be updated for another environment.

## Possible Improvements

Potential next steps include:

- Add a configuration file for dataset and artifact paths.
- Refactor the notebook's preprocessing and training logic into reusable Python modules.
- Add automated tests for profile construction, recommendation ranking, and API responses.
- Load model artifacts once when the Flask application starts rather than once per request.
- Add input validation and clear responses for unknown user IDs.
- Add pagination and configurable `top_n` recommendations.
- Add a metadata endpoint to return article titles, URLs, and languages with recommendation scores.
- Compare content-based filtering with collaborative filtering and hybrid recommendation methods.
- Evaluate with additional ranking metrics such as Precision@K, MAP@K, and NDCG@K.
- Use temporal evaluation to better reflect the real-world task of recommending future content from past interactions.
- Explore multilingual semantic embeddings to improve matching across English and Portuguese articles.
- Add monitoring, logging, and deployment configuration if the service is prepared for production use.

These are proposed improvements, not features claimed to exist in the current implementation.

## Technologies

| Area | Tools |
|---|---|
| Programming language | Python |
| Data manipulation | Pandas, NumPy |
| Scientific computing | SciPy |
| Machine learning | scikit-learn |
| Text processing | NLTK |
| Text representation | TF-IDF |
| Recommendation algorithm | Content-Based Filtering |
| Similarity metric | Cosine Similarity |
| API | Flask |
| Model persistence | Pickle |
| Experiment environment | Jupyter Notebook |

---

## Author

**Mahmoud Morsy**

- GitHub: [github.com/morsycoo](https://github.com/morsycoo)
- LinkedIn: [linkedin.com/in/morsycoo](https://www.linkedin.com/in/morsycoo)

---

## Acknowledgements

- **CI&T DeskDrop** for the dataset describing article sharing and user interactions on an internal communication platform.
- The open-source Python ecosystem, including Pandas, NumPy, SciPy, scikit-learn, NLTK, and Flask.
