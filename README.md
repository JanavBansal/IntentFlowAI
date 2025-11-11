# IntentFlow AI

IntentFlow AI is a research and trading signal platform focused on identifying 5–20 day swing opportunities within the NIFTY 200 universe. The system combines heterogeneous data sources (flows, transactions, fundamentals, narrative tone, and price confirmation) to surface low-drawdown, small-capital strategies using LightGBM models with meta-labeling and regime filters.

## Repository layout
- `intentflow_ai/`: Core Python package with modular components for configs, data ingestion, feature engineering, modeling, and pipelines.
- `data/`: Local storage for raw, intermediate, and curated datasets. Populated via ingestion jobs.
- `dashboard/`: Streamlit app for visualizing signals, diagnostics, and portfolio context.
- `scripts/`: Convenience entry points for orchestrating training, evaluation, and reporting tasks.
- `notebooks/`: Exploratory analysis, prototyping, and research documentation.
- `models/` & `experiments/`: Persisted artifacts such as trained weights, evaluation reports, and metadata.

## Getting started
1. Create and activate a Python 3.10+ virtual environment.
2. Install dependencies: `pip install -r requirements.txt`.
3. Configure secrets and data paths in `intentflow_ai/config/settings.py`.
4. Run the training script: `python -m scripts.run_training`.
5. Launch the Streamlit dashboard: `streamlit run dashboard/app.py`.

Each module contains docstrings and inline notes describing its responsibilities to make onboarding and future expansion straightforward.
