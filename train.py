"""
PHASE 1 + 2: Train a Spam Classifier with Experiment Tracking
==============================================================
This script does 5 things:
  1. Loads the SMS Spam Collection dataset
  2. Initializes W&B experiment tracking (Phase 2)
  3. Trains a text classification pipeline (TF-IDF + Naive Bayes)
  4. Logs metrics & visualizations to W&B
  5. Saves the trained model

WHY EXPERIMENT TRACKING?
------------------------
Remember when we compared Naive Bayes vs Logistic Regression?
We had to eyeball the terminal output and mentally compare numbers.

With W&B (Weights & Biases), every training run is automatically:
  - Logged with its hyperparameters (what settings you used)
  - Logged with its metrics (how well it performed)
  - Visualized with charts (confusion matrix, ROC curve, etc.)
  - Compared side-by-side in a dashboard

Think of it like version control (git) but for experiments.
Instead of "I think the model with alpha=0.5 was better...",
you can just open the dashboard and SEE the comparison.

KEY CONCEPTS:
-------------
- wandb.init():   Starts a new "run" — like opening a new log file
- wandb.config:   Records your settings (hyperparams, model type, etc.)
- wandb.log():    Records your results (accuracy, F1, etc.)
- wandb.finish(): Closes the run and syncs everything to the cloud
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
import joblib
import os

# ---- Phase 2: Experiment tracking ----
import wandb


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the SMS Spam Collection dataset.
    Tab-separated, no header: label<TAB>message
    """
    print("📂 Loading dataset...")
    df = pd.read_csv(filepath, sep='\t', names=['label', 'message'], encoding='latin-1')
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

    print(f"   Total messages: {len(df)}")
    print(f"   Ham (not spam): {len(df[df['label'] == 'ham'])}")
    print(f"   Spam:           {len(df[df['label'] == 'spam'])}")
    print()
    return df


def train_model(df: pd.DataFrame) -> tuple:
    """Train the spam classifier pipeline."""

    # --- Split data ---
    print("✂️  Splitting data: 80% train, 20% test...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['message'],
        df['label_num'],
        test_size=0.2,
        random_state=42,
        stratify=df['label_num'],
    )
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples:     {len(X_test)}")
    print()

    # --- Hyperparameters ---
    # We define these as variables so we can:
    #   1. Pass them to the model
    #   2. Log them to W&B (so we know what settings produced what results)
    hparams = {
        "model_type": "MultinomialNB",
        "vectorizer": "TfidfVectorizer",
        "stop_words": None,          # CHANGED: Keep ALL words (including "the", "is", etc.)
        "max_features": 10000,       # CHANGED: 10k features instead of 5k
        "alpha": 0.1,                # CHANGED: Less smoothing (sharper predictions)
        "test_size": 0.2,
        "random_state": 42,
    }

    # ---- Phase 2: Log hyperparameters to W&B ----
    # This records WHAT SETTINGS you used for this run.
    # Later in the dashboard, you can filter/sort runs by these values.
    # e.g., "Show me all runs where alpha > 0.5"
    wandb.config.update(hparams)

    # --- Build Pipeline ---
    print("🔧 Building pipeline: TfidfVectorizer → MultinomialNB...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            stop_words=hparams['stop_words'],
            max_features=hparams['max_features'],
        )),
        ('classifier', MultinomialNB(
            alpha=hparams['alpha'],
        )),
    ])

    # --- Train ---
    print("🏋️  Training the model...")
    pipeline.fit(X_train, y_train)
    print("   ✅ Training complete!")
    print()

    return pipeline, X_train, X_test, y_train, y_test


def evaluate_model(pipeline, X_test, y_test) -> dict:
    """Evaluate the model and log metrics to W&B."""
    print("📊 Evaluating model on test data...")
    predictions = pipeline.predict(X_test)

    metrics = {
        'accuracy': accuracy_score(y_test, predictions),
        'f1_score': f1_score(y_test, predictions),
        'precision': precision_score(y_test, predictions),
        'recall': recall_score(y_test, predictions),
    }

    # Print to terminal
    print(f"   Accuracy:  {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.1f}%)")
    print(f"   F1 Score:  {metrics['f1_score']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall:    {metrics['recall']:.4f}")
    print()

    print("   Detailed Classification Report:")
    print("   " + "-" * 55)
    report = classification_report(y_test, predictions, target_names=['ham', 'spam'])
    for line in report.split('\n'):
        print(f"   {line}")
    print()

    # ---- Phase 2: Log metrics to W&B ----
    # This records HOW WELL this run performed.
    # These show up as charts in your W&B dashboard.
    wandb.log(metrics)

    # ---- Phase 2: Log sklearn diagnostic plots ----
    # This generates a bunch of useful visualizations automatically:
    #   - Confusion Matrix: shows where the model gets confused
    #   - ROC Curve: trade-off between true positives and false positives
    #   - Precision-Recall Curve: useful for imbalanced datasets (like ours!)
    #   - Feature Importances: which words matter most
    try:
        y_probas = pipeline.predict_proba(X_test)
        wandb.sklearn.plot_classifier(
            pipeline, X_train=None, X_test=X_test, y_train=None, y_test=y_test,
            y_pred=predictions, y_probas=y_probas,
            labels=["ham", "spam"],
            model_name="MultinomialNB",
            feature_names=None,
        )
        print("   📈 Logged diagnostic plots to W&B")
    except Exception as e:
        # If W&B is disabled or plotting fails, that's fine — training still works
        print(f"   ⚠️  Skipped W&B plots: {e}")
    print()

    return metrics


def save_model(pipeline, filepath: str):
    """Save the trained pipeline to disk using joblib."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(pipeline, filepath)
    file_size = os.path.getsize(filepath) / (1024 * 1024)
    print(f"💾 Model saved to: {filepath}")
    print(f"   File size: {file_size:.2f} MB")


def demo_predictions(pipeline):
    """Run example predictions to see the model in action."""
    print()
    print("🔮 Demo Predictions:")
    print("=" * 60)

    test_messages = [
        "Hey, are we still meeting for lunch tomorrow?",
        "CONGRATULATIONS! You've won a £1000 gift card! Call NOW!",
        "Can you pick up milk on the way home?",
        "FREE entry to win a brand new iPhone! Text WIN to 80085",
        "I'll be there in 5 minutes",
        "URGENT! Your account has been compromised. Click here to verify.",
    ]

    for msg in test_messages:
        prediction = pipeline.predict([msg])[0]
        probability = pipeline.predict_proba([msg])[0]
        confidence = probability.max()
        label = "🚫 SPAM" if prediction == 1 else "✅ HAM "

        print(f"   {label} ({confidence:.1%} confident)")
        print(f"   └─ \"{msg[:60]}{'...' if len(msg) > 60 else ''}\"")
        print()


# ============================================================
#  MAIN: Run the full training pipeline
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  SPAM CLASSIFIER — Training Pipeline")
    print("=" * 60)
    print()

    # ---- Phase 2: Initialize W&B run ----
    # This starts a new tracked "run" in your W&B project.
    # - project: groups related runs together (like a repo)
    # - name:    a human-readable label for THIS specific run
    #
    # If WANDB_MODE=disabled (e.g., in CI tests), this is a no-op.
    run = wandb.init(
        project="spam-classifier",
        name="nb-experiment-2",
        tags=["naive-bayes", "tfidf", "tuned"],
    )

    # 1. Load data
    data = load_data("data/SMSSpamCollection")

    # 2. Train model
    model, X_train, X_test, y_train, y_test = train_model(data)

    # 3. Evaluate (and log to W&B)
    metrics = evaluate_model(model, X_test, y_test)

    # 4. Save model
    save_model(model, "model/spam_classifier.joblib")

    # 5. Demo predictions
    demo_predictions(model)

    # ---- Phase 2: Finish the W&B run ----
    # IMPORTANT: Always call this! It flushes all logged data to the cloud.
    # Without this, your metrics might not show up in the dashboard.
    wandb.finish()

    print("=" * 60)
    print("  ✅ Phase 1+2 Complete! Model trained, tracked, and saved.")
    print("  📁 Model file: model/spam_classifier.joblib")
    print("  📊 Check your W&B dashboard for metrics & plots!")
    print("  🔜 Next: Phase 3 — Serve the model with FastAPI")
    print("=" * 60)
