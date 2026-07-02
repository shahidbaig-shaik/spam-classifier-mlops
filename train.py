"""
Train a spam classifier pipeline (TF-IDF + Naive Bayes) with W&B experiment tracking.
Saves the fitted pipeline to disk for serving via FastAPI.
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

import wandb


def load_data(filepath: str) -> pd.DataFrame:
    """Load the SMS Spam Collection dataset (tab-separated, no header: label<TAB>message)."""
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

    hparams = {
        "model_type": "MultinomialNB",
        "vectorizer": "TfidfVectorizer",
        "stop_words": None,          # keep all words — removing stop words hurt recall
        "max_features": 10000,       # 10k features instead of 5k
        "alpha": 0.1,                # less smoothing for sharper predictions
        "test_size": 0.2,
        "random_state": 42,
    }

    wandb.config.update(hparams)

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

    wandb.log(metrics)

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
        # non-fatal — plots are nice-to-have, not required
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
    """Run example predictions to sanity-check the trained model."""
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


if __name__ == "__main__":
    print("=" * 60)
    print("  SPAM CLASSIFIER — Training Pipeline")
    print("=" * 60)
    print()

    # WANDB_MODE=disabled makes this a no-op in CI
    run = wandb.init(
        project="spam-classifier",
        name="nb-experiment-2",
        tags=["naive-bayes", "tfidf", "tuned"],
    )

    data = load_data("data/SMSSpamCollection")
    model, X_train, X_test, y_train, y_test = train_model(data)
    metrics = evaluate_model(model, X_test, y_test)
    save_model(model, "model/spam_classifier.joblib")
    demo_predictions(model)

    wandb.finish()

    print("=" * 60)
    print("  ✅ Training complete. Model saved to model/spam_classifier.joblib")
    print("=" * 60)
