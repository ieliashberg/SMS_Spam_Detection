from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
from cpu_tracker import CPUTracker


def train_svm(X_train, y_train, X_test, y_test):
    """
    Train a linear SVM with CPU tracking. Returns model, metrics, and CPU usage data.
    """
    svc = SVC(kernel='linear', probability=True, random_state=42)
    tracker = CPUTracker()
    tracker.start()
    svc.fit(X_train, y_train)
    timestamps, cpu_usage = tracker.stop()

    y_pred = svc.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    y_prob = svc.predict_proba(X_test)[:, 1]

    return {
        'model': svc,
        'accuracy': acc,
        'confusion_matrix': cm,
        'classification_report': report,
        'y_prob': y_prob,
        'timestamps': timestamps,
        'cpu_usage': cpu_usage
    }


def train_random_forest(X_train, y_train, X_test, y_test):
    """
    Train a Random Forest with CPU tracking. Returns model, metrics, and CPU usage data.
    """
    clf = RandomForestClassifier(random_state=42)
    tracker = CPUTracker()
    tracker.start()
    clf.fit(X_train, y_train)
    timestamps, cpu_usage = tracker.stop()

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    y_prob = clf.predict_proba(X_test)[:, 1]

    return {
        'model': clf,
        'accuracy': acc,
        'confusion_matrix': cm,
        'classification_report': report,
        'y_prob': y_prob,
        'timestamps': timestamps,
        'cpu_usage': cpu_usage
    }


def train_naive_bayes_grid_search(X_train, y_train, X_test, y_test):
    """
    Grid search over alpha and fit_prior for MultinomialNB. Returns best model,
    metrics, grid results, and CPU usage data.
    """
    best_accuracy = 0.0
    best_model = None
    best_params = {}
    results = []
    tracker = CPUTracker()
    tracker.start()

    for alpha in np.arange(0.1, 1.1, 0.1):
        for fit_prior in [True, False]:
            nb = MultinomialNB(alpha=alpha, fit_prior=fit_prior)
            nb.fit(X_train, y_train)
            y_pred = nb.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            results.append({'alpha': alpha, 'fit_prior': fit_prior, 'accuracy': acc})
            if acc > best_accuracy:
                best_accuracy = acc
                best_model = nb
                best_params = {'alpha': alpha, 'fit_prior': fit_prior}

    timestamps, cpu_usage = tracker.stop()

    y_pred_best = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)
    report = classification_report(y_test, y_pred_best, output_dict=True)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    return {
        'model': best_model,
        'best_params': best_params,
        'accuracy': best_accuracy,
        'confusion_matrix': cm,
        'classification_report': report,
        'y_prob': y_prob,
        'grid_results': results,
        'timestamps': timestamps,
        'cpu_usage': cpu_usage
    }
