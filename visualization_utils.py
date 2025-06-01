import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import numpy as np


def plot_label_distribution(label_counts):
    plt.figure(figsize=(6, 6))
    colors = ['#66c2a5', '#fc8d62']
    plt.pie(
        label_counts,
        labels=['non-spam', 'spam'],
        autopct='%1.1f%%',
        startangle=140,
        colors=colors
    )
    plt.title('Distribution of Spam vs. Non-Spam SMS')
    plt.axis('equal')
    plt.show()


def plot_bar_metric(spam_value, ham_value, title, y_label):
    plt.figure(figsize=(5, 4))
    plt.bar(['Spam', 'Ham'], [spam_value, ham_value], color=['#fc8d62', '#66c2a5'])
    plt.title(title)
    plt.xlabel('SMS Type')
    plt.ylabel(y_label)
    plt.show()


def plot_correlation_matrix(df, cols):
    corr = df[cols].corr()
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Matrix of Text Features')
    plt.show()


def show_top_words(text_series, title):
    all_words = ' '.join(text_series).split()
    word_counts = Counter(all_words)
    common = word_counts.most_common(10)

    print(f"Top 10 Most Common Words in {title}:")
    for w, c in common:
        print(f"  {w}: {c} occurrences")

    wc = WordCloud(width=800, height=400, background_color='white')
    wc = wc.generate_from_frequencies(dict(common))

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud: {title}')

    plt.subplot(1, 2, 2)
    words, counts = zip(*common)
    plt.bar(words, counts, color='#8da0cb')
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Bar Graph: {title}')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, labels, title, cmap='Blues'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.show()


def plot_roc_curve(y_true, y_prob, title):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.show()


def plot_cpu_usage(timestamps, cpu_usage, title):
    plt.figure(figsize=(6, 4))
    plt.plot(timestamps, cpu_usage, marker='o', linestyle='-')
    plt.xlabel("Time (s)")
    plt.ylabel("CPU Usage (%)")
    plt.title(title)
    plt.grid(True)
    plt.show()
