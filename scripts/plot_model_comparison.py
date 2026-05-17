"""Plot comparison of model metrics across domains.

This script uses hard-coded metrics collected from the training runs in this
session to generate summary plots: AUC, F1, FPR, TPR for Mixed and UWF tests.

Outputs PNG files under docs/figures/.
"""
import os
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'docs', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

# Collected metrics
models = [
    'Mixed-frozen',
    'UWF-adapted',
    'Domain-adaptive (mean)'
]

# Mixed test metrics
mixed_auc = [0.9961, 0.5881, 0.9963]
mixed_f1  = [0.9033, 0.0,    0.9113]
mixed_fpr = [0.0043, 0.0370, 0.0049]
mixed_tpr = [0.8237, 0.0,    0.8371]

# UWF test metrics
uwf_auc = [0.4290, 1.0000, 1.0000]
uwf_f1  = [0.0,    0.9903, 0.9842]
uwf_fpr = [0.0038, 0.0000, 0.0011]
uwf_tpr = [0.0,    0.9808, 1.0000]

x = np.arange(len(models))
width = 0.6

# Plot AUC comparison
plt.figure(figsize=(8,5))
plt.bar(x - width/2, mixed_auc, width/2, label='Mixed test')
plt.bar(x + width/2, uwf_auc, width/2, label='UWF test')
plt.xticks(x, models, rotation=20)
plt.ylabel('AUC')
plt.title('AUC by model and test domain')
plt.ylim(0.0, 1.02)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'model_auc_comparison.png'), dpi=200)
plt.close()

# Plot F1 comparison
plt.figure(figsize=(8,5))
plt.bar(x - width/2, mixed_f1, width/2, label='Mixed test')
plt.bar(x + width/2, uwf_f1, width/2, label='UWF test')
plt.xticks(x, models, rotation=20)
plt.ylabel('F1')
plt.title('F1 by model and test domain')
plt.ylim(0.0, 1.02)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'model_f1_comparison.png'), dpi=200)
plt.close()

# Plot FPR comparison (log scale for tiny values)
plt.figure(figsize=(8,5))
plt.bar(x - width/2, mixed_fpr, width/2, label='Mixed test')
plt.bar(x + width/2, uwf_fpr, width/2, label='UWF test')
plt.xticks(x, models, rotation=20)
plt.ylabel('FPR')
plt.title('False Positive Rate by model and test domain')
plt.yscale('log')
plt.tight_layout()
plt.legend()
plt.savefig(os.path.join(OUT_DIR, 'model_fpr_comparison.png'), dpi=200)
plt.close()

# Plot TPR comparison
plt.figure(figsize=(8,5))
plt.bar(x - width/2, mixed_tpr, width/2, label='Mixed test')
plt.bar(x + width/2, uwf_tpr, width/2, label='UWF test')
plt.xticks(x, models, rotation=20)
plt.ylabel('TPR (Recall)')
plt.title('TPR by model and test domain')
plt.ylim(0.0, 1.02)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'model_tpr_comparison.png'), dpi=200)
plt.close()

# Multi-seed boxplots for domain-adaptive seeds (Mixed and UWF F1)
seed_labels = ['11','42','1337']
mixed_f1_seeds = [0.9107, 0.9067, 0.9165]
uwf_f1_seeds   = [0.9905, 0.9811, 0.9811]

plt.figure(figsize=(8,5))
plt.boxplot([mixed_f1_seeds, uwf_f1_seeds], labels=['Mixed F1 (seeds)', 'UWF F1 (seeds)'])
plt.title('Seed variability (F1) — Domain-adaptive model')
plt.ylabel('F1')
plt.ylim(0.8, 1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'domain_adaptive_seed_f1_boxplot.png'), dpi=200)
plt.close()

print('Saved figures to', OUT_DIR)
