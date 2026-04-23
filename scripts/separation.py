import torch
import json
from pathlib import Path

# Run this as save_model.py
checkpoint = {
    "model_state_dict": torch.load("checkpoints/best_model.pth"),
    "classes"         : ["animal", "geometric", "symbolic"],
    "architecture"    : "efficientnet_b0",
    "test_accuracy"   : 0.78,
    "weighted_f1"     : 0.78,
    "input_size"      : 224,
}

torch.save(checkpoint, "checkpoints/kyrgyz_classifier_final.pth")
print("Model saved with metadata.")