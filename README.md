# Explainable Brain MRI Classification

A four-class MRI classification and explanation prototype developed for DECO7281 Human-Centred Explainable AI.

The project combines a MobileNetV3-Small classifier, Grad-CAM visualisation, and optional Gemini teaching explanations in English and Chinese Gradio interfaces. It demonstrates how a prediction can be presented alongside model attention and an explanation. This is an educational prototype, not a clinically validated diagnostic tool.

## Main features

- Classifies a single two-dimensional brain MRI image as `NonDemented`, `VeryMild`, `Mild`, or `Moderate`.
- Displays class probabilities and the predicted class confidence.
- Generates a Grad-CAM overlay for the predicted class at the uploaded image's original size.
- Provides optional Gemini explanations describing highlighted regions and uncertainty.
- Offers an English interface (`app.py`) and a Chinese interface (`appcn.py`).

The English interface has prediction and combined explanation buttons, and displays a manual-review message when confidence is below 0.50. The Chinese interface provides separate prediction, heatmap, and explanation buttons.

## Model and workflow

The classifier uses torchvision's pretrained MobileNetV3-Small backbone with the following head:

```text
Backbone features → Linear(in_features, 256) → Hardswish
                  → Dropout(0.25) → Linear(256, 4)
```

Images are converted to RGB, resized to 160 × 160, and normalised with channel mean 0.5 and standard deviation 0.5. The class order is `NonDemented`, `VeryMild`, `Mild`, `Moderate`.

The training pipeline in `train.py`:

1. Collects images from class folders and applies class caps of 6,000 / 6,000 / 5,000 / 488.
2. Extracts the OASIS subject ID from each filename and stratifies subjects into approximately 70% training, 15% validation, and 15% test sets. All slices and repeat scans belonging to a subject stay in one set.
3. Uses weighted sampling and class-balanced cross-entropy, with horizontal flips and additional augmentation for Mild and Moderate images.
4. Trains the classifier and the last six backbone parameter tensors using AdamW, an initial learning rate of `1e-3`, batch size 64, and up to eight epochs.
5. Selects the checkpoint using validation macro-F1, then reports a test confusion matrix and classification report.

The split proportions apply to subjects, so image counts may differ. Three-way stratification requires sufficient subjects in every class; unrecognised subject IDs, conflicting subject labels, or insufficient class coverage raise an error.

Grad-CAM uses the final convolutional layer and blends its heatmap with the input image. The overlay indicates model sensitivity rather than lesion segmentation. Confidence is the maximum softmax score and is not calibrated.

## Repository guide

| File or directory | Role |
| --- | --- |
| `app.py` | English four-class inference and explanation interface |
| `appcn.py` | Chinese four-class inference and explanation interface |
| `train.py` | Four-class training, evaluation, and optional Grad-CAM helpers |
| `gradcam.py` | Batch Grad-CAM generation using an existing four-class checkpoint |
| `oasis_fourclass_balanced.pth` | Saved four-class checkpoint referenced by the interfaces |
| `DownloadData.py` | Download helper for the Kaggle `ninadaithal/imagesoasis` dataset |
| `outputs/` | Saved figures, example overlays, and checkpoints |
| `Other_function_may_use_in_future/` | Retained figure utilities and early experiments, including a separate binary baseline |

The `train.py` entry point calls training only; its Grad-CAM helpers are optional. The standalone `gradcam.py` selects up to three examples per class from the configured image folders. These examples are visual illustrations, not a held-out evaluation.

## Coursework artifacts

Existing outputs include [dataset examples](outputs/fig_dataset_grid.png), a [labelled Grad-CAM grid](outputs/gradcam_grid_labeled.png), and a [performance curve](outputs/train_val_curve_manual.png).

The curve is plotted from manually entered values in `Other_function_may_use_in_future/训练曲线图生成.py`:

| Recorded quantity | Final value |
| --- | --- |
| Training accuracy | 0.925 |
| Validation macro-F1 | 0.872 |

These are historical coursework records. Original run logs and final split manifests are not included; the archived scores do not establish performance for the subject-level split described above.

## Technology and configuration

Python, PyTorch, torchvision, Gradio, NumPy, pandas, Pillow, OpenCV, scikit-learn, Matplotlib, and tqdm. The dataset helper uses `kagglehub`; optional teaching text uses `google.generativeai`.

The scripts retain local dataset and checkpoint paths. The configured `Data/` directory and a dependency manifest are not included in this snapshot. The download helper prints a Kaggle cache path and does not automatically populate `Data/`.

Gemini reads the optional `GEMINI_API_KEY` environment variable. With no key, the existing explanation callback returns a configuration message. When enabled, Gemini receives the MRI overlay and a prompt containing the predicted label. Classification and Grad-CAM are local; text generation uses the external service and may be inaccurate.

The Chinese interface stops with an error if its trained checkpoint cannot be loaded.
