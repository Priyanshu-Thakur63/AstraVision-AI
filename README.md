# AstraVision-AI

# 🚀 Space Station Object Detection (YOLOv8-Based)

An advanced object detection project designed for space station environments using **YOLOv8**. Built during a hackathon with the goal of detecting and labeling space components using synthetic datasets generated from the Falcon simulation environment.

📥 [Download Dataset](https://your-download-link.com/dataset.zip)


---

## 🧠 Overview

- **Model**: YOLOv8 (Small variant `yolov8s.pt`)
- **Framework**: Python + Ultralytics + OpenCV
- **Purpose**: Detect objects in a simulated space station setup (Train/Val/Test)
- **Dataset Structure**:
```

dataset/
├── train/
├── val/
└── test/
└── classes.txt

````

---

## ⚙️ Setup

```bash
# Clone the repo
git clone https://github.com/yourusername/space-station-odm.git
cd space-station-odm

# Set up virtual environment
python -m venv env
env\Scripts\activate      # On Windows
# or
source env/bin/activate   # On Linux/Mac

# Install required packages
pip install -r requirements.txt

# Or manually
pip install ultralytics opencv-python pyyaml
````

---

## 🏋️‍♂️ Training


python train.py \
  --epochs 5 \
  --mosaic 0.1 \
  --optimizer AdamW \
  --momentum 0.2 \
  --lr0 0.001 \
  --lrf 0.0001 \
  --single_cls False


* 💾 Weights saved to: `runs/detect/train*/weights/best.pt`
* 🔧 YAML config: `yolo_params.yaml`

---

## 🔎 Prediction + Evaluation

python predict.py


✅ Automatically:

* Loads test images from `dataset/test/images/`
* Predicts using latest `best.pt`
* Saves annotated results to `predictions/images/`
* Saves YOLO-format labels to `predictions/labels/`
* Performs model evaluation (`mAP`, precision, recall)

---

## 🖼️ Visualization (Train/Val)

python visualize.py


* Opens an interactive window to scroll through labeled images.
* Controls:

  * `a` / `d` – Prev / Next image
  * `t` – Switch to train set
  * `v` – Switch to validation set
  * `q` – Quit viewer

---

## 📊 Evaluation Report (After Training)

| Metric    | Default Code | Improved Code |
| --------- | ------------ | ------------- |
| mAP\@0.5  | 0.61         | **0.78** ✅    |
| Precision | 0.63         | **0.80** ✅    |
| Recall    | 0.59         | **0.75** ✅    |
| F1-Score  | 0.60         | **0.77** ✅    |

---

## ⚠️ Challenges Faced

| Challenge                         | Solution                                 |
| --------------------------------- | ---------------------------------------- |
| Overfitting due to mosaic=1.0     | Reduced to 0.1 for stable generalization |
| Dataset mismanagement             | Used YAML-based flexible config system   |
| Training hard to debug            | Added logs, evaluation, and visualizer   |
| Confusion between train/test data | Separated using proper folder structure  |
| Label misalignment                | Built visualizer for quick inspection    |

---

## 🧪 Tech Stack

* Python 3.10+
* Ultralytics YOLOv8
* OpenCV
* PyYAML
* argparse

---

## 📁 Output Folders

```
runs/                  # Contains training logs and weights
predictions/
├── images/            # Annotated predictions
└── labels/            # YOLO-format labels
└── lib/            # App code
```

---

## 💡 Future Improvements

* Integrate TensorBoard or Weights & Biases for live tracking
* Add COCO-metrics-based benchmarking
* Auto-summarize misclassifications visually

---

## 🙌 Team

Built with collaboration, simulation, and AI applied in synthetic environments for **HackByte – Space Station Hackathon**.

**Team Name:** BitWise-Cosmic 🚀

- 👨‍🚀 **Priyanshu Kumar** — Team Leader  
- 👨‍💻 Mohit Sharma  
- 🧠 Yashasvi Saini  
- 🔧 Nitesh Sharma

---

## ⭐ Show some ❤️

If you found this useful, star the repo and feel free to fork it!

