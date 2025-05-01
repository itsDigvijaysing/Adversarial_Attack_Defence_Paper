# Evaluating and Enhancing the Adversarial Robustness of Florence-2 Model

This repository contains the Jupyter notebooks, research report (PDF), and necessary setup files for the project focused on evaluating the adversarial robustness of the Florence-2 base model (specifically for object detection) and implementing a multi-layered defense strategy.

**Authors:** Rajput Digvijaysing Bhatesing
**Supervisors:** Popat Raj Rameshkumar (TA), Prof. C. Krishna Mohan (Instructor)
**Institution:** Indian Institute of Technology Hyderabad

---

## Table of Contents

1.  [Overview](#overview)
2.  [Key Contributions](#key-contributions)
3.  [Repository Contents](#repository-contents)
4.  [Setup Instructions](#setup-instructions)
    *   [Environment Setup (Conda)](#environment-setup-conda)
    *   [Dataset Setup (COCO 2017)](#dataset-setup-coco-2017)
5.  [Usage / Running Experiments](#usage--running-experiments)
    *   [Running FGSM Evaluation Notebook](#running-fgsm-evaluation-notebook)
    *   [Running PGD Evaluation Notebook](#running-pgd-evaluation-notebook)
    *   [Notes on Execution](#notes-on-execution)
6.  [Expected Output](#expected-output)
7.  [Research Paper](#research-paper)

---

## Overview

Vision Foundation Models (VFMs) like Florence-2 \cite{xiao2024florence2} demonstrate remarkable capabilities across diverse visual tasks. However, their robustness against adversarial attacks is often under-explored yet critical for real-world deployment. This project provides an empirical study on the adversarial resilience of the Florence-2-Base model for COCO object detection against standard FGSM and PGD attacks. We find significant vulnerability and propose a practical, multi-layered defense pipeline combining input and feature-level transformations. Our results show that this defense recovers a substantial portion ($\approx$21.6\%) of the performance lost due to these attacks without requiring model retraining.

---

## Key Contributions

*   **First Robustness Evaluation:** Quantitative assessment of Florence-2-B's object detection vulnerability to FGSM and PGD attacks on COCO val2017.
*   **Multi-Layered Defense Proposal:** Introduction of a practical defense framework combining JPEG compression, prompt/image smoothing ensembles, adaptive Gaussian noise, spatial smoothing, and quantization/mixing.
*   **Defense Validation:** Empirical results demonstrating significant mAP recovery under both FGSM and PGD attacks using the proposed defense.
*   **Reproducibility Insights:** Documentation of challenges faced (baseline discrepancy, compute constraints, specification gaps) when evaluating large VFMs.

---

## Repository Contents

*   **`F2_final_fgsm.ipynb`**: Jupyter Notebook containing the code to run the full evaluation pipeline (Clean, FGSM Attack without Defense, FGSM Attack with Defense) on COCO val2017. Includes attack generation and defense implementations.
*   **`F2_final_pgd.ipynb`**: Jupyter Notebook containing the code to run the full evaluation pipeline (Clean, PGD Attack without Defense, PGD Attack with Defense) on COCO val2017. Includes attack generation and defense implementations.
*   **`paper`**: Directory likely containing the research report.
    *   `Report.pdf`: The final research report detailing the project. *(Modify filename as needed)*
*   **`environment.yml`**: Full Conda environment file listing all dependencies required to run the notebooks.
*   **`mini_env.yml`**: (If applicable) A minimal Conda environment file with only essential packages.
*   **`README.md`**: This file.

---

## Setup Instructions

### Environment Setup (Conda)

It is recommended to use Conda to manage dependencies for the Jupyter notebooks.

1.  **Create Environment:** Open your terminal or Anaconda Prompt and navigate to the project's root directory. Create the environment using the provided YAML file:
    ```bash
    conda env create -f environment.yml
    ```
    *(Alternatively, use `mini_env.yml` and install `jupyter`, `matplotlib`, `gputil`, `ipywidgets` manually if needed).*

2.  **Activate Environment:** Activate the newly created environment:
    ```bash
    conda activate florence2_robustness
    ```
    *(Note: Check the `name:` field in `environment.yml` for the exact environment name).*

3.  **Install Jupyter Kernel (Optional but Recommended):** To make the environment easily accessible from Jupyter:
    ```bash
    python -m ipykernel install --user --name=florence2_robustness --display-name="Python (Florence2 Robustness)"
    ```
    *(Replace `florence2_robustness` with the actual environment name if different).*

4.  **Key Dependencies:** The environment includes PyTorch, Transformers, PIL (Pillow), NumPy, tqdm, pycocotools, GPUtil, Matplotlib, Jupyter, etc. Ensure you have compatible CUDA drivers installed if using a GPU.

### Dataset Setup (COCO 2017)

1.  **Download:** Obtain the COCO 2017 dataset:
    *   Validation images (`val2017.zip`)
    *   Annotations (`annotations_trainval2017.zip`)

2.  **Extract and Place:** Extract the files. The Jupyter notebooks expect the data in the following structure relative to the notebook's location:
    ```
    ./Dataset/
    └── coco/
        ├── images/
        │   └── val2017/      <-- Contains all 5000 validation images
        └── annotations/
            └── annotations/  <-- Note the double 'annotations' folder
                └── instances_val2017.json
    ```

3.  **IMPORTANT - Path Modification:**
    *   If your dataset is located elsewhere, **you MUST modify the paths** defined within the code cells of both Jupyter notebooks (`F2_final_fgsm.ipynb` and `F2_final_pgd.ipynb`). Look for lines defining `ann_file` and `image_dir` (likely near the beginning or in the setup cells) and update them with the correct paths.

---

## Usage / Running Experiments

Ensure the Conda environment is activated (`conda activate florence2_robustness`).

1.  **Start Jupyter:** Launch Jupyter Lab or Jupyter Notebook from your terminal:
    ```bash
    jupyter lab
    # or
    jupyter notebook
    ```

2.  **Open Notebook:** Navigate to and open either `F2_final_fgsm.ipynb` or `F2_final_pgd.ipynb` in the Jupyter interface.

3.  **Select Kernel:** Ensure the correct kernel corresponding to your Conda environment (e.g., "Python (Florence2 Robustness)") is selected for the notebook.

### Running FGSM Evaluation Notebook

*   Open `F2_final_fgsm.ipynb`.
*   Run the cells sequentially ("Run All Cells" or step-by-step). This will perform the setup, load the model, execute the clean evaluation, generate FGSM attacks, run the attacked evaluation (no defense), run the attacked evaluation (with defense), and print results.

### Running PGD Evaluation Notebook

*   Open `F2_final_pgd.ipynb`.
*   Run the cells sequentially. This follows the same process as the FGSM notebook but uses the PGD attack.

### Notes on Execution

*   **GPU Requirement:** A CUDA-enabled GPU is highly recommended for reasonable execution times. The code should automatically attempt to use an available GPU.
*   **Execution Time:** Full evaluations on COCO val2017 are **very time-consuming** (potentially 6-10+ hours per notebook, especially PGD).
*   **Testing with Subset:** To test quickly, modify the file iteration loop within the notebooks. Look for the line processing `files` (e.g., `for fname in tqdm(files, ...):`) and consider adding a slice like `files = files[:50]` *after* sorting but *before* the loop to process only the first 50 images. Remember to revert this for the full evaluation.
*   **Visualization:** Attack visualization code (using `matplotlib`) might be present in the notebooks. Ensure this runs correctly in your Jupyter environment or disable it if not needed.

---

## Expected Output

Running the cells in the notebooks will:

1.  Display cell outputs, including status messages (device info, progress bars via `tqdm`).
2.  Generate and save intermediate detection results to JSON files in the same directory as the notebooks (e.g., `coco_clean_results.json`, `coco_fgsm_results.json`, etc.).
3.  Display the standard COCO evaluation summary tables within the notebook output for each scenario (Clean, Attacked, Defended).
4.  Display the "Defense Effectiveness Analysis" summary comparing AP scores and recovery percentages.
5.  Display the total execution time for the evaluation portion.

---

## Research Paper

The detailed findings, methodology, and discussion are presented in the research paper PDF included in this repository (e.g., in the `paper/` directory or as a top-level file like `Report.pdf`).

---