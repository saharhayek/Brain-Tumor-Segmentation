# Brain-Tumor-Segmentation
Process multimodal MRI scans from the BraTS 2020 dataset, segment tumor regions,  and reconstruct a 3D model of the tumor using machine learning and deep learning methods.

Dataset:
BraTS 2020 dataset (multi-institutional MRI):- Kaggle
  - Modalities: T1, T1Gd, T2, T2-FLAIR
  - NIfTI (.nii.gz) images converted to HDF5 for efficient loading
  - Tumor labels:
      1 = Necrotic / Non-Enhancing Tumor Core (NCR/NET)
      2 = Edema (ED)
      3 = Enhancing Tumor (ET)
  - Images standardized: co-registered, skull-stripped, 1mm³ resolution
  - ~7GB total, ~57,000 HDF5 slices

Pipeline implemented in the notebook:
# 1. Data loading:
   - Reads thousands of .h5 files containing MRI slices and masks.
   - Organizes file paths, shuffles dataset, splits into train/validation (90/10).

# 2. Preprocessing:
   - Normalize MRI slices.
   - Display single-modality channels (e.g., T1, T2, FLAIR).
   - Display tumor mask channels (NCR/NET, ED, ET).
   - Overlay masks on MRI images for inspection.
   - Convert images to torch tensors for model training.

# 3. Dataset class:
   - Custom PyTorch Dataset for loading MRI slices and masks.
   - Returns (image, mask) as tensors.
   - Handles dimension corrections and scaling.

# 4. Dataloaders:
   - Training and validation dataloaders built with batch_size=5.
   - Shuffles training data, preserves validation order.

# 5. Segmentation models:
   Models tested:
     - UNet
     - Upgraded UNet (reduced parameters)
     - Attention UNet (attention gate mechanism)
   Parameter counts:
     UNet: ~5.49M parameters
     Upgraded UNet: ~3.08M parameters
     Attention UNet: ~3.26M parameters

  #  Training notes:
     - ~20k slices used for training out of 57k
     - 12 epochs, 50 batches/epoch, batch size = 32 (original setup)
     - Loss decreases steadily in upgraded and attention models
     - Attention UNet gives best boundary separation and lowest validation loss

# 6. Model Evaluation:
   Metrics:
     - Accuracy
     - Precision
     - Recall
     - Dice-based loss curves
   Evaluation based on comparing predicted masks vs ground truth for NEC, ED, ET.
   Visual inspection of predicted vs ground-truth RGB mask overlays.

# 7. 3D Reconstruction:
   Steps:
     - Load normalized MRI slices and corresponding tumor masks.
     - Combine predicted mask slices into a 3D volume stack.
     - Visualize the final 3D tumor shape.
     - Useful for surgical planning, tumor morphology understanding, and teaching.

What the notebook demonstrates:
- End-to-end MRI tumor segmentation pipeline
- Visualization of raw modalities, tumor masks, prediction masks
- Benchmarking of three deep-learning architectures
- 3D reconstruction from 2D masks
- Comparison of segmentation quality across UNet variants

Dependencies:
pip install numpy pandas torch torchvision h5py matplotlib

How to run:
1. Ensure HDF5 (.h5) MRI slices and masks are in the expected directory.
2. Open the notebook:
   jupyter notebook
   open GBM548_Project.ipynb
3. Run cells from top to bottom.
4. Adjust file paths for .h5 files if needed.

Output:
- Segmented tumor regions for each MRI slice
- Training/validation loss curves for all models
- Visual comparison of predicted vs ground truth masks
- 3D tumor reconstruction volume
- Benchmark results showing best model = Attention UNet

Notes:
- GPU recommended for training UNet-based models.
- HDF5 dataset must follow the same naming and folder structure as in the notebook.
- This project matches the content described in the presentation and report.

License:
For academic and research use.
