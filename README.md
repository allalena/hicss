## Appendix S2. Online Reproducibility Package (Code • Models • Instructions)
Scope: This appendix accompanies the paper and provides everything needed to run, test, and extend the full pipeline using the single Python script shared with the submission (teacher partial fine‑tuning → distillation + semi‑hard triplet → few‑shot evaluation → dynamic quantization → speed demo). It mirrors the implementation used to generate the results in the paper.
## S2.1 What’s in the Scripts 
* (imports, seeding, Colab mount) → Reproducibility & environment
* Dataset classes (WeedNet, SubsetWithLabels, SubsetWithTransform)
* Transforms & splits (StratifiedShuffleSplit)
* Teacher (EfficientNet‑B7) with partial unfreeze of features.6–8 + 2048‑D head 
* Student backbones (MobileNetV3‑Small, ShuffleNetV2‑1.0, EfficientNet‑B0) each → 2048‑D 
* Weighted embedding ensemble (softmax‑learned weights) 
* <b>Losses:</b> MSE KD on embeddings + TripletMarginLoss with semi‑hard miner (margin=1.0) 
* Few‑shot evaluation via prototypical episodes (EasyFSL TaskSampler) 
* Quantization: post‑training dynamic INT8 on nn.Linear 
* Speed demo: synthetic episodes throughput
* Ablation analysis
**Main pipeline + 3 ablations:** `weed_classification_pipeline_plus_ablation_analysis`  
  Implements the full pipeline (EfficientNet‑B7 partial fine‑tuning → MSE knowledge distillation + semi‑hard triplet loss → few‑shot evaluation → dynamic INT8 quantization).  
  **Includes:** three ablation studies (**Teacher‑only**, **Distillation‑only**, **Triplet‑only**), the **inference speed demo**, and the **figures/visualizations** used in the paper.

- **Ablation notebooks (separate):**
  - **No partial fine‑tuning (Frozen teacher):** `<weed_classification_pipeline_no_partial_fine_tuning_ablation>.ipynb`  
    Keeps the teacher frozen at ImageNet initialization; trains students with **MSE + semi‑hard triplet**.
  - **Single student only:** `<weed_classification_pipeline_single_student>.ipynb`  
    Replaces the ensemble with a single lightweight backbone (MobileNetV3) and trains with **MSE + semi‑hard triplet**.

> ℹ️ The main file contains the end‑to‑end pipeline, three ablations, inference speed benchmarking, and the plots reproduced in the paper. The **“no partial fine‑tuning”** and **“single student”** ablations are provided as two separate `.ipynb` notebooks.
