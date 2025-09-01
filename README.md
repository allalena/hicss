## Appendix S2. Online Reproducibility Package (Code • Models • Instructions)
Scope: This appendix accompanies the paper and provides everything needed to run, test, and extend the full pipeline using the single Python script shared with the submission (teacher partial fine‑tuning → distillation + semi‑hard triplet → few‑shot evaluation → dynamic quantization → speed demo). It mirrors the implementation used to generate the results in the paper.
## S2.1 What’s in the Script 
* (imports, seeding, Colab mount) → Reproducibility & environment
* Dataset classes (WeedNet, SubsetWithLabels, SubsetWithTransform)
* Transforms & splits (StratifiedShuffleSplit)
Teacher (EfficientNet‑B7) with partial unfreeze of features.6–8 + 2048‑D head 
Student backbones (MobileNetV3‑Small, ShuffleNetV2‑1.0, EfficientNet‑B0) each → 2048‑D 
Weighted embedding ensemble (softmax‑learned weights) 
<b>Losses:</b> MSE KD on embeddings + TripletMarginLoss with semi‑hard miner (margin=1.0) 
Few‑shot evaluation via prototypical episodes (EasyFSL TaskSampler) 
Quantization: post‑training dynamic INT8 on nn.Linear 
Speed demo: synthetic episodes throughput
Ablation analysis
