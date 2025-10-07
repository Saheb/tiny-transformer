
![](metrics_plot_1000_steps_2025-10-07_11-37-34.png)

Training output
```

    --- Model Hyperparameters ---
    Max Steps:              1000
    Model Dim (d_model):    128
    FFN Dim (d_ff):         512
    Dropout Rate:           0.2
    Encoder/Decoder Layers: 6
    Attention Heads:        8
    ---------------------------
    
Loaded 17496 samples
Sample pair: {'en': 'Anna Karenina', 'ru': 'Анна Каренина'}
Vocab size: 20000 (capped at 20000)
Vocab size: 20000
Dataset size: 17496
Train size: 13996, Val size: 3500
len(en_tokenized): 13996, len(val_en_tokenized): 3500
Number of train batches per cycle: 437
Number of val batches per cycle: 109
Step 0, Train Loss: 28.3166, Val Loss: 28.7861, Val Accuracy: 0.0000
Step 50, Train Loss: 2.7085, Val Loss: 2.9974, Val Accuracy: 0.7042
Step 100, Train Loss: 2.8334, Val Loss: 2.6232, Val Accuracy: 0.7042
Step 150, Train Loss: 2.6370, Val Loss: 2.5619, Val Accuracy: 0.7042
Step 200, Train Loss: 2.8413, Val Loss: 2.5526, Val Accuracy: 0.7042
Step 250, Train Loss: 2.7146, Val Loss: 2.4896, Val Accuracy: 0.7042
Step 300, Train Loss: 2.8373, Val Loss: 2.4827, Val Accuracy: 0.7042
Step 350, Train Loss: 2.8465, Val Loss: 2.4568, Val Accuracy: 0.7042
Step 400, Train Loss: 2.8874, Val Loss: 2.4512, Val Accuracy: 0.7042
Step 450, Train Loss: 2.7617, Val Loss: 2.5066, Val Accuracy: 0.7042
Step 500, Train Loss: 2.5902, Val Loss: 2.5388, Val Accuracy: 0.7042
Step 550, Train Loss: 2.1935, Val Loss: 2.4718, Val Accuracy: 0.7042
Step 600, Train Loss: 2.5112, Val Loss: 2.4608, Val Accuracy: 0.7042
Step 650, Train Loss: 2.4110, Val Loss: 2.4342, Val Accuracy: 0.7042
Step 700, Train Loss: 2.3642, Val Loss: 2.4376, Val Accuracy: 0.7042
Step 750, Train Loss: 2.6286, Val Loss: 2.4089, Val Accuracy: 0.7102
Step 800, Train Loss: 2.1904, Val Loss: 2.4063, Val Accuracy: 0.7042
Step 850, Train Loss: 2.6101, Val Loss: 2.4071, Val Accuracy: 0.7104
Step 900, Train Loss: 2.1275, Val Loss: 2.4086, Val Accuracy: 0.7044
Step 950, Train Loss: 2.2030, Val Loss: 2.4249, Val Accuracy: 0.7110
Step 1000, Train Loss: 2.0078, Val Loss: 2.4257, Val Accuracy: 0.7114
Max steps reached. Breaking at step 1000
final_steps: 1000, len(losses_post_train): 1001, len(losses_post_val): 21
Training complete. Saving model weights...
✅ Model weights saved to transformer_weights.msgpack
```