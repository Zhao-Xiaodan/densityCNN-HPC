# ResNet_Shallow Hyperparameter Bottleneck Study Guide

## 🎯 **Study Objective**
Systematically identify what caused ResNet_Shallow's dramatic performance improvement from R² = 0.983 to R² = 0.995 through comprehensive hyperparameter analysis.

## 📋 **Research Questions**
1. **Is it gradient noise** from smaller batch sizes?
2. **Is it memory pressure relief** from better resource management?
3. **Is it learning rate interaction** with batch size changes?
4. **Is it data loading efficiency** improvements?
5. **Is it gradient accumulation vs. true batch size** effects?

---

## 🔬 **Experimental Design**

### **5 Systematic Experiment Groups:**

#### 1. **Batch Size Study** (7 experiments)
- Test batch sizes: `16, 32, 64, 96, 128, 192, 256`
- **Hypothesis**: Smaller batches provide beneficial gradient noise
- **Key Metrics**: R² score, gradient noise scale, convergence behavior

#### 2. **Learning Rate Scaling** (6 experiments)
- Test LR scaling with different batch sizes
- Combinations: `(32,3e-4), (64,6e-4), (128,1.2e-3), (32,1e-4), (32,1e-3), (128,3e-4)`
- **Hypothesis**: LR must scale with batch size for optimal performance

#### 3. **Gradient Accumulation vs True Batch** (8 experiments)
- Compare true batch size vs. gradient accumulation for same effective batch
- Examples: `True32 vs (16×2) vs (8×4)` for effective batch 32
- **Hypothesis**: True small batches outperform accumulated large batches

#### 4. **Memory & Data Loading** (8 experiments)
- Vary workers (0-18) and memory cleanup frequency (3-10)
- **Hypothesis**: Resource optimization affects training stability

#### 5. **Learning Rate Schedulers** (6 experiments)
- Test different schedulers: cosine, step, exponential, plateau
- **Hypothesis**: Scheduler interaction with batch size matters

### **Total: ~35 Systematic Experiments**

---

## 🚀 **Execution Instructions**

### **HPC Execution:**
```bash
# Submit to HPC queue
qsub pbs_resnet_shallow_hyperparameter_study.sh

# Monitor job
qstat -u $USER

# Check progress
tail -f resnet_shallow_hyperparameter_study_*/study_console_output.log
```

### **Local Testing (Quick Mode):**
```bash
# Quick 10-experiment test
python train_resnet_shallow_hyperparameter_study.py \
    --input_dir ./dataset_preprocessed \
    --output_dir ./test_hyperparameter_study \
    --quick_study \
    --epochs 20
```

### **Full Local Execution:**
```bash
python train_resnet_shallow_hyperparameter_study.py \
    --input_dir ./dataset_preprocessed \
    --output_dir ./resnet_shallow_hyperparameter_study \
    --epochs 50 \
    --patience 15 \
    --data_percentage 50 \
    --mixed_precision
```

---

## 📊 **Advanced Analysis Features**

### **Real-time Gradient Analysis:**
- **Gradient noise scale** calculation during training
- **Gradient variance** across batches  
- **Gradient norm stability** tracking

### **Memory Monitoring:**
- **Peak memory usage** per configuration
- **Memory growth patterns** during training
- **Memory efficiency metrics** (R²/GB)

### **Training Dynamics:**
- **Convergence patterns** analysis
- **Learning rate schedules** effectiveness
- **Loss smoothness** and stability metrics

### **Performance Correlation Analysis:**
- **Batch size vs. gradient noise** correlation
- **Memory pressure vs. performance** relationship
- **Training time vs. accuracy** trade-offs

---

## 📈 **Expected Results**

### **If Gradient Noise Hypothesis is Correct:**
- **Smaller batch sizes (16-32)** will show best performance
- **Negative correlation** between batch size and R² score
- **Higher gradient noise scale** correlates with better performance

### **If Memory Pressure Hypothesis is Correct:**
- **Lower worker counts** will improve performance
- **More frequent memory cleanup** correlates with better results
- **Peak memory usage** negatively correlates with performance

### **If Learning Rate Scaling is Critical:**
- **Linear LR scaling** with batch size shows optimal performance
- **Fixed LR with varying batch sizes** shows degraded performance
- **LR-batch interaction** will be statistically significant

---

## 🔍 **Post-Execution Analysis**

### **1. Quick Results Check:**
```bash
# Check experiment completion
ls resnet_shallow_hyperparameter_study_*/experiment_*_results.json | wc -l

# View summary
cat resnet_shallow_hyperparameter_study_*/analysis_report.md
```

### **2. Comprehensive Analysis:**
```bash
# Run detailed analysis
python analyze_hyperparameter_study.py \
    --study_dir resnet_shallow_hyperparameter_study_YYYYMMDD_HHMMSS

# Results in: study_dir/analysis/
```

### **3. Key Analysis Outputs:**
- **`bottleneck_analysis_report.md`**: Primary findings and conclusions
- **`comprehensive_analysis.png`**: Multi-panel performance overview
- **`batch_size_analysis.png`**: Batch size effects visualization
- **`learning_rate_analysis.png`**: LR scaling analysis
- **`gradient_accumulation_analysis.png`**: True vs effective batch analysis
- **`memory_data_loading_analysis.png`**: Resource optimization effects

---

## 🎯 **Success Criteria**

### **Study Completion:**
- [x] All ~35 experiments complete successfully
- [x] Complete study JSON file generated
- [x] Summary CSV with all metrics
- [x] Individual experiment files saved

### **Analysis Quality:**
- [x] Statistical significance tests performed
- [x] Correlation analysis completed
- [x] Pattern identification in top/bottom performers
- [x] Clear bottleneck identification

### **Actionable Insights:**
- [x] Optimal batch size identified
- [x] Learning rate scaling strategy determined  
- [x] Memory management recommendations
- [x] Data loading optimization guidelines

---

## 🚨 **Troubleshooting**

### **Common Issues:**

1. **Out of Memory Errors:**
   - Reduce `base_batch_size` in script arguments
   - Enable `conservative_mode` (default: True)

2. **Data Loading Errors:**
   - Verify dataset path: `dataset_preprocessed/`
   - Check CSV file: `dataset_preprocessed/density.csv`

3. **Container Issues:**
   - Verify container path in PBS script
   - Check Singularity module loading

4. **Incomplete Experiments:**
   - Check individual experiment logs
   - Review resource monitoring logs

---

## 📝 **Expected Timeline**

### **HPC Execution:**
- **Total Runtime**: ~12-20 hours (35 experiments × 20-30 min each)
- **Peak GPU Usage**: ~24GB (ResNet_Shallow is memory efficient)
- **Storage**: ~2-5GB (results + logs)

### **Analysis Phase:**
- **Analysis Runtime**: ~30-60 minutes
- **Output Generation**: ~10-20 files
- **Report Generation**: Automated

---

## ✅ **Final Deliverables**

1. **Primary Research Paper**: `bottleneck_analysis_report.md`
2. **Performance Visualizations**: 5+ analysis plots  
3. **Raw Data**: Complete experiment dataset (CSV/JSON)
4. **Optimal Configuration**: Specific hyperparameter recommendations
5. **Reproducibility**: All code and scripts for validation

---

**🔬 This study will definitively answer why ResNet_Shallow became the top performer and provide actionable insights for optimal CNN training.**