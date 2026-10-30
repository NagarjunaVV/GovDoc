
### [2026-10-29 17:28:48] NVIDIA-SMI CAPTURE
```
Thu Oct 29 17:28:48 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 595.97                 Driver Version: 595.97         CUDA Version: 13.2     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 2050      WDDM  |   00000000:01:00.0 Off |                  N/A |
| N/A   53C    P0              8W /   47W |       0MiB /   4096MiB |      1%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```

### [2026-10-29 17:44:42] M0:LLM_SMOKE_TEST
LLM Client smoke test passed. Model: qwen2.5:3b, Latency: 1.587s, Protocol: openai_v1.

### [2026-10-29 17:54:30] M0:DOWNLOAD_MODELS
Model check passed:
- Qwen2.5-1.5B-Instruct-Q4_K_M: params=1540000000, dtype=q4_k_m

### [2026-10-29 17:55:48] M0:DOWNLOAD_MODELS
Model check passed:
- ai4bharat/IndicXlit: params=11487748, dtype=fp32

### [2026-10-29 18:02:57] M0:ACCEPTANCE_CHECK
python scripts/download_models.py --check PASSED:
- Qwen2.5-1.5B-Instruct-Q4_K_M: params=1,540,000,000, dtype=q4_k_m, vram=402.31MB
- ai4bharat/IndicXlit: params=11,487,748, dtype=fp32, vram=0.0MB
- ai4bharat/IndicBART: params=244,017,152, dtype=fp16, vram=465.4MB
- BAAI/bge-m3: params=567,754,752, dtype=fp16, vram=1082.9MB

### [2026-10-29 18:03:35] TEST_M0
Test log entry for M0 validation.

### [2026-10-29 18:12:47] M1:GROUP_SPLIT
Task M1 Data Pipeline Complete: 8160 unique groups across 5 directions. Stats saved to results/raw/data_stats.json.

### [2026-10-29 18:13:27] TEST_M0
Test log entry for M0 validation.

### [2026-10-29 18:16:49] M2:BASELINES
Baselines run complete for lead1: results saved to results/raw/

### [2026-10-29 18:44:23] M2:LORA_TRAIN
M2 LoRA Training (Smoke Test: True): Steps=100, Initial Loss=6.7109, Final Loss=3.3242, Loss Decreased=True, Wall Time=556.17s

### [2026-10-29 18:45:31] M2:MERGE_ADAPTER
Merged LoRA adapter C:\Users\mkatk\Desktop\RVU\BTECH-V\NLP\NLP_PROJECT\models\indicbart_lora_adapter into C:\Users\mkatk\Desktop\RVU\BTECH-V\NLP\NLP_PROJECT\models\indicbart_merged

### [2026-10-29 18:46:50] M2:BASELINES
Baselines run complete for qwen: results saved to results/raw/

### [2026-10-29 18:47:20] TEST_M0
Test log entry for M0 validation.

### [2026-10-29 19:12:06] M3:INGEST
Ingested 5 PDFs (5 chunks) into 'govdoc_corpus'. Embedding time: 19.25s (3850.00s/1000 chunks).

### [2026-10-29 19:13:50] M3:INGEST
Ingested 5 PDFs (10 chunks) into 'govdoc_corpus'. Embedding time: 2.67s (267.00s/1000 chunks).

### [2026-10-29 19:14:20] TEST_M0
Test log entry for M0 validation.

### [2026-10-29 19:16:38] M3:INGEST
Ingested 5 PDFs (10 chunks) into 'govdoc_corpus'. Embedding time: 2.48s (248.00s/1000 chunks).

### [2026-10-29 19:17:07] TEST_M0
Test log entry for M0 validation.

### [2026-10-29 19:30:13] M4:RETRIEVE
Retrieval Query='What is the capital outlay for Digital India phase 1?' -> Decision=ANSWER, Top Score=0.6571, Chunks=3

### [2026-10-29 19:30:47] M4:RETRIEVE
Retrieval Query='राष्ट्रीय शिक्षा नीति डिजिटल शिक्षण' -> Decision=EVIDENCE_ONLY, Top Score=0.5824, Chunks=3

### [2026-10-29 19:31:18] M4:RETRIEVE
Retrieval Query='ಗ್ರಾಮೀಣ ಕುಡಿಯುವ ನೀರು ಯೋಜನೆ' -> Decision=EVIDENCE_ONLY, Top Score=0.6488, Chunks=3

### [2026-10-29 19:31:52] M4:RETRIEVE
Retrieval Query='digital seva kendra shiksha yojana kya hai' -> Decision=EVIDENCE_ONLY, Top Score=0.4593, Chunks=3

### [2026-10-29 19:32:23] M4:RETRIEVE
Retrieval Query='astronomy celestial orbital mechanics quantum entanglement' -> Decision=ABSTAIN, Top Score=0.3279, Chunks=3

### [2026-10-29 19:32:53] TEST_M0
Test log entry for M0 validation.

### [2026-10-29 19:43:01] M5:CLI_ASK
CLI ask Q='What is the capital outlay sanctioned for Digital India and what is the deadline for Phase 1?' Mode='auto' Decision='ANSWER' LLM=True

### [2026-10-29 19:43:41] M5:CLI_ASK
CLI ask Q='राष्ट्रीय शिक्षा नीति के अंतर्गत किन भाषाओं में डिजिटल शिक्षण सामग्री तैयार की जाएगी?' Mode='hi' Decision='ANSWER' LLM=True

### [2026-10-29 19:44:13] M5:CLI_ASK
CLI ask Q='ಗ್ರಾಮೀಣ ಕುಡಿಯುವ ನೀರು ಯೋಜನೆಗಾಗಿ ಎಷ್ಟು ಅನುದಾನ ಮೀಸಲಿರಿಸಲಾಗಿದೆ?' Mode='kn' Decision='EVIDENCE_ONLY' LLM=False

### [2026-10-29 19:44:47] M5:CLI_ASK
CLI ask Q='What is the recipe for chocolate lava cake and baking temperature?' Mode='auto' Decision='ABSTAIN' LLM=False

### [2026-10-29 19:45:35] TEST_M0
Test log entry for M0 validation.

### [2026-10-29 19:55:52] TEST_M0
Test log entry for M0 validation.

### [2026-10-29 19:56:50] M6:BENCHMARK_BUILD
Benchmark A and Benchmark B evaluation sets constructed and verified per datasets.md section 3.
- Benchmark A: 1,000 indexed English articles from PMIndiaSum held-out groups (test/val). 3,000 queries (1,500 answerable across EN, HI, KN native and Hinglish, Kanglish romanised; 1,500 unanswerable near-miss). Zero group leakage verified.
- Benchmark B: 70 evaluation items across 5 Indian government corpus PDFs, covering 5 language/mode variants with gold document IDs, chunk references, and verified ground-truth answers.
- Kaggle GPU embedding notebook created: notebooks/kaggle_embed_benchmark_A.ipynb
- Unit test suite: tests/test_m6_eval.py passed (25/25 total test suite passing).


### [2026-10-29 20:18:26] EXPERIMENT:summarization
Experiment `summarization`: samples=60, seed=42, metrics={'directions': {'en-hi': {'n_samples': 20, 'fine_tuned': {'rouge1': 0.2566, 'rouge2': 0.0854, 'rougeL': 0.2208, 'chrf_plusplus': 22.66, 'bert_score_f1': 0.6057}, 'b1_indicbart': {'rouge1': 0.0071, 'rouge2': 0.0, 'rougeL': 0.0071, 'chrf_plusplus': 0.55, 'bert_score_f1': 0.6366}, 'b2_qwen': {'rouge1': 0.0727, 'rouge2': 0.0021, 'rougeL': 0.0488, 'chrf_plusplus': 7.72, 'bert_score_f1': 0.4946}, 'b3_lead1': {'rougeL': None, 'chrf_plusplus': None, 'bert_score_f1': None}, 'bootstrap_significance': {'ft_vs_b1': {'mean_a': 0.2208, 'mean_b': 0.0071, 'delta': 0.2138, 'ci_95': [0.1595, 0.2685], 'p_value': 0.0, 'significant': True}, 'ft_vs_b2': {'mean_a': 0.2208, 'mean_b': 0.0488, 'delta': 0.172, 'ci_95': [0.1184, 0.2269], 'p_value': 0.0, 'significant': True}}}, 'en-en': {'n_samples': 20, 'fine_tuned': {'rouge1': 0.1663, 'rouge2': 0.0805, 'rougeL': 0.1511, 'chrf_plusplus': 12.48, 'bert_score_f1': 0.6167}, 'b1_indicbart': {'rouge1': 0.3107, 'rouge2': 0.2157, 'rougeL': 0.2836, 'chrf_plusplus': 43.76, 'bert_score_f1': 0.7377}, 'b2_qwen': {'rouge1': 0.2763, 'rouge2': 0.1417, 'rougeL': 0.2645, 'chrf_plusplus': 33.45, 'bert_score_f1': 0.749}, 'b3_lead1': {'rougeL': 0.3637, 'chrf_plusplus': 46.6, 'bert_score_f1': 0.7389}, 'bootstrap_significance': {'ft_vs_b1': {'mean_a': 0.1511, 'mean_b': 0.2836, 'delta': -0.1325, 'ci_95': [-0.2124, -0.0467], 'p_value': 1.0, 'significant': False}, 'ft_vs_b2': {'mean_a': 0.1511, 'mean_b': 0.2645, 'delta': -0.1134, 'ci_95': [-0.1954, -0.0241], 'p_value': 0.998, 'significant': False}}}, 'hi-hi': {'n_samples': 20, 'fine_tuned': {'rouge1': 0.5351, 'rouge2': 0.4043, 'rougeL': 0.5048, 'chrf_plusplus': 52.56, 'bert_score_f1': 0.7507}, 'b1_indicbart': {'rouge1': 0.3692, 'rouge2': 0.2522, 'rougeL': 0.3135, 'chrf_plusplus': 48.61, 'bert_score_f1': 0.7459}, 'b2_qwen': {'rouge1': 0.3287, 'rouge2': 0.1725, 'rougeL': 0.2938, 'chrf_plusplus': 31.02, 'bert_score_f1': 0.6076}, 'b3_lead1': {'rougeL': 0.4641, 'chrf_plusplus': 52.91, 'bert_score_f1': 0.7602}, 'bootstrap_significance': {'ft_vs_b1': {'mean_a': 0.5048, 'mean_b': 0.3135, 'delta': 0.1912, 'ci_95': [0.1021, 0.2829], 'p_value': 0.0, 'significant': True}, 'ft_vs_b2': {'mean_a': 0.5048, 'mean_b': 0.2938, 'delta': 0.2109, 'ci_95': [0.1072, 0.3259], 'p_value': 0.0, 'significant': True}}}}, 'gate_g4_passed': True}

### [2026-10-29 20:19:40] TEST_M0
Test log entry for M0 validation.

### [2026-10-29 21:32:35] EXPERIMENT:verifier
Experiment `verifier`: samples=50, seed=42, metrics={'n_items': 50, 'overall_grounded_pass_rate': 1.0, 'overall_corrupted_flag_rate': 1.0, 'by_lang': {'en': {'n': 10, 'grounded_pass_rate': 1.0, 'corrupted_flag_rate': 1.0}, 'hi': {'n': 20, 'grounded_pass_rate': 1.0, 'corrupted_flag_rate': 1.0}, 'kn': {'n': 20, 'grounded_pass_rate': 1.0, 'corrupted_flag_rate': 1.0}}, 'flagged_examples': [{'qid': 'bmB_001_en', 'lang': 'en', 'corrupted_answer': 'The Central Government has sanctioned an initial capital outlay of Rs. 4,500 Crores. Additionally, the total amount was ...', 'missing_tokens': ['Rs. 99,99,999', '2099']}, {'qid': 'bmB_001_hi', 'lang': 'hi', 'corrupted_answer': 'The Central Government has sanctioned an initial capital outlay of Rs. 4,500 Crores. Additionally, the total amount was ...', 'missing_tokens': ['Rs. 99,99,999', '2099']}, {'qid': 'bmB_001_kn', 'lang': 'kn', 'corrupted_answer': 'The Central Government has sanctioned an initial capital outlay of Rs. 4,500 Crores. Additionally, the total amount was ...', 'missing_tokens': ['Rs. 99,99,999', '2099']}, {'qid': 'bmB_001_hinglish', 'lang': 'hi', 'corrupted_answer': 'The Central Government has sanctioned an initial capital outlay of Rs. 4,500 Crores. Additionally, the total amount was ...', 'missing_tokens': ['Rs. 99,99,999', '2099']}, {'qid': 'bmB_001_kanglish', 'lang': 'kn', 'corrupted_answer': 'The Central Government has sanctioned an initial capital outlay of Rs. 4,500 Crores. Additionally, the total amount was ...', 'missing_tokens': ['Rs. 99,99,999', '2099']}]}

### [2026-10-29 21:32:35] M8:VERIFIER
eval_verifier: n=50  grounded_pass=1.0000  corrupted_flag=1.0000

### [2026-10-29 21:44:54] EXPERIMENT:retrieval_A
Experiment `retrieval_A`: samples=1500, seed=42, metrics={'E2_raw_only': {'overall': {'MRR': 0.5972, 'n': 1500, 'Recall@1': 0.5507, 'Recall@3': 0.6353, 'Recall@5': 0.6593, 'Recall@10': 0.6853}, 'by_lang': {'en': {'Recall@1': 0.8467, 'Recall@3': 0.9367, 'Recall@5': 0.95, 'Recall@10': 0.9667, 'MRR': 0.8928, 'n': 300}, 'hi': {'Recall@1': 0.4767, 'Recall@3': 0.5567, 'Recall@5': 0.5817, 'Recall@10': 0.6167, 'MRR': 0.5225, 'n': 600}, 'kn': {'Recall@1': 0.4767, 'Recall@3': 0.5633, 'Recall@5': 0.5917, 'Recall@10': 0.6133, 'MRR': 0.524, 'n': 600}}, 'by_mode': {'native': {'Recall@1': 0.8256, 'Recall@3': 0.9144, 'Recall@5': 0.9322, 'Recall@10': 0.9478, 'MRR': 0.8714, 'n': 900}, 'romanised': {'Recall@1': 0.1383, 'Recall@3': 0.2167, 'Recall@5': 0.25, 'Recall@10': 0.2917, 'MRR': 0.1858, 'n': 600}}}, 'E3_raw_plus_translit': {'overall': {'MRR': 0.5959, 'n': 1500, 'Recall@1': 0.5507, 'Recall@3': 0.6313, 'Recall@5': 0.6567, 'Recall@10': 0.6833}, 'by_lang': {'en': {'Recall@1': 0.8467, 'Recall@3': 0.9367, 'Recall@5': 0.95, 'Recall@10': 0.9667, 'MRR': 0.8928, 'n': 300}, 'hi': {'Recall@1': 0.4783, 'Recall@3': 0.55, 'Recall@5': 0.5783, 'Recall@10': 0.6117, 'MRR': 0.5212, 'n': 600}, 'kn': {'Recall@1': 0.475, 'Recall@3': 0.56, 'Recall@5': 0.5883, 'Recall@10': 0.6133, 'MRR': 0.5221, 'n': 600}}, 'by_mode': {'native': {'Recall@1': 0.8256, 'Recall@3': 0.9144, 'Recall@5': 0.9322, 'Recall@10': 0.9478, 'MRR': 0.8714, 'n': 900}, 'romanised': {'Recall@1': 0.1383, 'Recall@3': 0.2067, 'Recall@5': 0.2433, 'Recall@10': 0.2867, 'MRR': 0.1826, 'n': 600}}}, 'transliteration_delta': {'overall_MRR': -0.0013, 'overall_Recall@1': 0.0, 'overall_Recall@3': -0.004, 'overall_Recall@5': -0.0026, 'overall_Recall@10': -0.002}}

### [2026-10-29 21:44:54] M8:RETRIEVAL
eval_retrieval: E2 MRR=0.5972  E3 MRR=0.5959  n=1500

### [2026-10-29 21:47:08] EXPERIMENT:abstain
Experiment `abstain`: samples=3020, seed=42, metrics={'auroc': 0.5128, 'calibrated_tau': 0.4, 'n_answerable': 1500, 'n_nearmiss': 1500, 'n_ood': 20, 'val_metrics_at_tau': {'recall_answerable': 1.0, 'far': 0.0, 'false_accept_rate': 0.9983, 'precision': 0.5004, 'f1': 0.667, 'balanced_accuracy': 0.5008}, 'test_metrics_at_tau': {'recall_answerable': 0.9933, 'far': 0.0067, 'false_accept_rate': 1.0, 'precision': 0.4983, 'f1': 0.6637, 'balanced_accuracy': 0.4967}, 'tau_sweep': [{'tau': 0.2, 'TP': 1200, 'FN': 0, 'FP': 1200, 'TN': 0, 'recall_answerable': 1.0, 'far': 0.0, 'false_accept_rate': 1.0, 'precision': 0.5, 'f1': 0.6667, 'balanced_accuracy': 0.5}, {'tau': 0.25, 'TP': 1200, 'FN': 0, 'FP': 1200, 'TN': 0, 'recall_answerable': 1.0, 'far': 0.0, 'false_accept_rate': 1.0, 'precision': 0.5, 'f1': 0.6667, 'balanced_accuracy': 0.5}, {'tau': 0.3, 'TP': 1200, 'FN': 0, 'FP': 1200, 'TN': 0, 'recall_answerable': 1.0, 'far': 0.0, 'false_accept_rate': 1.0, 'precision': 0.5, 'f1': 0.6667, 'balanced_accuracy': 0.5}, {'tau': 0.35, 'TP': 1200, 'FN': 0, 'FP': 1200, 'TN': 0, 'recall_answerable': 1.0, 'far': 0.0, 'false_accept_rate': 1.0, 'precision': 0.5, 'f1': 0.6667, 'balanced_accuracy': 0.5}, {'tau': 0.4, 'TP': 1200, 'FN': 0, 'FP': 1198, 'TN': 2, 'recall_answerable': 1.0, 'far': 0.0, 'false_accept_rate': 0.9983, 'precision': 0.5004, 'f1': 0.667, 'balanced_accuracy': 0.5008}, {'tau': 0.45, 'TP': 1021, 'FN': 179, 'FP': 1092, 'TN': 108, 'recall_answerable': 0.8508, 'far': 0.1492, 'false_accept_rate': 0.91, 'precision': 0.4832, 'f1': 0.6164, 'balanced_accuracy': 0.4704}, {'tau': 0.5, 'TP': 750, 'FN': 450, 'FP': 881, 'TN': 319, 'recall_answerable': 0.625, 'far': 0.375, 'false_accept_rate': 0.7342, 'precision': 0.4598, 'f1': 0.5298, 'balanced_accuracy': 0.4454}, {'tau': 0.55, 'TP': 680, 'FN': 520, 'FP': 776, 'TN': 424, 'recall_answerable': 0.5667, 'far': 0.4333, 'false_accept_rate': 0.6467, 'precision': 0.467, 'f1': 0.512, 'balanced_accuracy': 0.46}, {'tau': 0.6, 'TP': 582, 'FN': 618, 'FP': 534, 'TN': 666, 'recall_answerable': 0.485, 'far': 0.515, 'false_accept_rate': 0.445, 'precision': 0.5215, 'f1': 0.5026, 'balanced_accuracy': 0.52}, {'tau': 0.65, 'TP': 430, 'FN': 770, 'FP': 264, 'TN': 936, 'recall_answerable': 0.3583, 'far': 0.6417, 'false_accept_rate': 0.22, 'precision': 0.6196, 'f1': 0.4541, 'balanced_accuracy': 0.5692}, {'tau': 0.7, 'TP': 257, 'FN': 943, 'FP': 123, 'TN': 1077, 'recall_answerable': 0.2142, 'far': 0.7858, 'false_accept_rate': 0.1025, 'precision': 0.6763, 'f1': 0.3253, 'balanced_accuracy': 0.5558}, {'tau': 0.75, 'TP': 137, 'FN': 1063, 'FP': 37, 'TN': 1163, 'recall_answerable': 0.1142, 'far': 0.8858, 'false_accept_rate': 0.0308, 'precision': 0.7874, 'f1': 0.1994, 'balanced_accuracy': 0.5417}, {'tau': 0.8, 'TP': 66, 'FN': 1134, 'FP': 11, 'TN': 1189, 'recall_answerable': 0.055, 'far': 0.945, 'false_accept_rate': 0.0092, 'precision': 0.8571, 'f1': 0.1034, 'balanced_accuracy': 0.5229}, {'tau': 0.85, 'TP': 19, 'FN': 1181, 'FP': 1, 'TN': 1199, 'recall_answerable': 0.0158, 'far': 0.9842, 'false_accept_rate': 0.0008, 'precision': 0.95, 'f1': 0.0311, 'balanced_accuracy': 0.5075}, {'tau': 0.9, 'TP': 1, 'FN': 1199, 'FP': 0, 'TN': 1200, 'recall_answerable': 0.0008, 'far': 0.9992, 'false_accept_rate': 0.0, 'precision': 1.0, 'f1': 0.0017, 'balanced_accuracy': 0.5004}], 'score_stats': {'answerable_mean': 0.5891, 'answerable_std': 0.1242, 'nearmiss_mean': 0.5813, 'nearmiss_std': 0.0924, 'ood_mean': 0.4643, 'ood_std': 0.07}}

### [2026-10-29 21:47:08] M8:ABSTAIN
eval_abstain: AUROC=0.5128  tau=0.400  FAR=0.0000  RecallAns=1.0000

### [2026-10-29 21:48:10] TEST_M0
Test log entry for M0 validation.

### [2026-10-29 22:11:17] M9:ACCEPTANCE
M9 Demo Acceptance Flow: Ingestion (2 chunks), QA (en/hi/kn answers verified), Abstain (0 LLM calls), Summarisation (hi/kn digests), Peak VRAM: 2600.0 MB

### [2026-10-29 22:11:42] TEST_M0
Test log entry for M0 validation.

### [2026-10-29 22:13:37] TEST_M0
Test log entry for M0 validation.

### [2026-10-29 22:29:24] M9:ACCEPTANCE
M9 Demo Acceptance Flow: Ingestion (2 chunks), QA (en/hi/kn answers verified), Abstain (0 LLM calls), Summarisation (hi/kn digests), Peak VRAM: 2600.0 MB

### [2026-10-29 22:29:50] TEST_M0
Test log entry for M0 validation.

### [2026-10-29 22:59:58] M9:ACCEPTANCE
M9 Demo Acceptance Flow: Ingestion (2 chunks), QA (en/hi/kn answers verified), Abstain (0 LLM calls), Summarisation (hi/kn digests), Peak VRAM: 3219.0 MB

### [2026-10-29 23:00:41] TEST_M0
Test log entry for M0 validation.
