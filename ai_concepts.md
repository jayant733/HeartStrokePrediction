# AI Concepts Glossary

## Transformer
A Transformer is a deep learning architecture based on self-attention, which allows the model to understand relationships between tokens in a sequence. In self-attention, the model uses Query, Key, and Value vectors to determine how much focus to place on other tokens when processing a given token.

## LoRA and QLoRA
- **LoRA (Low-Rank Adaptation):** A parameter-efficient fine-tuning method that injects trainable low-rank matrices into transformer layers while keeping the original weights frozen.
- **QLoRA:** An extension of LoRA that combines quantization (like 4-bit precision) with LoRA fine-tuning, enabling fine-tuning of large models with drastically reduced memory requirements.

## RAG (Retrieval-Augmented Generation)
RAG is a technique that enhances large language models by retrieving relevant information from an external knowledge base before generating a response, thereby grounding the output in factual and up-to-date information.

## Agentic AI and ReAct
- **Agentic AI:** AI systems designed to act autonomously (or semi-autonomously) to achieve specific goals. They can perceive their environment, make decisions, and execute actions using tools.
- **ReAct (Reason and Act):** A prompting framework for agentic AI that interleaves reasoning (thinking about what to do) and acting (using tools or interacting with the environment) to solve complex tasks step by step.

## Guardrails in AI Agents
Guardrails are safety mechanisms and constraints placed on AI agents to ensure they operate within predefined ethical, operational, and security boundaries. They prevent agents from taking harmful, unauthorized, or unintended actions.

## Agent Frameworks (LangChain, LangGraph, CrewAI)
- **LangChain:** A framework designed to simplify the creation of applications using large language models, providing tools for chaining prompts, memory, and agents.
- **LangGraph:** An extension of LangChain for building stateful, multi-actor applications with cyclic computational graphs, ideal for complex agentic workflows.
- **CrewAI:** A framework for orchestrating role-playing, autonomous AI agents. It enables multiple agents to work together collaboratively to achieve a common goal.

## Reasoning and Planning (Chain-of-Thought, Plan-and-Execute)
- **Chain-of-Thought (CoT):** A prompting strategy where the model is encouraged to explicitly output intermediate reasoning steps before arriving at a final answer, which improves performance on complex reasoning tasks.
- **Plan-and-Execute:** An agentic architecture where the agent first creates a comprehensive multi-step plan to solve a problem, and then executes the steps sequentially, often evaluating progress along the way.

## Human-in-the-loop (HITL)
Human-in-the-loop is a design pattern where human intervention, feedback, or approval is required at critical decision points during an AI system's operation, ensuring safety, accuracy, and alignment with human intent.

## Model Monitoring
Model monitoring involves tracking the performance, behavior, and health of machine learning models in production. It includes checking for data drift, concept drift, latency, error rates, and output quality to ensure the model remains reliable over time.

## Unsupervised Learning & Patient Phenotyping (Clustering-Based Prescriptions)
- **Unsupervised Learning:** A class of machine learning algorithms that discover hidden patterns, geometric structures, or natural groupings in unlabelled multidimensional data without explicit supervision (target labels).
- **Patient Phenotyping & Pattern Grouping:** In clinical healthcare AI, unsupervised learning (e.g., K-Means, Agglomerative Hierarchical Clustering, DBSCAN) partitions heterogeneous patient populations into distinct clinical archetypes/phenotypes based on multivariate physiological, demographic, and lifestyle features (e.g., arterial stiffness, metabolic stress, nicotine exposure, age resilience).
- **Cluster-Driven Targeted Prescriptions:** Rather than relying solely on binary classification (stroke vs. no stroke), clinical decision-support systems first group the patient into their corresponding unsupervised phenotype, and then retrieve specialized targeted pharmacotherapy protocols (such as ARBs/ACE inhibitors for vascular stiffness cohorts, GLP-1/metformin for metabolic cohorts, or smoking cessation regimens for workplace stress cohorts) and tailored diagnostic workups.

## Z-Test
- A parametric hypothesis test used to determine whether there is a statistically significant difference between two population proportions or between a sample mean and a known population mean when the sample size is large ($n \ge 30$) and the variance is known (or approximated via the Central Limit Theorem).
- **Two-Sample Z-Test for Proportions:** Used to determine whether two cohorts exhibit different event rates (e.g., comparing the stroke incidence of an unsupervised patient cluster against the general population baseline):
  $$Z = \frac{\hat{p}_1 - \hat{p}_2}{\sqrt{\hat{p}(1-\hat{p})\left(\frac{1}{n_1} + \frac{1}{n_2}\right)}}$$
  where $\hat{p}$ is the pooled proportion. A large absolute Z-score ($|Z| > 1.96$ for $\alpha = 0.05$) indicates that the observed disparity between cohorts is statistically significant and unlikely to be due to chance.

## P-Value and Hypothesis Decision Rule (P-Test)
- **P-Value:** In statistical hypothesis testing, the p-value is the probability of observing a test statistic at least as extreme as the value calculated from the sample data, assuming that the null hypothesis ($H_0$) is true.
- **Null ($H_0$) vs. Alternative ($H_1$) Hypothesis:**
  - $H_0$: There is no significant difference between the groups (any observed difference is due to random sampling variability).
  - $H_1$: There is a statistically significant difference between the groups.
- **Decision Rule:**
  - If $p < \alpha$ (commonly $\alpha = 0.05$ or $0.01$): Reject $H_0$ — the evidence against the null hypothesis is statistically significant.
  - If $p \ge \alpha$: Fail to reject $H_0$ — there is insufficient statistical evidence to claim a significant difference.

## F-Test (ANOVA & Variance Ratio)
- An omnibus parametric test based on the F-distribution that evaluates whether the variances of two populations are equal, or whether the means of three or more groups differ significantly:
  - **One-Way ANOVA F-Test:** Tests the equality of means across multiple unsupervised clusters (e.g., comparing mean glucose, age, or BMI across 4 distinct clinical phenotypes simultaneously):
    $$F = \frac{\text{MS}_{\text{between}}}{\text{MS}_{\text{within}}} = \frac{\text{SS}_{\text{between}} / (k - 1)}{\text{SS}_{\text{within}} / (N - k)}$$
    A statistically significant F-statistic ($p < 0.05$) proves that the unsupervised clustering algorithm has identified distinct, well-separated clinical patient cohorts rather than arbitrary random clusters.
  - **Fisher's F-Test for Equality of Variances:** Assesses whether the dispersion or spread of a biomarker differs significantly between two clinical cohorts ($F = s_1^2 / s_2^2$).

## Log Transformation & Statistical Audit Logging (Log)
- **Log Transformation ($\ln$ or $\log_{10}$):** Biological and clinical biomarkers (such as fasting blood glucose, BMI, and triglyceride levels) typically follow a right-skewed, log-normal distribution. Applying a logarithmic transformation ($y = \ln(x)$) compresses the long right tail, normalizes the distribution, and stabilizes variance, satisfying the normality and homoscedasticity assumptions required for reliable parametric tests (Z-tests and F-tests).
- **Statistical Audit Logging:** A regulatory and clinical ML governance practice where every automated hypothesis test, data drift detection run, test statistic ($Z, F$), degrees of freedom, p-value, and hypothesis decision is persistently logged with timestamps and execution metadata. This ensures traceability, reproducibility, and audit compliance for AI-driven clinical decision-support systems.

## Sampling Methods in Statistics and Machine Learning

Sampling is the process of selecting a representative subset of observations from a target population to estimate population parameters, train predictive machine learning models, or overcome acute data challenges (such as severe class imbalance). In clinical and biomedical data science (like stroke risk prediction), choosing the correct sampling strategy is critical to ensure model generalizability, avoid algorithmic bias, and eliminate the "accuracy paradox."

### 1. Probability Sampling Methods (Random & Statistical Selection)
In probability sampling, every individual in the population has a known, non-zero probability of being selected. This eliminates selection bias and enables valid inferential statistics (e.g., Z-tests, ANOVA F-tests, confidence intervals).

- **Simple Random Sampling (SRS):**
  - *Mechanism:* Every member of the population has an identical and independent probability of selection ($P = n / N$).
  - *Types:* With Replacement (SRSWR, where items can be selected multiple times) vs. Without Replacement (SRSWOR, standard in data splitting).
  - *Clinical Use-case:* Selecting an unbiased control group from a registry when the population is homogeneous.

- **Stratified Random Sampling:**
  - *Mechanism:* The population is divided into mutually exclusive, exhaustive subgroups called **strata** based on key attributes (e.g., stroke status, hypertension, gender, age brackets). Independent random samples are then drawn from each stratum.
  - *Proportional Stratification:* Sample sizes within strata are proportional to stratum sizes in the population ($n_h = n \times \frac{N_h}{N}$).
  - *Disproportional Stratification:* Deliberately oversamples smaller strata (e.g., stroke cases) to ensure sufficient statistical power.
  - *Clinical Use-case:* Generating train/test splits where minority stroke cases ($\approx 4.8\%$) are guaranteed to be represented in identical proportions in both training and test sets.

- **Systematic Sampling:**
  - *Mechanism:* Elements are chosen at fixed, regular periodic intervals from an ordered list ($k = N / n$). The starting element is chosen randomly between index $1$ and $k$.
  - *Clinical Use-case:* Quality-auditing every 10th patient record admitted to an emergency neurology department over a fiscal quarter.

- **Cluster Sampling:**
  - *Mechanism:* The heterogeneous population is naturally divided into clusters or primary sampling units (e.g., hospitals, clinics, geographical districts). A random sample of *entire clusters* is selected, and all individuals within chosen clusters are surveyed (Single-Stage) or randomly sampled (Two-Stage).
  - *Difference from Stratified:* In Stratified, *some* elements from *all* strata are sampled (homogeneous within, heterogeneous between). In Cluster, *all* elements from *some* clusters are sampled (heterogeneous within, homogeneous between).
  - *Clinical Use-case:* Clinical trials evaluating hospital protocol compliance across 50 regional healthcare networks.

- **Multi-Stage Sampling:**
  - *Mechanism:* Combines multiple probability sampling techniques in hierarchical stages (e.g., Stage 1: Cluster sample of hospital networks; Stage 2: Stratified sample of outpatient clinics; Stage 3: Simple random sample of patient EHRs).

---

### 2. Non-Probability Sampling Methods (Non-Random Selection)
In non-probability sampling, elements are selected based on subjective judgment, accessibility, or specific quotas rather than random chance. Probabilities of selection cannot be computed, precluding formal parametric margin-of-error calculations.

- **Convenience Sampling:**
  - *Mechanism:* Selecting individuals who are most easily accessible to the researcher (e.g., recruiting stroke survey participants exclusively from a hospital lobby). High risk of severe selection bias.

- **Purposive / Judgmental Sampling:**
  - *Mechanism:* The researcher deliberately selects subjects based on expert knowledge and predefined clinical criteria (e.g., selecting only patients with a confirmed ischemic stroke undergoing thrombectomy for an intensive neuro-rehab study).

- **Quota Sampling:**
  - *Mechanism:* The non-random equivalent of stratified sampling. Specific quotas are set for subgroups (e.g., 50 male stroke patients and 50 female stroke patients), but individuals within each quota are recruited via convenience.

- **Snowball / Referral Sampling:**
  - *Mechanism:* Existing study participants refer or recruit future participants from their personal networks. Indispensable for rare diseases, hidden populations, or stigmatized health conditions.

---

### 3. Imbalanced Class Resampling Techniques (Healthcare & Stroke AI)
In real-world healthcare datasets, positive outcomes (like stroke occurrence) are heavily outnumbered by negative outcomes (e.g., stroke prevalence $\approx 4.8\%$, giving a 1:20 imbalance ratio). Standard machine learning algorithms optimizing overall accuracy fall victim to the **Accuracy Paradox** (predicting non-stroke for $100\%$ of patients yields $95.2\%$ accuracy but $0\%$ sensitivity/recall). Resampling techniques rebalance the class priors:

#### A. Oversampling Techniques (Increasing Minority Class Instances)
- **Random Oversampling (ROS):**
  - Duplicates existing minority stroke cases at random until parity is reached.
  - *Drawback:* Leads to overfitting because the model memorizes exact duplicates, creating overly rigid decision boundaries.
- **SMOTE (Synthetic Minority Over-sampling Technique):**
  - Creates synthetic, interpolated minority instances rather than copying existing ones.
  - *Mathematical Formula:* For a minority sample $x_i$, find its $k$-nearest minority neighbors ($x_{zi}$). Generate a synthetic instance:
    $$x_{\text{new}} = x_i + \lambda \times (x_{zi} - x_i) \quad \text{where } \lambda \sim \text{Uniform}(0, 1)$$
  - *Benefit:* Expands the decision boundary convex hull and prevents duplicate overfitting.
- **Borderline-SMOTE:**
  - Identifies "borderline" minority instances that lie dangerously close to the decision boundary (where $m$ of their $k$ nearest neighbors belong to the majority class).
  - *Borderline-1:* Generates synthetic samples along lines connecting borderline minority instances to other minority neighbors.
  - *Borderline-2:* Generates synthetic samples along lines connecting to both minority and majority neighbors.
- **ADASYN (Adaptive Synthetic Sampling):**
  - An extension of SMOTE that adaptively weights minority samples according to their learning difficulty. Minority samples surrounded by more majority instances receive a higher synthetic sampling density, forcing the classifier to focus on difficult decision areas.

#### B. Undersampling Techniques (Decreasing Majority Class Instances)
- **Random Undersampling (RUS):**
  - Randomly discards majority non-stroke instances until desired balance is achieved.
  - *Drawback:* May discard valuable, rich clinical variance and information.
- **Tomek Links:**
  - A Tomek link is a pair of minimally distant instances $(x_i, x_j)$ where $x_i$ and $x_j$ belong to different classes.
  - *Mechanism:* Discarding the majority class instance from each Tomek link cleans ambiguous overlap and sharpens class boundaries.
- **Edited Nearest Neighbors (ENN):**
  - Removes majority instances whose class label disagrees with the majority of their $k$-nearest neighbors, removing noisy borderline points.
- **NearMiss (NearMiss-1, 2, and 3):**
  - *NearMiss-1:* Retains majority samples with the smallest average distance to the $k$-closest minority instances.
  - *NearMiss-2:* Retains majority samples with the smallest average distance to the $k$-farthest minority instances.
  - *NearMiss-3:* Retains a fixed number of majority instances closest to each minority instance.

#### C. Hybrid / Combined Resampling
- **SMOTE + Tomek Links (SMOTETomek):**
  - Applies SMOTE to synthesize minority instances, followed by Tomek Link detection to scrub ambiguous, overlapping boundary pairs. Highly recommended for clinical datasets.
- **SMOTE + ENN (SMOTEENN):**
  - Applies SMOTE oversampling followed by Edited Nearest Neighbors noise cleaning. Generates deeper, cleaner class separations than standard SMOTE.

---

### 4. Model Evaluation & Validation Resampling Methods
Resampling techniques used during model training and evaluation to obtain unbiased, robust performance estimates (AUC-ROC, Sensitivity, Specificity, F1-Score):

- **Stratified K-Fold Cross-Validation:**
  - Partitions data into $K$ equal-sized folds while strictly maintaining the original target class ratio ($\approx 4.8\%$ stroke in every fold). The model trains on $K-1$ folds and validates on the held-out fold, repeated $K$ times. Standard gold-standard evaluation for imbalanced healthcare AI.
- **Bootstrap Resampling (Bootstrapping / Bagging):**
  - Given a dataset of size $N$, draw $N$ samples *with replacement*.
  - *Statistical Guarantee:*
    $$P(\text{not selected}) = \left(1 - \frac{1}{N}\right)^N \xrightarrow[N \to \infty]{} \frac{1}{e} \approx 0.368 \ (36.8\% \text{ Out-of-Bag / OOB})$$
    $$P(\text{selected at least once}) \approx 1 - 0.368 = 0.632 \ (63.2\% \text{ In-Bag})$$
  - The $36.8\%$ Out-of-Bag (OOB) samples act as an independent, untouched validation test set, foundational to ensemble models like Random Forests.
- **Leave-One-Out Cross-Validation (LOOCV):**
  - An extreme case of K-Fold where $K = N$. Every single observation serves as a held-out test set once while training on the remaining $N-1$ points. Deterministic and variance-free, but computationally prohibitive for large datasets.
- **Monte Carlo / Repeated Random Sub-Sampling:**
  - Randomly splits data into training and validation sets multiple ($B$) times. Unlike K-Fold, the proportion of train/test is independent of the number of iterations, though some observations may never be validated while others are chosen multiple times.

---

### 5. Summary Matrix of Sampling Methods

| Category | Method | Key Mechanism | Best Healthcare / ML Use-Case |
| :--- | :--- | :--- | :--- |
| **Probability** | Simple Random (SRS) | Uniform $P = n/N$ selection | Baseline benchmarking on homogeneous registries |
| **Probability** | Stratified Sampling | Proportional sampling within subgroups | Imbalanced train/test split preservation |
| **Probability** | Systematic Sampling | Select every $k = N/n$-th record | Real-time EHR stream quality audits |
| **Probability** | Cluster Sampling | Whole primary units sampled | Multi-center hospital protocol evaluations |
| **Imbalanced** | SMOTE | Interpolation between $k$-NN minority points | Addressing $<5\%$ stroke prevalence |
| **Imbalanced** | Borderline-SMOTE | Synthesize only near decision boundary | Sharp boundary definition in noisy clinical data |
| **Imbalanced** | Tomek Links / ENN | Remove ambiguous majority pairs | Scrubbing noisy false positives near decision boundary |
| **Imbalanced** | SMOTE-Tomek | Combined oversampling + cleaning | Peak sensitivity/specificity balance in stroke AI |
| **Validation** | Stratified K-Fold | $K$ class-balanced cross-validation folds | Reliable generalizability estimation |
| **Validation** | Bootstrapping | $N$ draws with replacement ($36.8\%$ OOB) | Out-of-Bag validation, Bagging & Random Forests |

## Boosting Ensemble Algorithms (AdaBoost, Gradient Boosting, XGBoost)

Boosting is a sequential ensemble learning meta-algorithm that converts a collection of weak learners (models that perform slightly better than random guessing, such as single-split decision stumps) into a single strong, highly accurate predictor. Unlike **Bagging** (Bootstrap Aggregating / Random Forests), which trains independent trees in parallel to reduce model **variance**, **Boosting** trains models sequentially, where each new base estimator explicitly targets and corrects the residual errors made by its predecessors, fundamentally reducing model **bias**.

```
    Bagging (Random Forest):      [Data Sample 1] -> Model 1 \
                                  [Data Sample 2] -> Model 2  -> Parallel Voting / Averaging (Reduces Variance)
                                  [Data Sample 3] -> Model 3 /

    Boosting (Ada/GBM/XGBoost):    Model 1 -> Errors/Residuals -> Model 2 -> Residuals -> Model 3 (Reduces Bias)
```

---

### 1. AdaBoost (Adaptive Boosting)

Developed by Freund & Schapire (1997), **AdaBoost** is the foundational boosting algorithm. It operates by adaptively updating the **sample distribution weights** across successive training rounds, forcing subsequent estimators to focus heavily on previously misclassified observations.

#### Core Algorithmic Mechanics:
1. **Equal Initialization:** Assign an equal weight to all $N$ training observations:
   $$w_i^{(1)} = \frac{1}{N}, \quad i = 1, \dots, N$$
2. **Iterative Weak Learner Training (for $m = 1$ to $M$):**
   - Fit a weak learner $h_m(x)$ (typically a **Decision Stump** — a 1-level decision tree with depth 1) on the weighted dataset.
   - Compute the weighted classification error rate $\epsilon_m$:
     $$\epsilon_m = \frac{\sum_{i: y_i \neq h_m(x_i)} w_i^{(m)}}{\sum_{i=1}^N w_i^{(m)}}$$
   - Calculate the estimator's "say" or **stage weight** $\alpha_m$:
     $$\alpha_m = \frac{1}{2} \ln\left(\frac{1 - \epsilon_m}{\epsilon_m}\right)$$
     *Note:* If a stump has an error $\epsilon_m < 0.5$, $\alpha_m > 0$. If error is $0$, $\alpha_m \to \infty$. If error is $0.5$ (random guess), $\alpha_m = 0$.
   - Update sample weights for round $m + 1$:
     $$w_i^{(m+1)} = w_i^{(m)} \exp\left(-\alpha_m y_i h_m(x_i)\right)$$
     - **Misclassified points** ($y_i \neq h_m(x_i)$): Weights are scaled up by $e^{\alpha_m}$.
     - **Correctly classified points** ($y_i = h_m(x_i)$): Weights are scaled down by $e^{-\alpha_m}$.
   - Normalize weights so $\sum_{i=1}^N w_i^{(m+1)} = 1$.
3. **Final Aggregation:** The strong classifier $H(x)$ is a sign-weighted majority vote:
   $$H(x) = \text{sign}\left(\sum_{m=1}^M \alpha_m h_m(x)\right)$$

#### Clinical Pros & Cons:
- **Pros:** Fast, simple, few hyperparameters to tune (`n_estimators`, `learning_rate`), low variance when combined with simple stumps.
- **Cons:** Extremely sensitive to noisy biomedical data and outliers, because outliers get repeatedly magnified in weight until they dominate subsequent stumps.

---

### 2. Gradient Boosting (GBM - Gradient Boosting Machines)

Formulated by Jerome Friedman (1999), **Gradient Boosting** generalizes boosting into a generic functional gradient descent framework. Rather than re-weighting sample points like AdaBoost, Gradient Boosting trains each new weak learner directly to predict the **pseudo-residuals** (the negative gradient of the loss function with respect to model predictions).

#### Core Algorithmic Mechanics:
1. **Initialize Base Model:** Predict a constant value that minimizes the global loss function $L(y, \hat{y})$:
   $$F_0(x) = \arg\min_{\gamma} \sum_{i=1}^N L(y_i, \gamma)$$
   - For binary cross-entropy (Deviance/Log-Loss): $F_0(x) = \ln\left(\frac{p}{1 - p}\right)$ (log-odds of stroke).
2. **Sequential Residual Descent (for $m = 1$ to $M$):**
   - Calculate the **pseudo-residuals** (steepest descent direction in function space):
     $$r_{im} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F(x) = F_{m-1}(x)}$$
     - For Mean Squared Error ($L = \frac{1}{2}(y - \hat{y})^2$): $r_{im} = y_i - F_{m-1}(x_i)$ (exact residual error).
     - For Binary Logistic Loss: $r_{im} = y_i - \hat{p}_i$ (the probability residual).
   - Fit a regression tree $h_m(x)$ with terminal regions (leaves) $R_{jm}$ to the pseudo-residuals $r_{im}$.
   - For each leaf $j$, calculate the optimal terminal value $\gamma_{jm}$:
     $$\gamma_{jm} = \arg\min_{\gamma} \sum_{x_i \in R_{jm}} L(y_i, F_{m-1}(x_i) + \gamma)$$
   - Update the additive model using a **shrinkage parameter / learning rate ($\eta \in (0, 1]$)** to prevent overfitting:
     $$F_m(x) = F_{m-1}(x) + \eta \sum_{j=1}^{J_m} \gamma_{jm} \mathbb{I}(x \in R_{jm})$$
3. **Final Prediction:** Convert additive logits back to probabilities via the sigmoid link function:
   $$\hat{p}(x) = \sigma(F_M(x)) = \frac{1}{1 + e^{-F_M(x)}}$$

#### Clinical Pros & Cons:
- **Pros:** Highly flexible with custom differentiable loss functions; captures complex non-linear clinical interactions between biomarkers.
- **Cons:** Prone to overfitting if tree depth (`max_depth`) is too deep or learning rate ($\eta$) is too high; slow training due to strictly sequential tree building.

---

### 3. XGBoost (Extreme Gradient Boosting)

Created by Tianqi Chen & Carlos Guestrin (2016), **XGBoost** is an industrial-grade, highly scalable optimization of the Gradient Boosting Machine. It introduces major mathematical and architectural breakthroughs that make it the dominant algorithm for tabular datasets:

#### 1. Second-Order Taylor Expansion (Hessians)
Traditional GBM uses first-order gradients. XGBoost approximates the objective function using both the **first-order gradient ($g_i$)** and the **second-order gradient (Hessian $h_i$)**:
$$\mathcal{L}^{(t)} \approx \sum_{i=1}^N \left[ L(y_i, \hat{y}^{(t-1)}) + g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right] + \Omega(f_t)$$
where:
$$g_i = \partial_{\hat{y}^{(t-1)}} L(y_i, \hat{y}^{(t-1)}), \quad h_i = \partial_{\hat{y}^{(t-1)}}^2 L(y_i, \hat{y}^{(t-1)})$$
*Benefit:* Using curvature (Hessian) allows the optimizer to take much more accurate Newton-Raphson steps in function space.

#### 2. Formal Objective Regularization ($\Omega$)
To penalize tree complexity and avoid memorizing training noise, XGBoost explicitly integrates L1 and L2 regularization into its objective:
$$\Omega(f_t) = \gamma T + \frac{1}{2} \lambda \sum_{j=1}^T w_j^2 + \alpha \sum_{j=1}^T |w_j|$$
- $T$: Number of terminal leaves in tree $f_t$ (penalized by parameter $\gamma$).
- $w_j$: Leaf prediction weights (penalized by L2 ridge parameter $\lambda$ and L1 lasso parameter $\alpha$).
- **Optimal Leaf Weight Formula:**
  $$w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}$$
- **Gain Formula for Split Finding:**
  $$\text{Gain} = \frac{1}{2} \left[ \frac{\left(\sum_{i \in I_L} g_i\right)^2}{\sum_{i \in I_L} h_i + \lambda} + \frac{\left(\sum_{i \in I_R} g_i\right)^2}{\sum_{i \in I_R} h_i + \lambda} - \frac{\left(\sum_{i \in I} g_i\right)^2}{\sum_{i \in I} h_i + \lambda} \right] - \gamma$$
  A split is only permitted if $\text{Gain} > 0$ (built-in automatic pruning).

#### 3. Handling Imbalanced Healthcare Data (`scale_pos_weight`)
In imbalanced datasets (e.g., $4.8\%$ stroke prevalence):
$$\text{scale\_pos\_weight} = \frac{\text{Total Negative Instances}}{\text{Total Positive Instances}} \approx \frac{4861}{249} \approx 19.5$$
Multiplying the gradient of positive instances by this ratio forces XGBoost to penalize false negatives 19.5 times more heavily, drastically elevating clinical sensitivity/recall without requiring artificial resampling!

#### 4. Algorithmic & Hardware Innovations
- **Weighted Quantile Sketch:** Discovers optimal split candidates on distributed data with mathematical theoretical error guarantees.
- **Sparsity-Aware Split Finding:** Automatically learns a default split direction for missing feature values (e.g., missing BMI measurements in EHRs).
- **Column Subsampling:** Subsamples features at both tree and split levels (inspired by Random Forests) to decorrelate individual trees.
- **Cache-Aware Block Structure:** Stores data in compressed, pre-sorted columnar memory blocks for parallel multi-threading and cache-hit maximization.

---

### 4. Comprehensive Comparison: AdaBoost vs. Gradient Boosting vs. XGBoost

| Dimension | AdaBoost | Gradient Boosting (GBM) | XGBoost |
| :--- | :--- | :--- | :--- |
| **Creator & Year** | Freund & Schapire (1997) | Jerome Friedman (1999) | Tianqi Chen (2016) |
| **Correction Mechanism** | Increases sample weights of misclassified rows | Fits regression trees to pseudo-residuals (negative gradient) | Optimizes 2nd-order Taylor expansion (gradients + Hessians) |
| **Base Estimator** | Shallow decision stumps (depth = 1) | Deeper decision trees (depth = 3–6) | Depth-regulated trees with automatic pruning |
| **Optimization Method** | Exponential Loss minimization | First-order Gradient Descent | Second-order Newton-Raphson Optimization |
| **Regularization** | Implicit via early stopping and learning rate | Shrinkage ($\eta$) and subsampling | Explicit $\text{L}_1$ ($\alpha$), $\text{L}_2$ ($\lambda$), and leaf count ($\gamma$) |
| **Missing Value Handling** | Cannot handle natively (requires imputation) | Requires imputation in base scikit-learn | Built-in Sparsity-Aware default direction learning |
| **Outlier Sensitivity** | High (outliers exponentially magnify in weight) | Moderate (huber loss mitigates outliers) | Low (Hessian denominator + regularizers tame outliers) |
| **Imbalanced Data Handling** | Difficult without external sample weighting | Subsampling / class weights | Native `scale_pos_weight` hyperparameter |
| **Computational Speed** | Fast (simple stumps) | Slower (strictly sequential, unparallelized) | Highly parallelized, cache-aware, out-of-core streaming |
| **Role in Stroke Project** | Fast baseline boosting benchmarking | Non-linear clinical interaction modeling | Primary champion production classifier for stroke prediction |



