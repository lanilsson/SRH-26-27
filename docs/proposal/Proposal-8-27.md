
Effect of Heterophily-Aware GNN Architecture on Fraud Detection in Financial Transaction Graphs

Lucas Nilsson

2026-2027

Table of Contents

[Abstract**	**4](https://docs.google.com/document/d/1WOxW53Tpwv3mOcc0ceGw2dgZk-OKvZsHG8K631CCgws/edit?tab=t.0#heading=h.5o2gn9w8pc1s)

[Introduction**	**4](https://docs.google.com/document/d/1WOxW53Tpwv3mOcc0ceGw2dgZk-OKvZsHG8K631CCgws/edit?tab=t.0#heading=h.bu1qssmyolkk)

[[Section 1: The Problem]**	**5](https://docs.google.com/document/d/1WOxW53Tpwv3mOcc0ceGw2dgZk-OKvZsHG8K631CCgws/edit?tab=t.0#heading=h.1yu18k9l7dz6)

[[Section 2: Independent Variable]**	**6](https://docs.google.com/document/d/1WOxW53Tpwv3mOcc0ceGw2dgZk-OKvZsHG8K631CCgws/edit?tab=t.0#heading=h.j7ygi1r47pob)

[[Section 3: Dependent Variable and Model System]**	**6](https://docs.google.com/document/d/1WOxW53Tpwv3mOcc0ceGw2dgZk-OKvZsHG8K631CCgws/edit?tab=t.0#heading=h.l3cgavczfj1m)

[[Section 4: Implications]**	**7](https://docs.google.com/document/d/1WOxW53Tpwv3mOcc0ceGw2dgZk-OKvZsHG8K631CCgws/edit?tab=t.0#heading=h.vs4qld4dplqp)

[Research or Engineering Goal**	**7](https://docs.google.com/document/d/1WOxW53Tpwv3mOcc0ceGw2dgZk-OKvZsHG8K631CCgws/edit?tab=t.0#heading=h.xlgz96zp3tr)

[Methodology**	**7](https://docs.google.com/document/d/1WOxW53Tpwv3mOcc0ceGw2dgZk-OKvZsHG8K631CCgws/edit?tab=t.0#heading=h.w5mjmlxt1lpt)

[[System Environment &amp; Materials]**	**7](https://docs.google.com/document/d/1WOxW53Tpwv3mOcc0ceGw2dgZk-OKvZsHG8K631CCgws/edit?tab=t.0#heading=h.wiyb4sfp4w8v)

[[Experimental Design]**	**8](https://docs.google.com/document/d/1WOxW53Tpwv3mOcc0ceGw2dgZk-OKvZsHG8K631CCgws/edit?tab=t.0#heading=h.wbw5fu4g6nu5)

[[Experimental Assessment]**	**8](https://docs.google.com/document/d/1WOxW53Tpwv3mOcc0ceGw2dgZk-OKvZsHG8K631CCgws/edit?tab=t.0#heading=h.4rq8d5irkkt1)

[[Data Analysis]**	**8](https://docs.google.com/document/d/1WOxW53Tpwv3mOcc0ceGw2dgZk-OKvZsHG8K631CCgws/edit?tab=t.0#heading=h.k9ub3gweb84l)

[Materials and Safety**	**9](https://docs.google.com/document/d/1WOxW53Tpwv3mOcc0ceGw2dgZk-OKvZsHG8K631CCgws/edit?tab=t.0#heading=h.ga422sub343)

[References**	**10](https://docs.google.com/document/d/1WOxW53Tpwv3mOcc0ceGw2dgZk-OKvZsHG8K631CCgws/edit?tab=t.0#heading=h.30t8hrrpyji2)

# Abstract

[(250 words) the abstract is NOT a summer requirement but please keep this as a placeholder.]

[The percentages are guidelines, an outline of your proposal abstract is as follows:

* [Introduction:
  * What is the disease/problem? (one sentence)
  * What key piece of evidence supports the knowledge gap (one sentence)
* 3-4 sentences describing your methodology
  * 1 sentence describing model system
  * 1 sentence describing your IV (what you are comparing)
  * 1-3 sentences describing how you will collect your data (DV)
* 1 sentence describing your hypothesis (what you expect to prove and what it is based upon)
* 1 sentence describing the implications of your research if you hypothesis is correct]

# Introduction

## Section 1

Financial crime operates through complex networks of relationships among people, accounts, devices, and transactions. In 2023, an approximate $3.1 trillion in illicit funds flowed through the global financial system, while fraud scams and bank-fraud schemes produced approximately $485.66 billion in global losses([Nasdaq](https://drive.google.com/file/d/1n7Ht6_yx72FGsmtlb564SzNZYkZn7jVQ/view?usp=sharing), 2024, Figure 1). Traditional fraud detection often evaluates transactions using transaction-level features, but financial activity is relational. Fraud can be connected through accounts, devices, payment methods, or previous transactions. This data can be represented through graphs, where nodes represent transactions or financial entities and edges represent relationships between them. Graph based fraud detection can therefore incorporate  both node-level information and information from connected entities, motivating the use of relational approaches for fraud classification. ([Weber et al., 2019](https://drive.google.com/file/d/1jLPUX6DeO84MBej_0jAQCn6Yb0rNN2Z3/view?usp=sharing))

Most message passing Graph Neural Networks update a node by combining its own information with information from nearby nodes. Message-passing GNNs are particularly effective when connected nodes tend to have similar labels or characteristics, known as homophily([Zhu et al., 2020](https://drive.google.com/file/d/1LjYVCMwxWjfU-DrvMULULESwX6W5V23X/view?usp=sharing)). Conventional neighborhood aggregation may become less effective when a fraudulent node is primarily connected to legitimate nodes. In heterophilous graphs, however, connected nodes may have different labels or features making neighborhood information less reliable for classification. Financial fraud provides a good setting for this problem because fraudulent transactions can be connected to legitimate transactions and nodes rather than existing only in isolated fraudulent clusters. ([Dou et al., 2020](https://drive.google.com/file/d/15y2Rd_ds-IBk_4eecveNNC3CModtkCvU/view?usp=sharing); [Wang et al., 2023](https://drive.google.com/file/d/1LZ-GZn0g7dRrmvzxOKaPWdXrttniMcCi/view?usp=sharing))

Previous fraud detection research has identified high heterophily and misleading neighborhoods as challenging for conventional message passing approaches. Under these conditions, aggregating information from legitimate neighbors may dilute or obscure fraudulent information contained in the target node. However the effects of specific heterophily aware mechanisms particularly ego neighbor representation separation and explicit second order neighborhood aggregation on fraud classification remain insufficiently characterized in financial transaction graphs.

(Figure 1)

![]()

## Heterophily-Aware Neighborhood Aggregation

Prior fraud detection research indicates that neighborhood aggregation can become unreliable when fraudulent nodes are connected to misleading or legitimate neighbors. CARE-GNN was developed to address this issue by identifying and selecting more informative neighboring nodes, showing that total aggregation of neighbors can reduce performance. ([Dou et al., 2020](https://drive.google.com/file/d/15y2Rd_ds-IBk_4eecveNNC3CModtkCvU/view?usp=sharing)). More recently, Group aggregated enhanced transformers(GAGA) identified low homophily as a persistent challenge for graph-based fraud detection and showed that modifying how neighboring information is grouped and aggregated can improve classification in highly heterophyllous settings ([Wang et al., 2023](https://drive.google.com/file/d/1LZ-GZn0g7dRrmvzxOKaPWdXrttniMcCi/view?usp=sharing)). Together, these findings show that the aggregation process is an important target for improving GNN performance on fraud graphs and provide a basis for testing.

One example is ego neighbor representation separation. In a conventional Graph Convolutional Network(GCN), a node’s representation is repeatedly updated by averaging information from the node itself with information from its neighbors. Under heterophily, this mixing can dilute the features from each node. Zhu et al. (2020) identified ego neighbor separation as an effective design for combatting the dilution.([Zhu et al., 2020](https://drive.google.com/file/d/1LjYVCMwxWjfU-DrvMULULESwX6W5V23X/view?usp=sharing)) Instead of immediately averaging the two information sources, the node’s own representation is maintained separately from the aggregated group of neighbors before the representations are combined. Zhu et al. found that architectures preserving this separation performed substantially better than conventional GCN-style aggregation under heterophily.([Zhu et al., 2020](https://drive.google.com/file/d/1LjYVCMwxWjfU-DrvMULULESwX6W5V23X/view?usp=sharing)) In financial fraud graphs, this mechanism is particularly relevant because a fraudulent node may contain fraudulent features while being surrounded by legitimate neighbors. Maintaining the ego and neighborhood representations separately may preserve information that would have been diluted.

A second mechanism is explicit higher order neighborhood aggregation. Most conventional GCN layer aggregation begins with information from immediate, or second layer neighbors, while information from more distant nodes is incorporated indirectly though repeated message passing layers. Zhu et al. instead proposed explicitly aggregating information from neighborhoods at different graph distances.([Zhu et al., 2020](https://drive.google.com/file/d/1LjYVCMwxWjfU-DrvMULULESwX6W5V23X/view?usp=sharing)) In this framework, first layer neighbors are nodes one edge from the target node, while second layer neighbors are nodes are 2 edges away. Their analysis showed that although immediate neighborhoods may be heterophilous, higher order neighborhoods can contain different and potentially more informative class structure, making explicit higher order aggregation beneficial under some heterophilous conditions. This differs from standard modification of GCN depth because first and second layer neighborhood information remain distinguishable rather than being obtained only through repeated mixing. In financial transaction graphs, where fraud can take multiple steps through accounts, explicit second order aggregation may therefore provide useful structural information that is not available from the immediate neighborhood alone.

Zhu et al. combined ego neighbor separation, higher order neighborhoods and intermediate representation combination into H2GCN and demonstrated that these design principles can improve node classification under heterophily([Zhu et al., 2020](https://drive.google.com/file/d/1LjYVCMwxWjfU-DrvMULULESwX6W5V23X/view?usp=sharing)). However the proposed study will not evaluate H2GCN as a single packaged architecture. Instead, each principle was implemented independently within a controlled GCN. Four configurations will be compared: a baseline containing neither, ego-separation only, second order only and a combined configuration containing both. This design will allow the individual effects of each mechanism as well as their combined effect to be evaluated.

| Study                                                | Problem / Graph setting                                               | Relevant finding                                                                                                                                                       | Relevance to study                                                                                                           |
| ---------------------------------------------------- | --------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| [Zhu et al (2020)](https://arxiv.org/abs/2006.11468)  | Node classification across graphs with different levels of homophily. | Identified ego neighbor representation separation and higher order neighborhoods as effective principles under heterophily                                             | Provides the basis for the two architects manipulated in the study.                                                          |
| [Dou et al. (2020)](https://arxiv.org/abs/2008.08692) | Graph based fraud detection involving disguised fraudulent nodes      | Demonstrated that misleading/different neighboring nodes can interfere with GNN aggregation and that selecting more informative neighbors can improve fraud detection. | Establishes that neighborhood aggregation is a fraud specific problem thatcher than only a general graph learning limitation |
| [Want et al (2023)](https://arxiv.org/abs/2302.10407) | Graph based fraud detection under low homophily                       | Identified low homophily as a major challenge for existing fraud GNNs and demonstrated improved performance using specialized group aggregation                        | Supports the relevance of heterophily aware aggregation specifically within fraud detection graphs                           |

Table 1: Evidence supporting heterophily aware neighborhood aggregation. Prior research provides both general heterophily theory and fraud specific evidence supporting the use of ego neighbor separation and explicit second order neighborhood aggregation as the independent variables.

## Financial Transaction Graphs and Fraud Classification performance

The proposed architectural mechanisms will be evaluated on two public financial fraud graphs: the Elliptic Bitcoin transaction graph and IEEE-CIS-G from the H²GB benchmark. Elliptic represents Bitcoin transactions as nodes and payment flows as directed edges, with labeled transactions categorized as licit or illicit ([Weber et al., 2019](https://drive.google.com/file/d/1jLPUX6DeO84MBej_0jAQCn6Yb0rNN2Z3/view?usp=sharing)). The dataset contains more than 200,000 transactions. Because previous studies have used it to classify illicit transactions, it provides a suitable benchmark.

IEEE-CIS-G provides a structurally different system. It represents the IEEE-CIS Fraud detection dataset as a heterogeneous graph where transaction nodes are indirectly linked though shared metadata such as card information, address and email. The H²GB benchmark was developed to evaluate graph learning under heterogeneity and heterophily and provides a train, validation and test split ([Lin et al., 2024](https://drive.google.com/file/d/107KWBoIGqQ6ZG7jxJOucyqV98UymdJXb/view?usp=sharing)). Testing the same independent variables on both Elliptic and IEEE-CIS-G will properly evaluate whether the implementation's results persist across different financial graphs.

Fraud classification performance will be measured using PR-AUC, fraud class F1 score, and ROC-AUC. PR-AUC will serve as the primary measurement because precision recall analysis directly evaluates performance on the minority positive class and is particularly informative for highly imbalanced classification tasks ([Saito &amp; Rehmsmeier, 2015](https://drive.google.com/file/d/1IDv7jRJpl3kLfSpqfkER7BwuoH5OTVhD/view?usp=sharing)). The fraud-class F1 score will measure the balance between precision and recall at a selected classification threshold, while legitimate observations across thresholds. Because ROC-AUC can appear optimistic under severe class imbalance, it will be assessed alongside PR-AUC and F1([Saito &amp; Rehmsmeier, 2015](https://drive.google.com/file/d/1IDv7jRJpl3kLfSpqfkER7BwuoH5OTVhD/view?usp=sharing)).

Together, the two model systems and three performance metrics will allow the effects of ego neighbor separation and explicit second order neighborhood aggregation to be evaluated across structurally different graphs.

## Hypothesis and Implications

If ego neighbor representation separation and explicit second order neighborhood aggregation improve the ability of GNNs to learn from heterophilous financial transaction graphs then either mechanism is expected to return higher PR-AUC, F1 score, and ROC-AUC than the baseline GCN configuration. The  combination of both mechanisms is expected to produce the greatest improvement because it simultaneously preserves node specific information and incorporates structurally different information from more distant layers. This prediction is supported by prior findings that heterophily reduces the effectiveness of conventional neighborhood aggregation and that separating ego and neighbor representation and explicitly incorporating heifer order neighborhoods can improve results ([Zhu et al., 2020](https://drive.google.com/file/d/1LjYVCMwxWjfU-DrvMULULESwX6W5V23X/view?usp=sharing))

If supported, these results would demonstrate that the way neighborhood information is aggregated is an important factor in fraud classification rather than the choice of a broader GNN architecture. By isolating the effects of the two independent variables I will be implementing, the study could identify which mechanisms contribute the most to improved fraud classification. These findings could guide the development of future graph based fraud detection systems that better preserve informative signals when fraudulent activity is present.

# Research or Engineering Goal

The goal of this study is to determine how ego neighbor representation separation and explicit order neighborhood aggregation, individually and in combination, affect fraud classification performance in the Elliptic and IEEE-CIS-G financial graph using a controlled GCN style architecture.

The three measurable endpoints used to evaluate these effects will be:

* PR-AUC, the primary endpoint for minority class fraud detection performance
* Fraud-class F1 score, measuring the balance between precision and recall for illicit nodes
* ROC-AUC, measuring overall ability to distinguish illicit and licit transactions across classification thresholds.
* [The goal is to bridge your intent with your outcome
* State the singular, focused objective of your study in 1–2 sentences; for a Science project, this must be a testable question (e.g., "To determine the effect of X on Y"), while an Engineering project must define a specific need or functional goal (e.g., "To design and prototype a system that achieves X under Y constraints").
* Explicitly list the three quantitative design criteria (for engineering) or measurable endpoints (for science) that will define a "successful" outcome, ensuring these goals directly address the knowledge gap identified in your literature review.]

# Methodology

[The Methodology section must be written with enough precision that another researcher could replicate your exact work. Model your writing after the "Methods" or "Technical Specifications" sections of your primary source papers.

Tense: Since this is a proposal, use the future tense ("The prototype will be tested...").

Specifics over Generalities: Use units and part numbers. Instead of "strong wire," use "22 AWG copper hook-up wire." Instead of "liquid," use "50 μL of 10mM NaCl."]

## System Environment & Materials

The study will use the publicly available Elliptic Bitcoin dataset and IEEE-CIS-G from the H²GB benchmark. The published graph structures include features, labels, and data splits.

Models will be implemented in Python using PyTorch Geometric for graph processing and model training. Scikit-learn will be used to calculate PR-AUC, fraud class F1 score, and ROC-AUC, while numpy and Pandas will be used for data preparation and processing. Exact software and package versions will maintain consistency over the entire study.

Training will be performed using a A40 NVIDIA GPU with 48GB of VRAM on a cloud server. THe same environment will be maintained over each study to reduce variation. I estimate the entire study to cost roughly $800 to $1,400 with the GPU costing $0.48/Hr to $2/Hr. The high cost is mainly due to the high demand for high end GPUs and the inflated price of ram. No physical equipment or laboratory needed.

* [Science: Describe your Model System (e.g., S. cerevisiae or cell lines). Detail the source, strain, and environmental conditions (temperature, media) required to maintain it.
* Engineering: Describe your System Components and Build Environment. List all materials, sensors, software versions, and hardware specifications (including manufacturers and model numbers).]

## Experimental Design

A controlled comparison of four architectures will be conducted on the Elliptic and IEEE-CIS-G financial graphs. Two factors will be manipulated: ego neighbor representation separation and explicit second order neighborhood aggregation. Four GCN style configurations will be tested: 1. baseline 2. Ego neighbor separation only 3. Second order aggregation only 4. Both mechanisms combined. The baseline GCN will serve as the control.

The published graph structures, node features, labels and data splits will remain fixed across all runs. Unlabeled Elliptic nodes will remain in the graph for message passing but will not contribute to evaluation. Model settings will be optimized using only the training and validation data, with the same training budget for each. Final settings will be frozen before test set evaluation to prevent data leakage.

Each architecture dataset condition will be trained independently five times using random seeds, producing 40 final runs.

* [Science (Treatment Groups): Explicitly define your Negative Control, Positive Control, and Experimental Groups. Detail your IV application: manufacturer, concentration, and the vehicle (e.g., saline).
* Engineering (Design Iterations): Define your Control/Benchmark (the current standard you are trying to improve) and your Prototype Specifications. Detail the independent variables you are adjusting (e.g., different materials, code algorithms, or voltages).
* Visual Strategy: You must include an Overall Study Design Flowchart (Figure 1). For science, show the treatment timeline; for engineering, show a system schematics or a CAD model. Lead the reader to it in your text: "(See Figure 1)." Clearly define what constitutes 'Time Zero' in your study. Is it the moment of inoculation, the first application of the IV, or the baseline measurement?]

## Experimental Assessment

After each training run, model predictions on the fixed test set will be used to calculate the three dependent variables. The performance will be calculated using scikit-learn.

### PR-AUC

Precision and recall will be calculated across classification thresholds, and the area under the resulting precision recall curve will be measured. PR-AUC will serve as the primary performance metric because both datasets contain substantially fewer fraudulent than legitimate nodes.

### F1 Score

The F1 score for the illicit class will represent the balance between precision and recall. It will be calculated using a classification threshold selected on the validation set and then applied unchanged to the test set.

### ROC-AUC

ROC-AUC will be calculated by comparing each model’s predicted fraud probabilities with the actual fraud labels in the test set. This metric will measure the model’s ability to rank fraudulent nodes above legitimate nodes across classification thresholds.

Each measurement will be recorded for all five independently seeded runs of every architecture/dataset setup

.

* [Note: Create a sub-header (use header 3 for each assay) for each of your three required measurements (e.g., "3.1: Structural Integrity Stress Test").
* Science (Assays): Describe the step-by-step procedure for each measurement. Mention specific equipment and the exact timing of data collection.
* Engineering (Test Protocols): Describe the "Stress Tests" or "Performance Trials" for your prototype. What are the three specific metrics (DVs) you are measuring (e.g., efficiency, load capacity, battery life)?
* Replication: State your sample size (n) or number of trials. For example: "Each trial will be performed five times (n=5) to ensure reliability."]

## Data Analysis

For each architecture/dataset layout, PR-AUC, Fraud-class F1, and ROC-AUC will be summarized as the mean ± standard error of the mean across the five independent training runs. PR-AUC will be the primary measure used, while F1 and ROC-AUC will provide additional data of performance.

Each dataset will be analyzed separately to determine how ego neighbor separation and second order neighborhood aggregation affect model performance, both individually and together. Results with p < 0.05 will be considered significant.

Analyzing Elliptic and IEEE-CIS-G separately will allow the study to determine whether the effects of the heterophily aware mechanisms are consistent across different financial graph structures or perform differently on each dataset.

* [Science (Statistical Analysis): Identify your statistical test (e.g., t-test or ANOVA). State that data will be presented as Mean ± SEM and results considered significant at p < 0.05.
* Engineering (Success Criteria): Define the quantitative "Design Goals" that must be met to consider the project a success. Compare your prototype’s performance data against your Benchmark using percentage improvements or error margins.
* The "Why" of Analysis: Whether you are doing science or engineering, you must prove your results aren't a fluke. Science uses p-values to prove the effect is real; Engineering uses Tolerance and Reliability tests to prove the design is stable.]

## Materials and Safety

This study is entirely conducted on my computer and does not require any other physical hardware, tools or equipment. No dangerous chemicals will be handled making the SDS not applicable. The materials used will be public datasets, open-source software, and compute. The only cost is GPU computing though Run Pod costs from $800-$1,200.

[To ensure a seamless transition into the lab, you must complete a comprehensive audit of your required materials. Use the template below to organize your reagents, hardware, and safety protocols.

* The Procurement Table: Create a table (using the template below) listing every chemical, reagent, or hardware component required.
* Sourcing : For each item, check the attached [inventory sheet](https://docs.google.com/spreadsheets/d/1sRC6o5LlT2rD6YxUG0yTysvvx4ZpcSIGZZBLi4AxYus/edit?usp=sharing) for currently available chemicals and equipment. If we have it, just state “on inventory list”. If it is not there, identify the [vendor using the preferred supplier on this list](https://docs.google.com/spreadsheets/d/1MZeyehtNeNAX0XIn4dCig278WUPSZSc9vZy1G0WIfQI/edit?usp=sharing).
* SDS Documentation: You must link the Safety Data Sheet (SDS) for every material (google your reagent and SDS to find it). Under "Handling," write a brief Standard Operating Procedure (S.O.P.) summarizing the specific safety precautions (e.g., "Use only in fume hood" or "Requires secondary containment").
* Read Section 2 & 8: When looking at an SDS, focus on Section 2 (Hazard Identification) and Section 8 (Exposure Controls/Personal Protection) to write your handling summary.
* Gifted Model Systems: If your model system (e.g., transgenic Drosophila, C. elegans, or specific seeds) must be sourced from an external research lab, list the lab as the "Vendor." You must draft an email to the Principal Investigator (PI) requesting the material and CC me on the message.
* Verify Availability: Ensure the vendor actually has the item in stock. If an item is "Backordered," you need to find an alternative source now.
* Keep this table updated; you will use this information to meet with me in September to place your order. You will also submit this to the state science fair. No chart, no order, no study. ]

| Item Name                                                                                                                                                                               | Vendor / Source                | Catalog # (if known) | Amount needed           | Cost       | SDS Link | Primary Hazard & Handling Summary |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------ | -------------------- | ----------------------- | ---------- | -------- | --------------------------------- |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------       |                                |                      |                         |            |          |                                   |
| [Elliptic Bitcoin Dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)                                                                                                 | Public research dataset        |                      | 1                       | $0         |          | Digital dataset - N/A             |
| -                                                                                                                                                                                       | -                              | -                    | -                       | -          | -        | -                                 |
| [IEEE-CIS-G](https://www.kaggle.com/datasets/junhonglin/ieee-cis-dataset-for-h2gb)                                                                                                       | Public H²GB benchmark/dataset |                      | 1                       | $0         |          | Digital dataset - N/A             |
| [Python](https://www.python.org/)/[PyTorch](https://pytorch.org/)/[PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/)/[Scikit-Learn](https://scikit-learn.org/stable/) | Open source software/libraries |                      | 1                       | $0         |          | Digital software - N/A            |
| [GPU Computing](https://www.runpod.io/?ref=7n60davc&gad_source=1&gad_campaignid=23984366131)                                                                                             | Run Pod                        |                      | As required for trining | $800-$1200 |          | Digital cloud access to compute   |

# References

Full literature folder:
[Lucas Nillson SRH Literature Folder - 2026-27 ](https://drive.google.com/drive/folders/1LWKtGOTgvUtcM_c21TlZaeJmrbfbb0Y1?usp=sharing)

Dou, Y., Liu, Z., Sun, L., Deng, Y., Peng, H., & Yu, P. S. (2020). Enhancing graph neural network-based fraud detectors against camouflage fraudsters. In Proceedings of the 29th ACM International Conference on Information & Knowledge Management. Association for Computing Machinery. [(Dou, Y, 2020) - Enhancing Graph Neural Network-based Fraud Detectors against Camouflaged Fraudsters](https://drive.google.com/file/d/15y2Rd_ds-IBk_4eecveNNC3CModtkCvU/view?usp=sharing)

Lin, J., Guo, X., Zhang, S., Zhu, Y., & Shun, J. (2025). When heterophily meets heterogeneity: Challenges and a new large-scale graph benchmark. In Proceedings of the 31st ACM SIGKKDD Conference on Knowledge Discovery and Data Mining, Vol. 2. Association for Computing Machinery

[(Lin, J, 2025) - When heterophily meets heterogeneity: Challenges and a new large-scale graph benchmark.](https://drive.google.com/file/d/107KWBoIGqQ6ZG7jxJOucyqV98UymdJXb/view?usp=sharing)

Nasdaq. (2024). Nasdaq Verafin 2024 global financial crime report.

[(Nasdaq, 2024) - Nasdaq Releases First Global Financial Crime Report, Measuring the Scale and Human Impact of Financial Crime.pdf](https://drive.google.com/file/d/1n7Ht6_yx72FGsmtlb564SzNZYkZn7jVQ/view?usp=sharing)

Saito, T., & Rehmseier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. PLOS ONE, 10(3), e0118432.

[(Saito, T., &amp; Rehmseier, M., 2015) - The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets..pdf](https://drive.google.com/file/d/1IDv7jRJpl3kLfSpqfkER7BwuoH5OTVhD/view?usp=sharing)

Wang, Y., Zhang, J., Huang, Z., Li, W., Feng, S., Ma, Z., Sun, Y., Yu, D., DOng, F., Jin, J., Wang, B., & Luo, J. (2023). Label information enhances fraud detection against low homophily in graphs. In Proceedings of the ACM Web Conference 2023. Association for Computing Machinery.

[(Wang, Y., 2023) - Label information enhances fraud detection against low homophily in graphs.pdf](https://drive.google.com/file/d/1LZ-GZn0g7dRrmvzxOKaPWdXrttniMcCi/view?usp=sharing)

Weber, M., Domeniconi, G., Chen, J., Weidele, D. K. I., Bellei, C., Robinson, T., & Leiserson, C. E.(2019). Anti-money laundering in Bitcoin: Experimenting with graph convolutional networks for financial forensics. In KDD ‘19 Workshop on Anomaly Detection in Finance. Association for Computing Machinery.

[(Weber, M., 2019) - Anti-money laundering in Bitcoin- Experimenting with graph convolutional networks for financial forensics..pdf](https://drive.google.com/file/d/1jLPUX6DeO84MBej_0jAQCn6Yb0rNN2Z3/view?usp=sharing)

Zhu, J., Yan, Y., Zhao, L., Heimann, M., Akoglu, L., & Koutra, D. (2020). Beyond homophily in graph neural networks: Current limitations and effective designs. Advances in Neural Information Processing Systems, 33, 7793-7804.

[(Zhu, J., 2020) - Beyond homophily in graph neural networks- Current limitations and effective designs..pdf](https://drive.google.com/file/d/1LjYVCMwxWjfU-DrvMULULESwX6W5V23X/view?usp=sharing)

* [List all your references in APA style below.
* Source Quality: Use only peer-reviewed articles sourced from reputable databases like PubMed, Google Scholar, or Web of Science. Avoid using encyclopedias (like Wikipedia), general news articles, or non-peer-reviewed blog posts as primary evidence. You may use .org or .gov in some cases.  Peer-reviewed journals have a rigorous vetting process where other experts in the field check the math, logic, and ethics of the study before it is published. This is the "gold standard" of information.
* Direct Access Requirement: For every entry in your reference list, you must provide a hyperlink to the full-length PDF (not just the abstract) downloaded and found in your literature folder.
* Verification: Linking the full-length article allows me to verify that the “claims” and "results" you cited actually match the data in the paper.
* For any Elsevier papers, go to the library link Science Direct (school library login — credentials in the class doc, not stored in this repo).]
