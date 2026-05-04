# dorm-chef

CIS 5270 course project (Akshat, Rashi, Ritika). 
This repository contains work on fine-tuning `gpt-4.1-nano` with supervised fine-tuning (SFT) and direct preference optimization (DPO) for constrained, dorm-oriented recipe generation.

Models are evaluated with an LLM-as-judge protocol implemented in the notebooks under `python-notebooks/`. 
Aggregate metrics reported below are taken directly from the JSON summaries in `results/` for reproducibility.

## Task definition

The model receives a list of ingredients and, in the second experimental setting, an explicit cuisine or style constraint (for example, Italian-style). The target output is a brief recipe suitable for students: total time under 15 minutes, three to seven numbered steps, and use of only the specified ingredients.

The judge emits binary indicators for `ingredient_compliance`, `num_steps_in_range`, and `cook_time_in_range`, and in experiment 2 additionally `constraint_followed`. It also provides `coherence` and `student_friendly`. The metric `all_pass` is defined as satisfaction of all hard constraints (three in experiment 1; four in experiment 2). The composite `llm_score` is computed as 0.5 × coherence + 0.5 × student_friendly.

Training and validation data are located under `data/training data/SFT/`. Held-out evaluation uses `data/testing data/test-data.jsonl` (experiment 1) and `data/testing data/test-constrained.jsonl` (experiment 2); the latter includes a `constraint` field per example.

## Experiment 1

Experiment 1 evaluates the standard formulation without an additional cuisine-specific criterion in the judge beyond the usual recipe constraints.

The base model achieves strong constraint satisfaction on this split. Supervised fine-tuning matches the base model on `all_pass`. The DPO model shows a modest reduction on hard-constraint rates and a slight decrease in mean `llm_score` relative to the base and SFT runs.

Artifacts: `results/experiment1/base/base_summary.json`, `results/experiment1/sft/sft-alpha_summary.json`, and `results/experiment1/dpo/`
Training curves are stored under `training losses/experiment1/sft/` and `training losses/experiment1/dpo/`. 
The evaluation notebook is `python-notebooks/experiment1/Evaluation_on_Test_Run_.ipynb`.

## Experiment 2

Experiment 2 retains the same core recipe requirements but evaluates on prompts that specify a style or cuisine; the judge includes `constraint_followed`. Reported summaries use 200 completed generations per configuration.

Relative to experiment 1, overall pass rates decrease across configurations. Direct preference optimization attains the highest `all_pass` and competitive `constraint_followed`. The SFT model improves constraint adherence and soft scores but exhibits lower `ingredient_compliance`, which reduces `all_pass` relative to DPO.

Artifacts: `results/experiment 2 constrained/base/base_summary.json`, `results/experiment 2 constrained/sft/sft_gamma2.0_summary.json`, `results/experiment 2 constrained/dpo/dpo_gamma_summary.json`. 
Training losses appear under `training losses/experiment2/`. 
Notebooks: `python-notebooks/experiment2 - more constrained/Evaluation_on_Test_Set_Run_Constrained.ipynb` and `Synthetic_Data_Generation_Pipeline_constrained_Experiment_2_0.ipynb`.

## Additional run: harder constrained synthetic training data

A supplementary training and evaluation track, with metrics under `results/trained on harder constrained data/`, uses more aggressively constrained synthetic data (200 examples in the stored summaries). 
On that track, SFT (`constrained-sft`) achieves `all_pass` 0.810 versus DPO (`constrained-dpo`) at 0.835, with DPO higher on `constraint_followed` (0.935 vs. 0.895) and `ingredient_compliance` (0.935 vs. 0.885).

## Reproducing evaluations

Configure the Azure OpenAI client as in Step 0 of the notebooks. Set `ECFG.model` to the target deployment, select the appropriate test JSONL, execute the notebook pipeline, and aggregate outputs into the same schema as the `*_summary.json` files for direct comparison with the reported summaries.
