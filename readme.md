# Evaluating and Unlearning Bias in Language Models

## Overview

This project focuses on evaluating and mitigating biases in Large Language Models (LLMs) such as BERT. The primary goal is to ensure that LLMs produce fair and unbiased outputs, particularly in sensitive contexts such as gender, profession, race, and religion.

### Why This Project?

1. **Prevent Harmful Outputs:** Ensure that models do not reinforce harmful stereotypes or biases.
2. **Ensure Fairness:** Make sure that the model's outputs are fair across different domains.
3. **Compliance and Trust:** Meet legal and ethical standards for AI fairness.

## Bias Metrics

To evaluate and mitigate bias, the following metrics are used:

1. **StereoSet Score (ss):** Measures how well the model avoids generating biased or stereotypical text.
2. **Likelihood Score (lms):** Assesses the likelihood that the model generates unbiased content.
3. **In-Context Acceptance Score (icat):** Evaluates how the model handles contextually sensitive information.

## Usage

### Unlearning Bias

Use the `general_similarity_retrain.py` script to perform the unlearning process using PCGU (Projection-based Contextual Gradient Unlearning).

```bash
python general_similarity_retrain.py -m <model_path_or_name> [options]
```


#### Example

```bash
python general_similarity_retrain.py -m bert-base-uncased -n 5 -b 32 -l 1e-5 -k 10000
```


### Evaluating Models

Use the `evaluate_models.py` script to evaluate the bias of LLMs using the StereoSet dataset. It can evaluate both pre-trained models and fine-tuned versions.

```bash
python src/evaluate_models.py -t <model_type> -m <model_name> [options]
```

#### Arguments

- `-t`: The type of model to evaluate (e.g., bert-base-uncased).
- `-m`: The name of the model to evaluate.
- `--ss_dev_proportion`: Proportion of the StereoSet dev set to use as the dev set (default: 0.5).
- `--models_base_loc`: Relative path to the root of all model checkpoints (default: ./models).
- `--pretrained_only`: Flag to evaluate only the pretrained model.
- `--unshuffled`: Flag to use unshuffled data (default: uses shuffled data).

#### Example

```bash
python src/evaluate_models.py -t bert-base-uncased -m bert-base-uncased --pretrained_only
```


## Unlearning Process

The unlearning process is designed to remove the influence of specific training examples by adjusting the model’s parameters. Here’s a brief overview of the unlearning steps:

1.**Compute Gradients:** Calculate the gradients for both disadvantaged and advantaged sequences. The disadvantaged sequences represent examples that should be less influential, while advantaged sequences are those that the model should learn more about.
2.**Dynamic Gradient Selection:** Optionally, use dynamic gradient selection to adjust the gradients based on their impact on model performance. This involves multiplying the gradients by a factor depending on whether they are advantageous or disadvantaged.
3.**Parameter Update:** Update the model parameters using the computed gradients. The script uses similarity metrics to decide which parameters to keep and how to update them. Parameters are updated based on their gradients’ similarity, minimizing or maximizing their influence as specified.
4.**Gradient Calculation Functions:**
    •	_minimize_grads_2: Minimizes gradients from the second set.
    •	_maximize_grads_1: Maximizes gradients from the first set.
    •	_maximize_grads: Maximizes the sum of both gradient sets.
5.**Checkpointing:** Save model checkpoints periodically to allow for resuming training or to evaluate different stages of the model.



## Key Functions

- `compute_similarities`: Computes cosine similarities between gradients.
- `find_parameters_to_keep`: Finds parameters to keep based on gradient similarities.
- `update_model_param_grads`: Updates model gradients based on selected parameters.
- `take_optim_step`: Performs an optimization step based on updated gradients.
- `do_an_mlm_backprop` and `do_an_lm_backprop`: Perform backpropagation for masked language models (MLM) or language models (LM).

## Results

Evaluation results are stored in the `results` folder:

- `results/pretrained_<model_name>_results_map.json`: Results for the pretrained model.
- `results/<model_name>_results_map.json`: Results for both pretrained and fine-tuned models.
- `results/best_<model_name>_results.json`: Best results per level.
- `results/all_<model_name>_results.json`: All results collected.

## Additional Evaluation

To evaluate a model on CrowS, use the script `src/eval_on_crows.py`.

## Contributing

Feel free to extend or modify the implementation in `src/general_similarity_retrain.py`. The training procedure is simple and each epoch is cheap to train, allowing for experimentation with different configurations.

## Note

Remember to clear out checkpoints after use, as the scripts do not overwrite checkpoints for the same configuration of model, which can use a lot of storage.
