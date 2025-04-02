# Uncensoring Qwen-2.5-7B-Instruct-1M with Abliteration

## What is Abliteration?

Abliteration is a technique that removes the "refusal direction" from Large Language Models (LLMs), effectively uncensoring them. Modern LLMs are fine-tuned for safety and instruction-following, which means they are trained to refuse harmful requests. Research by Arditi et al. has shown that this refusal behavior is mediated by a specific direction in the model's residual stream. By preventing the model from representing this direction, it loses its ability to refuse requests.

## How Abliteration Works

The process involves several key steps:

1. **Data Collection**: Run the model on both harmful and harmless instructions, recording the residual stream activations at the last token position for each.

2. **Mean Difference Calculation**: Calculate the mean difference between the activations of harmful and harmless instructions to identify the "refusal direction" vector for each layer.

3. **Selection**: Normalize these vectors and evaluate them to select the single best "refusal direction."

4. **Ablation**: Remove the model's ability to represent this feature through either:
   - Inference-time intervention
   - Permanent weight orthogonalization

### Inference-time Intervention

For every component that writes to the residual stream (such as an attention head), calculate the projection of its output onto the refusal direction and subtract this projection. This is applied at every token and every layer.

### Weight Orthogonalization

This involves modifying the model weights directly by orthogonalizing the component weights with respect to the refusal direction, preventing the model from writing to this direction altogether.

## Implementation

The implementation relies on the [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) library for mechanistic interpretability and intervention on activations.

### Required Packages

```python
pip install transformers transformers_stream_generator tiktoken transformer_lens einops jaxtyping
```

### Key Components

1. **Datasets**: Two datasets are needed:
   - Harmless instructions (e.g., `mlabonne/harmless_alpaca`)
   - Harmful instructions (e.g., `mlabonne/harmful_behaviors`)

2. **Model Loading**: Using TransformerLens to load and manipulate the model

3. **Direction Identification**: Finding the refusal direction through activation analysis

4. **Ablation**: Implementing the ablation through weight orthogonalization

## Performance Considerations

Abliteration successfully uncensors LLMs but may cause some performance degradation across benchmarks. To address this issue, additional training can be applied:

- **DPO Fine-Tuning**: Direct Preference Optimization can help recover most of the performance drop due to abliteration without reintroducing censorship.

## Available Models

- [mlabonne/Daredevil-8B-abliterated](https://huggingface.co/mlabonne/Daredevil-8B-abliterated): Abliterated version of Llama 3 8B Instruct
- [mlabonne/NeuralDaredevil-8B-abliterated](https://huggingface.co/mlabonne/NeuralDaredevil-8B-abliterated): DPO fine-tuned version with recovered performance
- [Collection of abliterated models](https://huggingface.co/collections/failspy/abliterated-v3-664a8ad0db255eefa7d0012b) by FailSpy

## Resources

- [Original Blog Post](https://huggingface.co/blog/mlabonne/abliteration)
- [Google Colab Notebook](https://colab.research.google.com/drive/1VYm3hOcvCpbGiqKZb141gJwjdmmCcVpR?usp=sharing)
- [FailSpy's abliterator library](https://huggingface.co/collections/failspy/abliterated-v3-664a8ad0db255eefa7d0012b)
- [Research Paper: "Refusal in LLMs is Mediated by a Single Direction"](https://www.lesswrong.com/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction)

## Usage Considerations

While abliteration provides uncensored models with state-of-the-art performance, users should be aware of ethical considerations and responsible use of such models. These models are recommended for scenarios where censorship is not required while maintaining high-quality outputs.