from vllm import LLM, SamplingParams
from datasets import load_dataset
import llm_blender
import numpy as np
blender = llm_blender.Blender()
blender.loadranker("llm-blender/PairRM")

dataset = load_dataset("ShenaoZ/0.0005_withdpo_4iters_bs256_555lr_dataset", split="test_prefs_1")

with torch.inference_mode():
    sampling_params = SamplingParams(temperature=0, top_p=1.0, max_tokens=1024, stop=tokenizer.eos_token, skip_special_tokens=True)
    prompts = dataset['prompt']
    model = LLM(model="mistralai/Mistral-7B-Instruct-v0.2")
    responses = model.generate(prompts, sampling_params)
    responses_list = [response.outputs[0].text.strip() for response in responses]
    dataset = dataset.add_column("reference_response", responses_list)
    dataset.push_to_hub("ShenaoZ/ablation_dataset_mistral", split='test_prefs')
