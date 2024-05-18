#from vllm import LLM, SamplingParams
from datasets import load_dataset
import llm_blender
import numpy as np
import torch
blender = llm_blender.Blender()
blender.loadranker("llm-blender/PairRM")

dataset = load_dataset("ShenaoZhang/0.0001_zephyr_5551_4iters_bs256_dataset", split="test_prefs_1")

with torch.inference_mode():
    #sampling_params = SamplingParams(temperature=0, top_p=1.0, max_tokens=1024, stop=tokenizer.eos_token, skip_special_tokens=True)
    prompts = dataset['prompt']
    chosen_r = dataset['chosen']
    rej_r = dataset['rejected']
    
    existing_rejected_responses = []
    for idx, r in enumerate(dataset['reference_response']):
        res_content = r
        existing_rejected_responses.append(res_content)

    ref_candidates = [existing_rejected_responses[idx] for idx in range(len(prompts))]
   # candidates_texts = [[existing_chosen_responses[idx]]
   #                     + [new_responses[idx] for new_responses in all_responses] for idx in range(len(prompts))]
    candidates_texts = [[chosen_r[idx][1]["content"], rej_r[idx][1]["content"]] for idx in range(len(prompts))]
    rewards = blender.rank_with_ref(prompts, candidates_texts, return_scores=True, batch_size=2, ref_candidates=ref_candidates)
    with open("./pairrm_reward.txt", "w") as f:
        for row in rewards:
            f.write(','.join([str(item) for item in row]))
            f.write('\n')
#    model = LLM(model="mistralai/Mistral-7B-Instruct-v0.2")
#    responses = model.generate(prompts, sampling_params)
#    responses_list = [response.outputs[0].text.strip() for response in responses]
#    dataset = dataset.add_column("reference_response", responses_list)
#    dataset.push_to_hub("ShenaoZ/ablation_dataset_mistral", split='test_prefs')
