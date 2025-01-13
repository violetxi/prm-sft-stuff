import os
import json
import datasets


STEP_TOKEN = "\n\n"
dataset_name = "violetxi/prm-NUMINA-small"
ds = datasets.load_dataset(dataset_name)['train']
example_list = []

for dp in ds:
    problem = dp['problem']
    solution_steps = dp['solution_steps']
    undiscounted_rtgs = dp['undiscounted_rtgs']
    assert len(solution_steps) == len(undiscounted_rtgs)
    user_message = "Question: " + problem
    assistant_message = ""

    for i, (solution_step, rtg) in enumerate(zip(solution_steps, undiscounted_rtgs)):        
        user_message += solution_step.strip() + " " + STEP_TOKEN + " "    # need to add space before and after STEP_TOKEN to avoid tokenization issues
        assistant_message += f"{rtg}" + STEP_TOKEN
    
    example_list.append([
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_message}
    ])

json.dump(example_list, open("data/numina_small.json", "w"), indent=4, ensure_ascii=False)
        
