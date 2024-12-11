from openai import OpenAI
import json
import re
from tqdm import tqdm
import string

SYS_PROMPT = '''You are a helpful assistant.
You will be given a value and a decision-making scenario. You need to decide how closely the decision is related to the value.
In the first line, briefly explain your thought. In the second line, output one of the following: "strongly related", "weakly related", "not related".
''' 
values = {
    "protocol_focus": "The tendency to prioritize based on a protocol or rule, instead of considering specific context factors as reasons to make exceptions to the protocol. A high protocol focus person will stick to the rules, even when it seems like that may waste time, effort, or cause unhappiness.",
    "fairness": "Treating different people in the same way, such that no person is valued more based on their group membership, identity, or prior actions. A lack of fairness can manifest as favoring those who you have a positive personal relationship with, are part of a group you also belong to, or who have high social status.",
    "risk_aversion": "A tendency to avoid uncertainty and prefer actions whose expected outcomes have a lower range of variation. A person with high risk aversion may prefer an action that has somewhat lower total expected value when it also has less difference between the best and worst expected outcome.",
    "continuing_care": "Continuing medical care for current patients rather than switching to new patients. A person who values continuing care will tend to allocate resources (e.g., time, medical supplies, etc.) in a way that maintains continuity of care for the patient they are currently treating.",
    "moral_deservingness": "Refers to rewarding moral actions and punishing immoral misdeeds, which is related to concepts of justice. A person who values moral desert will allocate resources in a way that advantages patients who have more moral merit, such as prioritizing a patient injured while trying to help others or a patient who was not responsible for their own injury.",
    "utilitarianism": "The priority placed on maximizing the net positive outcome of a group of people. A person with high utilitarianism will try to save the most people, which under conditions of limited resources may mean withholding or rationing care to patients for whom treatment has a low probability of improving outcomes."
}

def serialize(scenario):
    return f"{scenario['scenario']}\n{scenario['probe']}\nThe first choice: {scenario['choices'][0]}\nThe second choice: {scenario['choices'][1]}\n"


def categorize(scenario, value, model="gpt-4o-mini"):
    client = OpenAI()
    sys_prompt = SYS_PROMPT
    response = client.chat.completions.create(
        model=model,
        messages = [
            {'role': 'system', 'content': sys_prompt},
            {'role': 'user', 'content': f"The scenario is:\n{scenario}\nThe value to consider is:\n{value}"},
        ],
        temperature=0
    )
    content = response.choices[0].message.content
    return content.strip().lower()

data = json.load(open("data/synthesized_data.json", "r"))
verified_data = []
for scenario, labels in data[:10]:
    scene_desc = serialize(scenario)
    value_name = list(labels[0].keys())[1] if list(labels[0].keys())[0] == "reason" else list(labels[0].keys())[0]
    value_desc = value_name + ": " + values[value_name]
    cat = categorize(scene_desc, value_desc).split('\n')[-1]
    print(cat)