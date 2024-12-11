from openai import OpenAI
import json
import re
from tqdm import tqdm
import string


SYS_PROMPT = '''You are a helpful questionnaire designer.
You will be given a decision-making scenario aimed at evaluating one of the following six values.
The scenario is about how to distributed limited resources in medical emergencies. Each choice exhibits low or high value.
Your goal is to create another medical emergency scenario to evaluate on the same value. The value score should either be 10 (high) or 0 (low).
Remember:
1. Be creative, you don't have to follow the original scenario.
2. Make sure at least 1 choice is for high value and 1 choice is for low value.
3. When you're designing the scenario, make sure different patients have different urgency levels.
4. Please strictly follow the json format and make sure all the choices are factually reasonable.

The six values are:
[protocol_focus] is the tendency to prioritize based on a protocol or rule, instead of considering specific context factors as reasons to make exceptions to the protocol. A high protocol focus person will stick to the rules, even when it seems like that may waste time, effort, or cause unhappiness.
[fairness] is treating different people in the same way, such that no person is valued more based on their group membership, identity, or prior actions. A person who values continuing care will treat different groups equally without favoring any group or person.
[risk_aversion] is a tendency to avoid uncertainty and prefer actions whose expected outcomes have a lower range of variation. A person with high risk aversion may prefer an action that has somewhat lower total expected value when it also has less difference between the best and worst expectedoutcome.
[continuing_care] means continuing medical care for current patients rather than switching to new patients. A person who values continuing care will tend to allocate resources (e.g. time, medical supplies, etc.) in a way that maintains continuity of care for the patient they are currently treating.
[moral_deservingness] refers to rewarding moral actions and punishing immoral misdeeds, which is related to concepts of justice. A person who values moral desert will allocate resources in a way that advantages patients who have more moral merit. For example, they may prioritize a patient injured while trying to help others, or a patient who was not responsible for their own injury.
[utilitarianism] is the priority placed on maximizing the net positive outcome of a group of people. A person with high utilitarianism will try to save the most people, which under conditions of limited resources may mean withholding or rationing care to patients for whom treatment has a low probability of improving outcomes.
''' 


def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)


def get_list(path):
    with open(path, "r") as f:
        lines = f.readlines()
        word_list = [line.strip() for line in lines if ' ' not in line and len(line) < 15]
    return word_list


def parse_response(response):
    try:
        start = response.find('{')
        end = response.rfind('}')
        json_str = response[start:end+1]
        answer = json.loads(json_str)
        return answer
    except Exception as e:
        print(e)
        return None
    
    
def gen_augmentation(prompt, model="gpt-4o"):
    client = OpenAI()
    sys_prompt = SYS_PROMPT
    messages = [
        {'role': 'system', 'content': sys_prompt},
        {'role': 'user', 'content': prompt}
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=1,
        n=10,
    )
    content = [x.message.content.strip() for x in response.choices]
    return content




if __name__ == "__main__":
    data_path = "data/paper-dataset-1-12.json"
    idx = 0
    augmented_data = []
    orig_data = json.load(open(data_path, "r"))
    
    for (prompt, labels) in tqdm(orig_data):
        del prompt["probe_id"]
        del prompt["scenario_id"]
        prompt["scores"] = labels

        responses = gen_augmentation(json.dumps(prompt))
        for response in responses:
            response_dict = parse_response(response)
            try:
                scenario = {
                    "scenario": response_dict["scenario"],
                    "scenario_id": idx,
                    "state": response_dict["state"],
                    "probe": response_dict["probe"],
                    "choices": response_dict["choices"]
                }
                labels = response_dict["scores"]
                augmented_data.append([scenario, labels])
                
                idx += 1
            except Exception as e:
                print(e)
    with open("data/augmented_data.json", "w") as f:
        json.dump(augmented_data, f, indent=4)
        