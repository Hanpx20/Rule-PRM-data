from openai import OpenAI
import json
import re
from tqdm import tqdm
import string

SYS_PROMPT = '''You are a helpful questionnaire designer.
You will be given six values and a piece of news. You need to:
1. summarize the situation mentioned in the news as a decision-making scenario.
2. Choose one value that is most relevant to the scenario.
3. Propose two choices that people with high and low values will make. Each choice includes a description, a value (1 or 0), and a reason.
You should output in json format.

The six values are:
[protocol_focus] is the tendency to prioritize based on a protocol or rule, instead of considering specific context factors as reasons to make exceptions to the protocol. A high protocol focus person will stick to the rules, even when it seems like that may waste time, effort, or cause unhappiness.
[fairness] is treating different people in the same way, such that no person is valued more based on their group membership, identity, or prior actions. A person who values continuing care will treat different groups equally without favoring any group or person.
[risk_aversion] is a tendency to avoid uncertainty and prefer actions whose expected outcomes have a lower range of variation. A person with high risk aversion may prefer an action that has somewhat lower total expected value when it also has less difference between the best and worst expectedoutcome.
[continuing_care] means continuing medical care for current patients rather than switching to new patients. A person who values continuing care will tend to allocate resources (e.g. time, medical supplies, etc.) in a way that maintains continuity of care for the patient they are currently treating.
[moral_deservingness] refers to rewarding moral actions and punishing immoral misdeeds, which is related to concepts of justice. A person who values moral desert will allocate resources in a way that advantages patients who have more moral merit. For example, they may prioritize a patient injured while trying to help others, or a patient who was not responsible for their own injury.
[utilitarianism] is the priority placed on maximizing the net positive outcome of a group of people. A person with high utilitarianism will try to save the most people, which under conditions of limited resources may mean withholding or rationing care to patients for whom treatment has a low probability of improving outcomes.

Remind that:
1. When writing the scenario and choices, state the objective facts without directly mentioning any values (e.g. DON'T write "... without considering disparities in healthcare").
2. You can add details not present in the news to provide more solid information for the decision-making scenario (e.g. you can include details about a solder's wound to assess its severity).
3. Please anonymize any entity mentioned in the news (e.g. use "a country" instead of "America").
''' 

shot = [
    ["A sudden storm at a local concert caused severe injuries as debris from the damaged stage struck attendees. In the first aid tent, a medic faced an agonizing choice: use all remaining hemostatic gauze to save one critically injured patient with multiple wounds or stop treatment to save four others, each with life-threatening injuries. With ambulances delayed by blocked roads, the crisis underscored the dire challenges of limited resources in emergency situations.",
    '''{
    "scenario": "You are staffing a first aid tent at a local concert. A sudden storm produces strong winds that damage the stage and send metal and glass debris flying. You have only 5 packages of hemostatic gauze to stop severe bleeding. The first patient that arrives has several serious injuries, and you estimate that you will need all 5 packages because of the multiple locations of their injuries. You have stopped bleeding in one area when 4 other casualties arrive. Each has one seriously injured area that requires bandaging to stop bleeding. Because the storm damaged trees along the road, ambulances will not be able to reach the venue in the near future. If you continue to treat the first more seriously injured patient, you will use all 4 remaining packages to save his life, but the other patients would die before ambulances arrive. If you stop treating the current patient, allowing him to die, you could save the other 4 injured casualties with the 4 remaining packages.",
    "value": "continuing_care",
    "choices": [
        {
            "description": "Use the 4 packages to treat the other 4 patients",
            "value": 0,
            "reason": "Use the 4 packages to treat the other 4 patients is contraty to the idea of continuing care. This choice would abandon the patient whose life you have already started saving, which violates the ethical value of nonmaleficence (do no harm)."
        },
        {
            "description": "Use the 4 packages to treat the more seriously injured patient",
            "value": 1,
            "reason": "Continuing to treat the more seriously injured patient aligns with the principle of continuing care. This choice ensures that you do not abandon a patient whose life you have already started saving, upholding the ethical value of nonmaleficence (do no harm)."
        }
    ]
}''']
]


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
    
    
def categorize(prompt, model="gpt-4o-mini"):
    client = OpenAI()
    sys_prompt = "You are a helpful assistant. You need to determine if the following news title is related to the medical domain. Response with a single yes or no."
    
    response = client.chat.completions.create(
        model=model,
        messages = [
            {'role': 'system', 'content': sys_prompt},
            {'role': 'user', 'content': prompt}
        ],
        temperature=0
    )
    content = response.choices[0].message.content
    return content.strip().lower() == "yes"

    
def gen_scenario(prompt, model="gpt-4o"):
    client = OpenAI()
    sys_prompt = SYS_PROMPT
    messages = [
        {'role': 'system', 'content': sys_prompt}]
    for user, assis in shot:
        messages.append({'role': 'user', 'content': user})
        messages.append({'role': 'assistant', 'content': assis})
    messages.append({'role': 'user', 'content': prompt})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        n=1,
    )
    content = response.choices[0].message.content
    return content.strip()




if __name__ == "__main__":
    data_path = "VOA/data.jsonl"
    idx = 0
    synthesized_data = []
    medical_word_list = set(get_list("medical-wordlist/en/wordlist.en.txt"))
    
    cnt = 0
    with open(data_path, "r") as f:
        for line in tqdm(f):
            cnt += 1
            if cnt < 300:
                continue
            Data = json.loads(line)
            prompt = Data["text"][:3000]
            if not categorize(prompt):
                continue
            response = gen_scenario(prompt)
            response_dict = parse_response(response)
            if response_dict:
                scenario = {
                    "original_news": prompt,
                    "scenario_id": idx,
                    "scenario": response_dict["scenario"],
                    "state": None,
                    "probe": "What would you do?",
                    "choices": [x["description"] for x in response_dict["choices"]],
                    "explanations": [x["reason"] for x in response_dict["choices"]],
                }
                labels = [{response_dict["value"]: x["value"] * 10} for x in response_dict["choices"]]
                synthesized_data.append([scenario, labels])
                
                idx += 1
                print(f"Generated {idx} scenarios.")
    with open("data/synthesized_data.json", "a") as f:
        json.dump(synthesized_data, f, indent=4)
        