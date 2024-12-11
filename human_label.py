import json
from termcolor import cprint
from sklearn.metrics import cohen_kappa_score


def label(data_path): # return human labels, save human labels, and return gpt-4 labels
    X = json.load(open(data_path, "r"))
    human_labels = []
    gpt_labels = []
        
    cprint("你将会看到若干决策场景，每个决策场景有一个对应的价值观和若干选项。对于每个选项，请你判断它体现了高的对应价值观还是低的对应价值观。对每个选项输出一个0或1，用空格分隔。", "blue")
    input("按任意键开始标注")
    for idx, (query, label) in enumerate(X):
        cprint(f"No. {idx}", "red")
        cprint("场景描述：", "blue")
        print(query["scenario"] + '\n' + (query["state"] if query["state"] else ""))
        cprint("场景选项：", "blue")
        print('\n'.join(['- ' + x for x in query["choices"]]))
        value = list(label[0].keys())[0]
        cprint("此场景判断的价值观：", "blue")
        print(value)
        cprint("请给出你的判断：", "blue")
        gpt_scores = [x[value] for x in label]

        while True:
            human_label = input()
            try:
                human_scores = [(int)(x) for x in human_label.split()]
            except:
                print("Invalid input. Please input again.")
                continue
            if len(human_scores) != len(gpt_scores):
                print("Invalid input. Please input again.")
                continue
            if any([x != 0 and x != 1 for x in human_scores]):
                print("You should only input 0 or 1. Please input again.")
                continue
            break
        
        human_labels.append(human_scores)
        gpt_labels.append([(int)(x/ 10) for x in gpt_scores])            
    
    json.dump(human_labels, open("data/human_labels.json", "w"), indent=4)
    return human_labels, gpt_labels
        
        


def calc_kappa(target, prediction):
    target = [x for y in target for x in y]
    prediction = [x for y in prediction for x in y]
    kappa = cohen_kappa_score(target, prediction)
    cprint(f"Cohen's Kappa: {kappa}", "blue")
    return kappa

if __name__ == "__main__":
    data_path = "data/sample_data.json"
    human_label, gpt_label = label(data_path)
    calc_kappa(human_label, gpt_label)