import json
import requests
from utils import ask_ai, finish_run, start_run
import pandas as pd
import traceback
import time

trials_processed = 0


def get_trial_data(trial_id):
    print("Finding trials data for", trial_id)
    try:
        trial_ids = trial_id.split(",")
    except:
        return None
    
    for id in trial_ids:
        if "NCT" in id:
            trial_id = id
            break
    if "NCT" in id:
        url = f"https://clinicaltrials.gov/api/v2/studies/{trial_id}"
        try:
            response = requests.get(url, timeout=5).json()
        except requests.exceptions.JSONDecodeError as e:
            print("Error in looking for trials dara for: ", trial_id)
            return None
        with open(f"trials/trial_{trial_id}.json", "w") as f:
            f.write(json.dumps(response, indent=4))
        return response
    return None


# load XLSX file and get the IDs from the first column and call get_trial_data for each ID

def get_trials_data_from_xlsx(file_path):
    df = pd.read_excel(file_path)
    trial_ids = df['Registrationnumber'].tolist()
    unique_ids = df['Unique.ID'].tolist()
    for trial_id, unique_id in zip(trial_ids, unique_ids):
        with open(f'pro_results.json', 'r') as file:
            data = json.load(file)
        if unique_id in data:
            continue
        get_trial_data(trial_id)
        extract_pros(trial_id, unique_id)


def extract_pros(trial_id, unique_id):
    global trials_processed
    
    try:
        with open(f"trials/trial_{trial_id}.json", "r") as f:
            data = json.load(f)
    except Exception as e:
        return None
    
    if 'secondaryOutcomes' not in data["protocolSection"]["outcomesModule"]:
       outcomes = data["protocolSection"]["outcomesModule"]["primaryOutcomes"]
    else:
        outcomes = data["protocolSection"]["outcomesModule"]["primaryOutcomes"] + data["protocolSection"]["outcomesModule"]["secondaryOutcomes"]
    title = data["protocolSection"]["identificationModule"]["briefTitle"]
    results = []
    for outcome in outcomes:
        # print(outcome)
        answer = json.loads(ask_ai(
            f"In a clinical trial, a patient-reported outcome (PRO) is any information about a patient's health condition that comes directly from the patient themselves, without interpretation by a doctor or anyone else. Below, you receive a JSON string of an Outcome Measure in the clinical trial '{title}'. Please respond if this outcome is PRO or not, specify the instrument used if any and give the reason for your assessment as a JSON string with keys 'is_pro', 'reason' and 'instrument'.\n{outcome}",
            system_role="You are an expert clinical analyst specialized in assessing integrity of clinical trial data",
            model="gpt-4o",
            json_mode=True
        ))
        answer["outcome"] = outcome
        results.append(answer)
        # print(f"{answer['is_pro']} - {outcome['measure']}")
        print(json.dumps(answer, indent=4))

    with open(f"results/trial_{trial_id}_pros.json", "w") as f:
        f.write(json.dumps(results, indent=4))
    
    with open(f'pro_results.json', 'a') as _:
        pass

    try:
        with open(f'pro_results.json', 'r') as file:
            data = json.load(file)
    except:
        data = {}

    if unique_id not in data:
        data[unique_id] = {}
    data[unique_id]["ai_results"] = results

    with open(f'pro_results.json', 'w') as file:
        json.dump(data, file, indent=4)

    trials_processed += 1
    return results

def match_results():
    with open(f'pro_results.json', 'r') as file:
        data = json.load(file)
    for key in data:
        print(key)
        print(data[key])
        

start_run()
try:
    get_trials_data_from_xlsx("ASPIRE_2012_OSKARI.xlsx")
except Exception as e:
    print("An exception occurred:", str(e))
    traceback.print_exc()
finally:
    print("Trials processed:", trials_processed)
    finish_run()
# get_trial_data("NCT00600756")
# extract_pros("NCT00600756")