import json
import math
import requests
from utils import ask_ai, finish_run, start_run
import pandas as pd
import traceback
import time

trials_processed = 0
matches_processed = 0


def get_trial_data(trial_id):
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
    # check the other ID formats to get the data
    return None


# load XLSX file and get the IDs from the first column and call get_trial_data for each ID

def get_trials_data_from_xlsx(file_path, limit=False):
    try:
        with open(f'pro_results.json', 'r') as file:
            data = json.load(file)
    except:
        data = {}
    df = pd.read_excel(file_path)
    trial_ids = df['Registrationnumber'].tolist()
    unique_ids = df['Unique.ID'].tolist()
    counter = 0
    for trial_id, unique_id in zip(trial_ids, unique_ids):
        if unique_id in data:
            continue
        counter += 1
        print("Processing trials data for", trial_id)
        get_trial_data(trial_id)
        extract_pros(trial_id, unique_id)
        if limit and counter >= limit:
            break


def extract_pros(trial_id, unique_id):
    global trials_processed
    try:
        with open(f'pro_results.json', 'r') as file:
            global_results = json.load(file)
    except:
        global_results = {}
    if unique_id in global_results.keys() and "outcomes_ai" in global_results[unique_id]:
        return None
    
    try:
        study_data_path = f"trials/trial_{trial_id}.json"
        with open(study_data_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        return None
    
    
    outcomes = []
    for d in data["protocolSection"]["outcomesModule"]["primaryOutcomes"]:
        d["is_primary"] = True
        outcomes.append(d)

    if 'secondaryOutcomes' in data["protocolSection"]["outcomesModule"]:
        for d in data["protocolSection"]["outcomesModule"]["secondaryOutcomes"]:
            d["is_primary"] = False
            outcomes.append(d)
    title = data["protocolSection"]["identificationModule"]["briefTitle"]
    results = []
    for i, outcome in enumerate(outcomes):
        # print(outcome)
        answer = json.loads(ask_ai(
            f"In a clinical trial, a patient-reported outcome (PRO) is any information about a patient's health condition that comes partially from the patient themselves, i.e. a subjective report. Below, you receive a JSON string of an Outcome Measure in the clinical trial '{title}'. Please respond if this outcome is PRO or not (partially PRO is still considered a PRO, such as ARC20), specify the instrument used if any and give the reason for your assessment as a JSON string with keys 'is_pro', 'reason' and 'instrument'.\n{outcome}",
            system_role="You are an expert clinical analyst specialized in assessing integrity of clinical trial data",
            model="gpt-4o",
            json_mode=True
        ))
        answer["outcome"] = outcome
        answer["number"] = i + 1
        results.append(answer)


    path = f"results/trial_{trial_id}_pros.json"
    with open(path, "w") as f:
        f.write(json.dumps(results, indent=4))
    
    with open(f'pro_results.json', 'a') as _:
        pass

    try:
        with open(f'pro_results.json', 'r') as file:
            data = json.load(file)
    except:
        data = {}

    if unique_id not in data:
        data[unique_id] = {"study_data_path": study_data_path, "title": title}
    data[unique_id]["outcomes_ai"] = results

    with open(f'pro_results.json', 'w') as file:
        json.dump(data, file, indent=4)

    trials_processed += 1
    return results

def compile_results_data(file_path):
    with open(f'pro_results.json', 'r') as file:
        results = json.load(file)
    
    df = pd.read_excel(file_path)
    protocol_sec_names = []
    protocol_sec_instruments = []
    for i in range(1, 22):
        protocol_sec_names.append(f"Protocol_sec_outcome{i}")
        protocol_sec_instruments.append(f"Protocol_sec_instrument{i}")
    
    protocol_sec_pub_names = []
    protocol_sec_pub_instruments = []
    for j in range(10):
        protocol_sec_pub_names.append(f"pub_pro_sec_{chr(ord('a')+j)}_name")
        protocol_sec_pub_instruments.append(f"pub_pro_sec_{chr(ord('a')+j)}_ins")

    for index, row in df.iterrows():
        unique_id = row['Unique.ID']
        if unique_id not in results:
            continue
        
        results[unique_id]["outcomes_ethical"] = [{
            "number": i+1, "name": row[protocol_sec_names[i]], "instrument": row[protocol_sec_instruments[i]], "is_primary": False}
            for i in range(len(protocol_sec_names))
            if row[protocol_sec_names[i]] and str(row[protocol_sec_names[i]]) != "nan" and str(row[protocol_sec_names[i]]) != "."]
        results[unique_id]["outcomes_publication"] = [
            {"number": i+1, "name": row[protocol_sec_pub_names[i]], "instrument": row[protocol_sec_pub_instruments[i]], "is_primary": False}
            for i in range(len(protocol_sec_pub_names))
            if row[protocol_sec_pub_names[i]] and str(row[protocol_sec_pub_names[i]]) != "nan" and str(row[protocol_sec_pub_names[i]]) != "."]
        
        # Adding the primary outcome to the data
        if row["Protocol_PrimaryOutcome"] and str(row["Protocol_PrimaryOutcome"]) != "nan":
            results[unique_id]["outcomes_ethical"].append({"number": len(protocol_sec_names)+1, "name": row["Protocol_PrimaryOutcome"], "instrument": "", "is_primary": True})

        if row["Pub_PrimaryOutcome"] and str(row["Pub_PrimaryOutcome"]) != "nan":
            results[unique_id]["outcomes_publication"].append({"number": len(protocol_sec_pub_names)+1, "name": row["Pub_PrimaryOutcome"], "instrument": "", "is_primary": True})
        
        
    with open(f'pro_results.json', 'w') as file:
        json.dump(results, file, indent=4)


def match_results():
    global matches_processed
    print("Matching results")
    with open(f'pro_results.json', 'r') as file:
        results = json.load(file)
    
    for unique_id, data in results.items():
        if "outcomes_ai" in data and "outcomes_ethical" in data and "outcomes_publication" in data and "matching" not in data:
            
            ethical_matches = []
            ethical_additional = []
            publication_matches = []
            publication_additional = []
            text = "Below, you receive a JSON list of Outcome Measures in the clinical trial '%s':\n%s\n\nOne of the above outcomes should match with this outcome, which has been formulated differently:\Measure: %s\nDescription: %s\nInstrument: %s\n\nPlease indicate the best matching number (-1 if no match) in a JSON dictionary with the key 'match_number'. In addition, report with the boolean key 'has_changed' if the Outcome Measure is significantly different (measure or instrument has changed).\n"
            for i, outcome_ai in enumerate(data["outcomes_ai"]):
                if outcome_ai["is_pro"]:
                    # Match registry measures with ethical submission measures
                    reserved_numbers_ethical = [m["match"] for m in ethical_matches]
                    prompt = text % (
                        data["title"],
                        json.dumps([o for o in data["outcomes_ethical"] if o["number"] not in reserved_numbers_ethical], indent=4),
                        outcome_ai["outcome"]["measure"],
                        outcome_ai["outcome"].get("description", "No description"),
                        outcome_ai["instrument"]
                    )
                    answer = json.loads(ask_ai(
                        prompt,
                        system_role="You are an expert clinical analyst specialized in assessing integrity of clinical trial data",
                        model="gpt-4o",
                        json_mode=True
                    ))
                    if answer["match_number"] > -1:
                        element = outcome_ai.copy()
                        element["match"] = answer["match_number"]
                        element["has_changed"] = answer["has_changed"]
                        ethical_matches.append(element)
                    else:
                        ethical_additional.append(outcome_ai)

                    # Match registry measures with publication measures
                    reserved_numbers_publication = [m["match"] for m in publication_matches]
                    prompt = text % (
                        data["title"],
                        json.dumps([o for o in data["outcomes_publication"] if o["number"] not in reserved_numbers_publication], indent=4),
                        outcome_ai["outcome"]["measure"],
                        outcome_ai["outcome"].get("description", "No description"),
                        outcome_ai["instrument"]
                    )
                    answer = json.loads(ask_ai(
                        prompt,
                        system_role="You are an expert clinical analyst specialized in assessing integrity of clinical trial data",
                        model="gpt-4o",
                        json_mode=True
                    ))
                    if answer["match_number"] > -1:
                        element = outcome_ai.copy()
                        element["match"] = answer["match_number"]
                        element["has_changed"] = answer["has_changed"]
                        publication_matches.append(element)
                    else:
                        publication_additional.append(outcome_ai)
            
            leftover_outcomes_ethical = [o for o in data["outcomes_ethical"] if o["number"] not in [m["match"] for m in ethical_matches]]
            leftover_outcomes_publication = [o for o in data["outcomes_publication"] if o["number"] not in [m["match"] for m in publication_matches]]

            # a-j has been translated to numbers
            results[unique_id]["matching"] = {
                    "outcomes_in_registry_matching_ethical":  ethical_matches,
                    "extra_outcomes_in_registry_wrt_ethical":  ethical_additional,
                    "missing_outcomes_in_registry_wrt_ethical":  leftover_outcomes_ethical,
                    "modified_outcomes_in_registry_wrt_ethical":  [m for m in ethical_matches if m["has_changed"]],
                    "outcomes_in_registry_matching_publication":  publication_matches,
                    "extra_outcomes_in_registry_wrt_publication":  publication_additional,
                    "missing_outcomes_in_registry_wrt_publication":  leftover_outcomes_publication,
                    "modified_outcomes_in_registry_wrt_publication":  [m for m in publication_matches if m["has_changed"]],
                }

            if not results[unique_id]["matching"]["extra_outcomes_in_registry_wrt_ethical"] and not results[unique_id]["matching"]["missing_outcomes_in_registry_wrt_ethical"] and not results[unique_id]["matching"]["modified_outcomes_in_registry_wrt_ethical"]:
                results[unique_id]["matching"]["ethical_match_ai"] = True
            else:
                results[unique_id]["matching"]["ethical_match_ai"] = False
            if not results[unique_id]["matching"]["extra_outcomes_in_registry_wrt_publication"] and not results[unique_id]["matching"]["missing_outcomes_in_registry_wrt_publication"] and not results[unique_id]["matching"]["modified_outcomes_in_registry_wrt_publication"]:
                results[unique_id]["matching"]["publication_match_ai"] = True
            else:
                results[unique_id]["matching"]["publication_match_ai"] = False

            matches_processed += 1
            if not (results[unique_id]["matching"]["publication_match_ai"] and results[unique_id]["matching"]["ethical_match_ai"]):
                print(f"Registry entry didn't match the data for entry: {unique_id}: {data['title']}")

            with open(f'pro_results.json', 'w') as file:
                json.dump(results, file, indent=4)


def convert_results_to_csv(input_file, output_file):
    with open(input_file, 'r') as file:
        results = json.load(file)
    
    separator = ';'
    output_data = [separator.join([
        "unique_id",
        "ethical_match_ai",
        "publication_match_ai",
        "extra_outcomes_in_registry_wrt_ethical",
        "missing_outcomes_in_registry_wrt_ethical",
        "modified_outcomes_in_registry_wrt_ethical",
        "extra_outcomes_in_registry_wrt_publication",
        "missing_outcomes_in_registry_wrt_publication",
        "modified_outcomes_in_registry_wrt_publication",
        "comments"])]
    for unique_id, data in results.items():
        row = []
        row.append(unique_id)
        row.append(data["matching"]["ethical_match_ai"])
        row.append(data["matching"]["publication_match_ai"])
        comments = []
        if len(data["matching"]["extra_outcomes_in_registry_wrt_ethical"]):
            row.append(True)
            comments.append("Registry has additional PRO outcomes that are not found in the ethical submission")
        else:
            row.append(False)

        if len(data["matching"]["missing_outcomes_in_registry_wrt_ethical"]):
            row.append(True)
            comments.append("Registry is missing PRO outcomes that are present in the ethical submission")
        else:
            row.append(False)

        if len(data["matching"]["modified_outcomes_in_registry_wrt_ethical"]):
            row.append(True)
            comments.append("Registry has outcomes matching the ethical submission but they might have been modified")
        else:
            row.append(False)

        if len(data["matching"]["extra_outcomes_in_registry_wrt_publication"]):
            row.append(True)
            comments.append("Registry has additional PRO outcomes that are not found in the publication")
        else:
            row.append(False)

        if len(data["matching"]["missing_outcomes_in_registry_wrt_publication"]):
            row.append(True)
            comments.append("Registry is missing PRO outcomes that are present in the publication")
        else:
            row.append(False)

        if len(data["matching"]["modified_outcomes_in_registry_wrt_publication"]):
            row.append(True)
            comments.append("Registry has outcomes matching the publication but they might have been modified")
        else:
            row.append(False)

        if comments:
            row.append(". ".join(comments) + ".")
        else:
            row.append("")

        output_data.append(separator.join([str(r) for r in row]))
    output_data = "\n".join(output_data)

    with open(output_file, 'w') as file:
        file.write(output_data)


start_run()
try:
    get_trials_data_from_xlsx("ASPIRE_2012_OSKARI.xlsx")
    compile_results_data("ASPIRE_2012_OSKARI.xlsx")
    match_results()
    convert_results_to_csv("pro_results.json", "pro_results.csv")
except Exception as e:
    print("An exception occurred:", str(e))
    traceback.print_exc()
finally:
    print("Trials processed:", trials_processed, "Matches processed:", matches_processed)
    finish_run()
