import os
import random
import time
from decouple import config
from fp.fp import FreeProxy
import requests
import json
import replicate
import tiktoken
from openai import OpenAI
import re
import threading

DEBUG = False


def parse_json_from_file(filename):
    with open(filename, 'r') as file:
        return json.load(file)
    

def ask_ai(content, system_role=None, model=False, json_mode=False):
    if not model:
        model, tokens, difficulty = choose_model(content, json_mode)
    else:
        tokens = count_tokens(content)
        difficulty = None
    if tokens > 100000 or (tokens > 10000 and DEBUG):
        if prompt(f"The prompt length is {tokens} tokens. Would you like to shorten it to 8k tokens?", default=True):
            content = trucate_to_tokens(content, 8000)
            tokens = count_tokens(content)
    say(f"Asking {model}: {content[:150]}... [{tokens} tokens, {difficulty}]")
    if model == 'gemini-pro':
        gemini_response = query_gemini(content, system_role)
        if gemini_response:
            return gemini_response
        return ask_ai(content, system_role, 'gpt-4o')
    elif model == 'llama3':
        llama3_response = query_llama3(content, system_role)
        if llama3_response:
            return llama3_response
        return ask_ai(content, system_role, 'gpt-4o')
    elif model == 'gpt-3.5-turbo-0125':
        openai_response = query_openai(content, system_role, 'gpt-3.5-turbo-0125', json_mode)
        if openai_response:
            return openai_response
        return ask_ai(content, system_role, 'gpt-4o')
    elif model == 'gpt-4o':
        openai_response = query_openai(content, system_role, 'gpt-4o', json_mode)
        if openai_response:
            return openai_response
        raise Exception("Failed to get a response from OpenAI API")
    
    return False


def count_tokens(string: str, encoding_name="cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def choose_model(content, json_mode=False):
    tokens = count_tokens(content)
    prompt_start = " ".join(content.split(" ")[:200])
    difficulty = estimate_prompt_difficulty(prompt_start)

    if not difficulty:
        return 'llama3', tokens, difficulty
    if difficulty == "easy":
        if tokens < 8000 and not json_mode:
            model = 'llama3'
        elif tokens < 15500:
            model = 'gpt-3.5-turbo-0125'
        elif tokens < 30000:
            model = 'gpt-4o'
        else:
            model = 'gemini-pro'
    elif difficulty == "moderate":
        if tokens < 8000 and not json_mode:
            model = 'llama3'
        elif tokens < 30000:
            model = 'gpt-4o'
        else:
            model = 'gemini-pro'
    else:
        if tokens < 30000:
            model = 'gpt-4o'
        else:
            model = 'gemini-pro'
    
    return model, tokens, difficulty
            

def estimate_prompt_difficulty(prompt, tries=0):
    response = query_openai(prompt, "Please estimate the difficulty and complexity grade of the following prompt. If the prompt is too long, you will only receive the beginning. Only provide a JSON string with the following keys: 'difficulty' enumerating options (easy, moderate, hard)", model='gpt-3.5-turbo-0125', json_mode=True)
    try:
        response = json.loads(response)
    except:
        response = False
    if response and response.get("difficulty") in ["easy", "moderate", "hard"]:
        return response.get("difficulty")
    if tries > 3:
        return False
    return estimate_prompt_difficulty(prompt, tries+1)


def query_gemini(content, system_role=None):
    # Load the last working proxy from the json file
    try:
        with open('last_proxy.json', 'r') as file:
            last_proxy = json.load(file)
    except FileNotFoundError:
        last_proxy = None

    # Use the last working proxy if available
    if last_proxy:
        proxy = last_proxy
    else:
        # Get a new proxy
        proxy = FreeProxy(country_id=['US'], https=True).get()

    # Save the current proxy to the json file
    with open('last_proxy.json', 'w') as file:
        json.dump(proxy, file)

    if system_role:
        prompt = f"Role description: {system_role}\n\nPrompt:\n{content}"
    else:
        prompt = content
    response = False
    for i in range(3):
        proxies = {
            'https': str(proxy)
        }
        url = f'https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent?key={config("GEMINI_API_KEY")}'
        data = {"contents": [{"parts": [{"text": prompt}]}]}
        try:
            response = requests.post(url, proxies=proxies, json=data, timeout=10)
            # print(response.text)
            if not response.status_code == 200:
                raise Exception("Failed to send request due to status code", response.status_code)
            with open('last_proxy.json', 'w') as file:
                json.dump(proxy, file)
            break
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                time.sleep(random.random()*5)
        except Exception as e:
            say("Failed to send request due to the following error:", e)
            proxy = FreeProxy(country_id=['US'], https=True).get()
    
    if response:
        try:
            return response.json()["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            say(response.text)
            say("Failed to parse response due to the following error:", e)
            return None
    return None


def query_llama3(content, system_role):
    if system_role is None:
        system_role = "You are a helpful assistant."
    os.environ["REPLICATE_API_TOKEN"] = config('REPLICATE_API_KEY')
    prompt = {
        "top_p": 0.95,
        "prompt": content,
        "system_prompt": system_role,
        "temperature": 0.7,
        "prompt_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        "presence_penalty": 0,
        "max_tokens": 2048,
    }
    output = False
    try:
        output = replicate.run(
            "meta/meta-llama-3-70b-instruct",
            input=prompt
        )
    except replicate.exceptions.ModelError as e:
        if "please retry" in str(e):
            output = replicate.run(
                "meta/meta-llama-3-70b-instruct",
                input=prompt
            )
    if output:
        return "".join(output)
    return output


def query_openai(content, system_role, model='gpt-4o', json_mode=False):
    if system_role is None:
        system_role = "You are a helpful assistant."

    ai_client = OpenAI(
        organization='org-OjCSzLcscYwYrWwsc7EWJZs7',
        project='proj_LL7UZDSKgr42Lp0P7QnO30i1',
        api_key=config('OPENAI_API_KEY')
    )
    
    messages = [
        {
            "role": "system",
            "content": system_role
        },
        {
            "role": "user",
            "content": content
        },
    ]

    try:
        response = ai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.6,
            response_format={ "type": "json_object" } if json_mode else { "type": "text" },
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

    except Exception as e:
        say(f"Error querying OpenAI API: {e}")
        return None
    
    _update_tokens(response.usage, model)
    
    if response:
        return response.choices[0].message.content
    else:
        return None


def trucate_to_tokens(prompt, max_tokens=8000):
    tokens = count_tokens(prompt)
    say("Tokens before truncation:", tokens)
    while tokens > max_tokens:
        prompt = " ".join(prompt.split(" ")[:int(max_tokens/tokens*len(prompt.split(" ")))-5])
        tokens = count_tokens(prompt)
    say("Tokens after truncation:", tokens)
    return prompt



def choice_menu(menu, title):
    """
    Display a menu of choices and prompt the user to make a selection.

    :param menu: list, the list of options to choose from
    :param title: str, the title of the menu
    :return: int|bool, the index of the chosen option or False if cancelled
    """
    print(title)
    for ind, option in enumerate(menu):
        print(f'{ind + 1}) {option}')
    print('q) Cancel')
    choice = input()
    while choice not in [str(i) for i in range(1, len(menu) + 1)] + ['q']:
        choice = input('Incorrect input. Try again:\n')
    return False if choice == 'q' else int(choice) - 1


def prompt(message, default=None):
    """
    Prompt the user with a yes/no question.

    :param message: str, the message to display
    :param default: bool, the default value if the user presses Enter
    :return: bool, the user's response
    """
    alternatives = ['N', 'n', 'y', 'Y', '0', '1']
    info = '\n[Y/n]: ' if default else '\n[y/N]: ' if default is False else '\n[y/n]: '
    choice = input(message + info).strip()
    while choice not in alternatives + ['']:
        print('Incorrect input. Please try again.')
        choice = input(message + info).strip()
    return {'N': False, 'n': False, 'y': True, 'Y': True, '0': False, '1': True}.get(choice, default)


def clean_string(input_string):
    # Replace whitespace with underscores
    underscore_string = input_string.replace(' ', '_')
    # Strip special characters
    stripped_string = re.sub(r'[^\w_]+', '', underscore_string)
    # Convert to lower case
    lower_case_string = stripped_string.lower()
    return lower_case_string


def cost_report():
    with open(f'run_details.json', 'r') as file:
        data = json.load(file)

    out = f"The current run has costed {round(data['current_run_cost'], 2)}$ so far while the project costs are {round(data['total_cost'], 2)}$ in total."
    print(out)
    return out

def load_profile(savefile, organization_name):
    """
    Load the organization profile from the savefile file.

    :param savefile: str, the path to the savefile file
    :param organization_name: str, the name of the organization
    :return: dict, the organization profile
    """
    # Check if the savefile file exists
    if not os.path.exists(savefile):
        # Create an empty dictionary to store profiles
        profiles = {}
    else:
        # Load existing profiles from the savefile file
        with open(savefile, "r") as file:
            profiles = json.load(file)["profiles"]

    # Check if the organization profile already exists
    if organization_name in profiles:
        # Profile already exists, do nothing
        profile = profiles[organization_name]
        return profile
    
    return False

def load_profiles(savefile):
    """
    Load the organization profiles from the savefile file.

    :param savefile: str, the path to the savefile file
    :return: dict, the organization profile
    """
    # Check if the savefile file exists
    if not os.path.exists(savefile):
        # Create an empty dictionary to store profiles
        profiles = {}
    else:
        # Load existing profiles from the savefile file
        with open(savefile, "r") as file:
            profiles = json.load(file)["profiles"]
        
    return profiles

def load_questions(savefile):
    """
    Load the questions.

    :param savefile: str, the path to the savefile file
    :return: dict, the organization profile
    """
    # Check if the savefile file exists
    if not os.path.exists(savefile):
        # Create an empty dictionary to store profiles
        questions = []
    else:
        # Load existing profiles from the savefile file
        with open(savefile, "r") as file:
            questions = json.load(file)["questions"]
        
    return questions

def save_question(savefile, question):
    """
    Save the question to the savefile file.

    :param savefile: str, the path to the savefile file
    :param profile: tuple, the question
    """
    # Load existing questions from the savefile file
    with open(savefile, "r") as file:
        data = json.load(file)

    # Save the question to the questions list
    if question[0] not in [q[0] for q in data["questions"]]:
        data["questions"].append(question)

    # Save the data to the savefile file
    with open(savefile, "w") as file:
        json.dump(data, file, indent=4)

def save_profile(savefile, organization_name, profile):
    """
    Save the organization profile to the savefile file.

    :param savefile: str, the path to the savefile file
    :param organization_name: str, the name of the organization
    :param profile: dict, the organization profile
    """
    lock = threading.Lock()  # Create a lock object

    with lock:
        # Load existing profiles from the savefile file
        with open(savefile, "r") as file:
            data = json.load(file)

        # Save the organization profile to the profiles dictionary
        data["profiles"][organization_name] = profile

        # Save the profiles dictionary to the savefile file
        with open(savefile, "w") as file:
            json.dump(data, file, indent=4)

def say(*args):
    if DEBUG:
        print(*args)
    else:
        pass

def start_run(project='default'):
    with open(f'run_details.json', 'a') as _:
        pass

    try:
        with open(f'run_details.json', 'r') as file:
            data = json.load(file)
    except:
        data = {project: {} }
    
    if project not in data.keys():
        data[project] = {}
    data[project]["current_run_cost"] = 0

    with open(f'run_details.json', 'w') as file:
        json.dump(data, file)


def _update_tokens(usage, model, project='default'):

    try:
        with open(f'run_details.json', 'r') as file:
            data = json.load(file)
    except:
        print("Please call start_run() if you want get up to date cost analysis.")
        return

    if "current_run_cost" not in data[project].keys():
        data[project]["current_run_cost"] = 0
    if "total_tokens" in data[project].keys():
        data[project]["total_tokens"] += usage.total_tokens
    else:
        data[project]["total_tokens"] = usage.total_tokens
    if "prompt_tokens" in data[project].keys():
        data[project]["prompt_tokens"] += usage.prompt_tokens
    else:
        data[project]["prompt_tokens"] = usage.prompt_tokens
    if "completion_tokens" in data[project].keys():
        data[project]["completion_tokens"] += usage.completion_tokens
    else:
        data[project]["completion_tokens"] = usage.completion_tokens

    if "total_cost" not in data[project].keys():
        data[project]["total_cost"] = 0
    if model == 'gpt-4-turbo-preview':
        cost_add = usage.prompt_tokens / 1000000 * 10.00 + usage.completion_tokens / 1000000 * 30.00
        data[project]["total_cost"] += cost_add
        data[project]["current_run_cost"] += cost_add
    elif model == 'gpt-4o':
        cost_add = usage.prompt_tokens / 1000000 * 5.00 + usage.completion_tokens / 1000000 * 15.00
        data[project]["total_cost"] += cost_add
        data[project]["current_run_cost"] += cost_add
    elif model == 'gpt-4':
        cost_add = usage.prompt_tokens / 1000000 * 30.00 + usage.completion_tokens / 1000000 * 60.00
        data[project]["total_cost"] += cost_add
        data[project]["current_run_cost"] += cost_add
    else:
        cost_add = usage.prompt_tokens / 1000000 * 10.00 + usage.completion_tokens / 1000000 * 30.00
        data[project]["total_cost"] += cost_add
        data[project]["current_run_cost"] += cost_add

    with open(f'run_details.json', 'w') as file:
        json.dump(data, file)



def finish_run(project='default'):
    with open(f'run_details.json', 'a') as _:
        pass

    try:
        with open(f'run_details.json', 'r') as file:
            data = json.load(file)
    except:
        data = {project: {}}

    if "current_run_cost" not in data[project].keys():
        data[project]["current_run_cost"] = 0

    if "total_cost" not in data[project].keys():
        data[project]["total_cost"] = 0

    print("The current run has costed", round(data[project]["current_run_cost"], 2), f"$ so far while the project '{project}' costs are", round(data[project]["total_cost"], 2), "$ in total.")

    data[project]["current_run_cost"] = 0

    with open(f'run_details.json', 'w') as file:
        json.dump(data, file)