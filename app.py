import requests
import base64
import os
import fnmatch
import sys
import json
import openai
import langchain
import re
from dotenv import load_dotenv
import tiktoken
import streamlit as st 

load_dotenv()
st.title('Github Automated Analysis using OpenAI and Langchain')


openai.api_key = os.getenv('OPENAPI_KEY')

headers = {
    'Authorization': str(os.getenv('GITHUB_TOKEN'))
}


def get_repos(username):
    url = f'https://api.github.com/users/{username}/repos'
    response = requests.get(url,headers=headers)
    if response.status_code != 200:
        print(response.text)
        return []
    repos = response.json()
    return repos

def get_token_count(text, model_name="cl100k_base"):
    encoding = tiktoken.get_encoding(model_name)
    tokens = encoding.encode(text)
    num_tokens = len(tokens)

    return num_tokens

def get_file_contents(owner, repo, path):
    url = f'https://api.github.com/repos/{owner}/{repo}/contents/{path}'
    response = requests.get(url, headers=headers)
    content = base64.b64decode(response.json()['content'])
    if isinstance(content, bytes):
        try:
            content = content.decode('utf-8')
        except UnicodeDecodeError:
            pass
    if path.endswith('.ipynb'):


    # Extract source code from Jupyter notebook
      try:
        data = json.loads(content)
      except json.JSONDecodeError:
        return ''
      source = ''
      for cell in data['cells']:
          if cell['cell_type'] == 'code':
              source += ''.join(cell['source']) + '\n'
      content = source
    return content

def get_dir_contents(owner, repo, path):
    url = f'https://api.github.com/repos/{owner}/{repo}/contents/{path}'
    response = requests.get(url,headers=headers)
    if response.status_code != 200:
        print(response.text)
        return []
    contents = response.json()
    return contents

def get_ignore_list(ignore_file_path):
    ignore_list = []
    with open(ignore_file_path, 'r') as ignore_file:
        for line in ignore_file:
            if sys.platform == "win32":
                line = line.replace("/", "\\")
            ignore_list.append(line.strip())
    return ignore_list

def should_ignore(file_path, ignore_list, allowed_extensions):
    # Ignore files based on ignore list
    for pattern in ignore_list:
        if fnmatch.fnmatch(file_path, pattern):
            return True
    
    # Ignore files that do not have an allowed extension
    file_extension = os.path.splitext(file_path)[1]
    if file_extension not in allowed_extensions:
        return True
    
    return False

def process_repository(owner, repo, ignore_list):
    allowed_extensions = [
    '.py', '.java', '.c', '.cpp', '.js', '.cs', '.php', '.rb', '.go',
    '.rs', '.kt', '.swift', '.m', '.h', '.scala', '.hs', '.sh', '.bat',
    '.pl', '.lua', '.tcl', '.r', '.jl', '.f90', '.f95', '.f03',
    '.sol', '.clj', '.ex', '.exs', '.elm', '.erl', '.fs', '.fsx',
    '.groovy', '.lisp', '.scm', '.ml', '.mli', '.nim',
    '.pas', 'pascal' ,'.pp','.purs','.re','.rei','.ts','.tsx',
    'v','.vhdl','.vhd'
     ]
    
    output_text = ""
    
    def process_directory(path):
        nonlocal output_text
        contents = get_dir_contents(owner, repo, path)
        for item in contents:
            if item['type'] == 'dir':
                process_directory(item['path'])
            elif item['type'] == 'file':
                relative_file_path = os.path.relpath(item['path'], '')
                if not should_ignore(relative_file_path, ignore_list, allowed_extensions):
                    file_contents = get_file_contents(owner, repo, item['path'])
                    output_text += "-" * 5 + "\n"
                    output_text += f"{relative_file_path}\n"
                    output_text += f"{file_contents}\n"
    
    process_directory('')
    output_text += f"--END--\n"
    return output_text


def break_up_file_to_chunks(text, chunk_size=2000, overlap=50):

    encoding = tiktoken.get_encoding("p50k_base")
    tokens = encoding.encode(text)
    num_tokens = len(tokens)
    if num_tokens<=2000:
      return [str(text)]
    else:
      chunks = []
      for i in range(0, num_tokens, chunk_size - overlap):
          chunk = tokens[i:i + chunk_size]
          chunk=encoding.decode(chunk)
          # chunk=" ".join
          chunks.append(chunk)
      
      return chunks

def get_complexity_score(chunks):

    # total_repo_text_token_count=get_token_count(text, model_name='gpt-4')
    # repo_text_chunks=[]

    # # output_file_index = 1
    # current_output_file = ""
    # written_token_count = 0
    # token_count = 0
    if len(chunks)==1:
      prompt = f"Please evaluate the complexity of the code in this Git repository on a scale of 1 to 10, where 1 is very simple and 10 is very complex. The following text is a Git repository with code. The structure of the text are sections that begin with -----, followed by a single line containing the file path and file name, followed by a variable amount of lines containing the file contents. The text representing the Git repository ends when the symbols --END-- are encounted.\n.{chunks[0]}"
      response = openai.Completion.create(
          engine="text-davinci-003",
          prompt=prompt,
          max_tokens=30,
          n=1,
          stop=None,
          temperature=0,
          # api_base="https://api.openai.com"
      )



      try:
          text_part_complexity=[]

          match = re.search(r'\d+', response['choices'][0]['text'].strip())
          if match:
            score = int(match.group())
          else:
            score=None

          # score = int(response['choices'][0]['text'].strip())
      except ValueError:
          score = None

      text_part_complexity.append(score)

      repo_complexity_score=sum(text_part_complexity)/len(text_part_complexity)

    else:


      text_part_complexity=[]
      
      for num,chunk in enumerate(chunks):

        prompt = f"Please evaluate the complexity of the code in this Git repository on a scale of 1 to 10, where 1 is very simple and 10 is very complex and just give straight forward number don't add anything like score/10, etc . The text provided below is {num+1} chunk from a full Git repository containing code. The structure of the text is as follows: sections begin with '-----', followed by a single line containing the file path and file name, followed by a variable number of lines containing the file contents. The text representing the Git repository ends when the symbols '--END--' are encountered. Please keep in mind that this is only a chunk of the full repository while evaluating its complexity.\n.{chunk}"
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=30,
            n=1,
            stop=None,
            temperature=0,
            # api_base="https://api.openai.com"
        )



        try:

          match = re.search(r'\d+', response['choices'][0]['text'].strip())
          if match:
            score = int(match.group())
          else:
            score=None

            # score = int(response['choices'][0]['text'].strip())
        except ValueError:
            score = None

        text_part_complexity.append(score)

      text_part_complexity=[x for x in text_part_complexity if x is not None]
      repo_complexity_score=sum(text_part_complexity)/len(text_part_complexity)
    return repo_complexity_score

def main(user_url):

    url = str(user_url)
    username = re.search(r"github.com/(.+)", url).group(1)
    repos = get_repos(username)

    ignore_list = get_ignore_list('C:/Users/Naveen Reddy/Downloads/Mercor Github Automated Analysis/.gptignore.txt')

    repo_scores = []
    for repo in repos:
        output_text = process_repository(username, repo['name'], ignore_list)
        # num_tokens=get_token_count(output_text)
        # if num_tokens<=2000:


        # else:
        chunks=break_up_file_to_chunks(output_text, chunk_size=1500, overlap=50)

        score = get_complexity_score(chunks)
        repo_scores.append((score, repo['name'], repo['html_url']))

    repo_scores.sort(reverse=True)

    most_complex_repo = repo_scores[0]
    return_to_display=f"The most complex repository is {most_complex_repo[1]} with a complexity score of {most_complex_repo[0]}. The URL for this repository is {most_complex_repo[2]}"
    
    return return_to_display

github_url = st.text_input('Please Type your github URL here.....') 

if github_url:
    display_value=main(github_url)

    st.write(display_value)