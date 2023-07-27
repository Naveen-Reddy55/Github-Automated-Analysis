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
    """
    Fetches a user's repositories from their GitHub user URL.
    
    :param username: The GitHub username of the user.
    :return: A list of repositories.
    """
    url = f'https://api.github.com/users/{username}/repos'
    response = requests.get(url,headers=headers)
    if response.status_code != 200:
        print(response.text)
        return []
    repos = response.json()
    return repos

def get_token_count(text, model_name="cl100k_base"):
    """
    Gets the token count of the text using the specified model.
    
    :param text: The text to get the token count for.
    :param model_name: The name of the model to use.
    :return: The token count of the text.
    """
    encoding = tiktoken.get_encoding(model_name)
    tokens = encoding.encode(text)
    num_tokens = len(tokens)

    return num_tokens

def get_file_contents(owner, repo, path):
    """
    Gets the contents of a file in a repository.
    
    :param owner: The owner of the repository.
    :param repo: The name of the repository.
    :param path: The path to the file in the repository.
    :return: The contents of the file.
    """
    url = f'https://api.github.com/repos/{owner}/{repo}/contents/{path}'
    response = requests.get(url, headers=headers)
    content = base64.b64decode(response.json()['content'])
    
    # Decode content if it is in bytes
    if isinstance(content, bytes):
        try:
            content = content.decode('utf-8')
        except UnicodeDecodeError:
            pass
    
    # Preprocess Jupyter notebooks
    if path.endswith('.ipynb'):
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
    """
    Gets the contents of a directory in a repository.
    
    :param owner: The owner of the repository.
    :param repo: The name of the repository.
    :param path: The path to the directory in the repository.
    :return: A list of items in the directory.
    """
    url = f'https://api.github.com/repos/{owner}/{repo}/contents/{path}'
    response = requests.get(url,headers=headers)
    if response.status_code != 200:
        print(response.text)
        return []
    contents = response.json()
    return contents

def get_ignore_list(ignore_file_path):
    """
    Gets the list of file patterns to ignore from the specified ignore file.
    
    :param ignore_file_path: The path to the ignore file.
    :return: A list of file patterns to ignore.
    """
    ignore_list = []
    with open(ignore_file_path, 'r') as ignore_file:
        for line in ignore_file:
            if sys.platform == "win32":
                line = line.replace("/", "\\")
            ignore_list.append(line.strip())
    return ignore_list

def should_ignore(file_path, ignore_list, allowed_extensions):
    """
    Determines if a file should be ignored based on the ignore list and allowed extensions.
    
    :param file_path: The path to the file.
    :param ignore_list: The list of file patterns to ignore.
    :param allowed_extensions: The list of allowed file extensions.
    :return: True if the file should be ignored, False otherwise.
    """
    
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
    """
    Processes a repository by getting its contents and filtering out ignored files.
    
    :param owner: The owner of the repository.
    :param repo: The name of the repository.
    :param ignore_list: The list of file patterns to ignore.
    :return: A string containing the contents of the repository's files.
    """
    allowed_extensions = [
    '.py', '.java', '.c', '.cpp', '.js', '.cs', '.php', '.rb', '.go',
    '.rs', '.kt', '.swift', '.m', '.h', '.scala', '.hs', '.sh', '.bat',
    '.pl', '.lua', '.tcl', '.r', '.jl', '.f90', '.f95', '.f03',
    '.sol', '.clj', '.ex', '.exs', '.elm', '.erl', '.fs', '.fsx',
    '.groovy', '.lisp', '.scm', '.ml', '.mli', '.nim',
    '.pas','pascal','.pp','.purs','.re','.rei','.ts','.tsx',
    'v','.vhdl','.vhd'
     ]
    
    output_text = ""
    
    def process_directory(path):
        """
        Processes a directory in the repository by getting its contents and filtering out ignored files.
        
        :param path: The path to the directory in the repository.
        """
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
    """
    Breaks up a file into chunks of the specified size with the specified overlap.
    
    :param text: The text to break up into chunks.
    :param chunk_size: The size of each chunk.
    :param overlap: The amount of overlap between chunks.
    :return: A list of chunks.
    """
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
    """
    Gets the complexity score of the chunks using GPT.
    
    :param chunks: The list of chunks to get the complexity score for.
    :return: The complexity score of the chunks.
    """
    if len(chunks)==1:
      
      prompt = f"""
        Please evaluate the technical complexity of this code on a scale of 1 to 10, where 1 is very simple and 10 is very complex. When making your evaluation, follow these steps:

        1. Analyze the code to identify its key components and features, such as data structures, algorithms, control structures, and programming concepts.

        2. Assess the complexity of each component and feature individually, taking into account factors such as algorithmic complexity, computational efficiency, and use of advanced programming techniques.

        3. Combine your assessments of the individual components and features to form an overall evaluation of the technical complexity of the code.

        4. Provide a detailed explanation for your evaluation, including specific examples from the code that support your assessment and a clear justification for your complexity score.

        5. Consider factors such as use of code organization, readability and maintainability, use of design patterns, computational efficiency, scalability, use of advanced programming languages and frameworks, optimization, use of cutting-edge technologies, and software architecture.

        To help you make your evaluation, here are some examples of code with different levels of complexity:

        Example 1 (Complexity: 1): This code calculates the sum of two numbers using the addition operator.

        Reasoning: This code is very simple because it only performs a basic arithmetic operation using a single operator. The algorithmic complexity is low because the calculation can be performed in constant time.

        Observations: The code is short and easy to understand. It does not use any advanced programming concepts or data structures.

        code example: ```
            def add(a: int, b: int) -> int:
                return a + b
                       ```

        Example 2 (Complexity: 9): This code implements a machine learning algorithm using advanced data structures and algorithms from scratch.

        Reasoning: This code is highly complex because it uses advanced data structures and algorithms to implement a machine learning algorithm, which requires a deep understanding of mathematical concepts and computational techniques. The algorithmic complexity is high because the machine learning operation takes longer for larger input values.

        Observations: The code is much longer and more difficult to understand than the previous examples. It uses advanced data structures and algorithms to implement the machine learning algorithm, which requires a deep understanding of these concepts.

        Example 3 (Complexity: 5): This code implements a binary search algorithm using recursion.

        Reasoning: This code is less complex than the previous examples because it uses an advanced algorithm (binary search) and a advanced programming concept (recursion) to perform a search operation. The algorithmic complexity is higher because the search operation takes longer for larger input values.

        Observations: The code is longer and but not that difficult to understand than the previous examples. It uses recursion to implement the binary search algorithm, which requires a deeper understanding of programming concepts.
        
        code example: ```

        def binary_search(arr: list[int], low: int, high: int, x: int) -> int:
            if high >= low:
                mid = (high + low) // 2
                if arr[mid] == x:
                    return mid
                elif arr[mid] > x:
                    return binary_search(arr, low, mid - 1, x)
                else:
                    return binary_search(arr, mid + 1, high, x)
            else:
                return -1

        Based on these examples and your own analysis of the given code, please provide a detailed evaluation of its technical complexity. Include specific examples from the code that support your assessment and explain your reasoning in detail. The structure of the text is as follows: sections begin with '-----', followed by a single line containing the file path and file name, followed by a variable number of lines containing the file contents. The text representing the Git repository ends when the symbols '--END--' are encountered and the following is the code: \n.{chunk}
        
        """

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

        prompt = f"""
        Please evaluate the technical complexity of this code on a scale of 1 to 10, where 1 is very simple and 10 is very complex and Please keep in mind that this is only a chunk of the full repository while evaluating its complexity. When making your evaluation, follow these steps:

        1. Analyze the code to identify its key components and features, such as data structures, algorithms, control structures, and programming concepts.

        2. Assess the complexity of each component and feature individually, taking into account factors such as algorithmic complexity, computational efficiency, and use of advanced programming techniques.

        3. Combine your assessments of the individual components and features to form an overall evaluation of the technical complexity of the code.

        4. Provide a detailed explanation for your evaluation, including specific examples from the code that support your assessment and a clear justification for your complexity score.

        5. Consider factors such as use of code organization, readability and maintainability, use of design patterns, computational efficiency, scalability, use of advanced programming languages and frameworks, optimization, use of cutting-edge technologies, and software architecture.

        To help you make your evaluation, here are some examples of code with different levels of complexity:

        Example 1 (Complexity: 1): This code calculates the sum of two numbers using the addition operator.

        Reasoning: This code is very simple because it only performs a basic arithmetic operation using a single operator. The algorithmic complexity is low because the calculation can be performed in constant time.

        Observations: The code is short and easy to understand. It does not use any advanced programming concepts or data structures.

        code example: ```
            def add(a: int, b: int) -> int:
                return a + b
                       ```

        Example 2 (Complexity: 9): This code implements a machine learning algorithm using advanced data structures and algorithms from scratch.

        Reasoning: This code is highly complex because it uses advanced data structures and algorithms to implement a machine learning algorithm, which requires a deep understanding of mathematical concepts and computational techniques. The algorithmic complexity is high because the machine learning operation takes longer for larger input values.

        Observations: The code is much longer and more difficult to understand than the previous examples. It uses advanced data structures and algorithms to implement the machine learning algorithm, which requires a deep understanding of these concepts.

        Example 3 (Complexity: 5): This code implements a binary search algorithm using recursion.

        Reasoning: This code is less complex than the previous examples because it uses an advanced algorithm (binary search) and a advanced programming concept (recursion) to perform a search operation. The algorithmic complexity is higher because the search operation takes longer for larger input values.

        Observations: The code is longer and but not that difficult to understand than the previous examples. It uses recursion to implement the binary search algorithm, which requires a deeper understanding of programming concepts.
        code example: ```

        def binary_search(arr: list[int], low: int, high: int, x: int) -> int:
            if high >= low:
                mid = (high + low) // 2
                if arr[mid] == x:
                    return mid
                elif arr[mid] > x:
                    return binary_search(arr, low, mid - 1, x)
                else:
                    return binary_search(arr, mid + 1, high, x)
            else:
                return -1
                 ```
        
        Based on these examples and your own analysis of the given code, please provide a detailed evaluation of its technical complexity. Include specific examples from the code that support your assessment and explain your reasoning in detail.. The text provided below is {num+1} chunk from a full Git repository containing code. The structure of the text is as follows: sections begin with '-----', followed by a single line containing the file path and file name, followed by a variable number of lines containing the file contents. The text representing the Git repository ends when the symbols '--END--' are encountered. Please keep in mind that this is only a chunk of the full repository while evaluating its complexity and the following is the code:\n.{chunk}
        """
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
    """
    The main function that takes a GitHub user URL and returns the most technically complex repository from that user's profile.
    
    :param user_url: The GitHub user URL.
    :return: A string containing the most complex repository and its complexity score.
    """
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
