# GitHub Automated Analysis

## Overview

This project is a Python-based tool that, when given a GitHub user's URL, returns the most technically complex and challenging repository from that user's profile. The tool uses GPT and LangChain to assess each repository individually before determining the most technically challenging one.

## Installation and Setup

To install and set up this project, follow these steps:

1. Clone the repository to your local machine.
2. Install the required dependencies by running `pip install -r requirements.txt`.
3. Set the `OPENAPI_KEY` and `GITHUB_TOKEN` environment variables with your OpenAI API key and GitHub personal access token, respectively.

## Usage

To use this project, follow these steps:

1. Run the `app.py` script using Python.
2. Open the web interface in your browser.
3. Enter a GitHub user's URL in the text box and click the "Analyze" button.
4. The interface will display a link to the most complex repository as well as GPT analysis justifying the selection.

## Implementation Details

This project is implemented using Python and several third-party libraries, including requests, base64, os, fnmatch, sys, json, openai, langchain, re, dotenv, tiktoken, and streamlit.

The main logic of the project is contained in several functions that perform tasks such as:

- Fetching a user's repositories from their GitHub user URL using the `get_repos` function.
- Preprocessing the code in repositories before passing it into GPT using functions such as `get_file_contents`, `get_dir_contents`, `get_ignore_list`, `should_ignore`, and `process_repository`.
- Implementing prompt engineering when passing code through GPT for evaluation to determine its technical complexity using the `get_complexity_score` function.
- Identifying which of the repositories is the most technically complex using the `main` function.
- Using GPT to justify the selection of the repository using the `main` function.
- Creating a simple web interface using Streamlit where users can enter a GitHub user URL and see the results of the analysis.

The project uses advanced techniques such as memory management for large repositories and prompt engineering to improve the accuracy of its complexity scoring.

## Results

This project is able to accurately determine the most technically complex repository from a given GitHub user's profile. It uses advanced techniques such as memory management for large repositories and prompt engineering to improve the accuracy of its complexity scoring.
