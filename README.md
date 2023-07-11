# Github Automated Analysis using OpenAI and Langchain

This project is a Python-based tool that, when given a GitHub user's URL, returns the most technically complex and challenging repository from that user's profile. The tool uses GPT and LangChain to assess each repository individually before determining the most technically challenging one.

## Requirements

- Fetch a user’s repositories from their GitHub user URL.
- Preprocess the code in repositories before passing it into GPT. Specifically, implement memory management techniques for large repositories and the files within them. Consider that repositories may contain large Jupyter notebooks or package files which, if passed raw through GPT, would greatly exceed token limits.
- Implement prompt engineering when passing code through GPT for evaluation to determine its technical complexity.
- Identify which of the repositories is the most technically complex. Use GPT to justify the selection of the repository.
- Deploy your solution on a hosting platform like Vercel, Netlify, or GitHub pages. The interface should include a simple text box where users can input a GitHub user URL for analysis. Then, the interface should display a link to the most complex repository as well as GPT analysis justifying the selection.
- Record a YouTube video demonstrating your tool in action, showcasing its ability to evaluate and compare repositories' complexity.

## Usage

1. Install dependencies by running `pip install -r requirements.txt`.
2. Set the `OPENAPI_KEY` and `GITHUB_TOKEN` environment variables with your OpenAI API key and GitHub personal access token respectively.
3. Run `streamlit run app.py` to start the Streamlit app.
4. Open the app in your browser and enter a GitHub user URL to analyze their repositories.