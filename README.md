
# An introduction to { YOUR TOPIC } or A deep dive into { YOUR TOPIC }
### Presentation: [Presentation_name](workshop/Presentation_template.pptx)

## Workshop description
Describe why your topic is important and what you want to share with your audience

## Requirements
PyLadies encourage the usage of [uv](https://astral.sh/blog/uv) for dependency management. You can get uv by running the following command in your shell:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
To get started, open the `pyproject.toml` file and set the required python version. The pre-selected version 3.8 is generally a safe choice for most use cases.

After you have specified the python version, you can create a virtual environemnt with `uv venv` and add packages with `uv add <package>`. Before the workshop, you can generate a requirements.txt file by running `uv export > requirements.txt`.

If you are not familiar with uv or are in a hurry, feel free to work with your familiar dependency management system and the PyLadies will handle the `pyproject.toml` for you.

## Usage
* Clone the repository
* Start { TOOL } and navigate to the workshop folder

## Video record
Re-watch [this YouTube stream](link)

## Credits
This workshop was set up by @pyladiesams and {your github handler}


## Appendix

### Google Colab

To run this project on Google Colab, follow the following instructions:
1. Visit [Google Colab](https://colab.research.google.com/)
2. In the top left corner select "File" &#8594; "Open Notebook"
3. Under "GitHub", enter the URL of the repo of this workshop
4. Select one of the Notebooks within the repo.
5. At the tob of the notebook, runthe following code:
```bash
!cd ./name-of-repo
!pip install -r requirements.txt
```

Happy Coding :)

