
# An introduction to { YOUR TOPIC } or A deep dive into { YOUR TOPIC }
### Presentation: [Presentation_name](workshop/Presentation_template.pptx)

## Workshop description
Describe why your topic is important and what you want to share with your audience

## Requirements
PyLadies use [uv](https://astral.sh/blog/uv) for dependency management. For installation instructions, please check the [official documentation](https://astral.sh/blog/uv) or the [PyPI page](https://pypi.org/project/uv/).


## Usage
To get started, open the `pyproject.toml` file and set the required python version. The pre-selected version 3.8 is generally a safe choice for most use cases.

After you have specified the python version, you can create a virtual environemnt with `uv venv` and add packages with `uv add <package>`. Before the workshop, you can generate a requirements.txt file, which is needed e.g. for running code in GoogleColab, by running `uv export > requirements.txt`.

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


### Pre-Commit Hooks

To ensure our code looks beautiful, PyLadies uses pre-commit hooks. You can enable them by running `pre-commit install`. You may have to install `pre-commit` first, using `uv sync`, `uv pip install pre-commit` or `pip install pre-commit`.