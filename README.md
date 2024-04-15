# Summarization and Sentiment Analysis of Federal Reserve FOMC Minutes and Statements

<!--- These are examples. See https://shields.io for others or to customize this set of shields. You might want to include dependencies, project status and licence info here --->
![Static Badge](https://img.shields.io/badge/NU-Spring_2024-red)
![Static Badge](https://img.shields.io/badge/python-3.11.5-blue)

In the rapidly evolving landscape of financial technology, the ability to distill complex information into actionable insights stands paramount. This project, centered on the summarization and sentiment analysis of Federal Reserve FOMC Minutes & Statements, aims to leverage cutting-edge Natural Language Processing (NLP) techniques to unlock nuanced understandings of central bank communications. Undertaken within the context of my role at a FinTech company expanding into regulatory and central bank coverage, this endeavor seeks to bridge the gap between sophisticated NLP methodologies and practical financial analysis, providing a foundation for informed decision-making in a domain critical to our company’s growth and innovation.

## Objective
This project aims to leverage NLP techniques for summarizing and performing sentiment analysis on the Federal Reserve's FOMC meeting minutes and statements. The goal is to extract key insights and sentiment trends from these documents, offering a clearer understanding of the Federal Reserve's perspective on economic conditions and policy decisions over time.

## Data
Utilizing the "Federal Reserve FOMC Minutes & Statements Dataset" from Kaggle, which comprises text data of FOMC meeting minutes and statements, providing a comprehensive overview of economic policy discussions.

https://www.kaggle.com/datasets/drlexus/fed-statements-and-minutes/data

## Prerequisites

Before you begin, ensure you have met the following requirements:
* You have installed Anaconda or Miniconda.
* You have access to a terminal on a Windows, Linux, or Mac machine. 
* You have read the documentation provided in the repo.

## Installing Summarization and Sentiment Analysis Tool

To install the Summarization and Sentiment Analysis Tool and set up the necessary environment, follow these steps:

1. **Clone the Repository**

   ```bash
   git clone https://github.com/kierblk/CS-6120-Final-Project.git
   cd CS-6120-Final-Project
   ```

2. **Create the Conda Environment**

   To create a Conda environment with all required dependencies, run:

   ```bash
   conda env create -f environment.yml
   ```

   This will install all the necessary packages as specified in the `environment.yml` file.

3. **Activate the Environment**

   Before running the project, activate your newly created Conda environment:

   ```bash
   conda activate myenv
   ```

   Replace `myenv` with the name of the environment specified in your `environment.yml` file.

## Usage

To run this project, follow these steps:

```bash
python main.py
```

Adjust the `main.py` file to point to your specific data sources or to change parameters such as model names.

## Contributing
To contribute to Sentiment Analysis and Summarization Tool, follow these steps:

1. Fork this repository.
2. Create a branch: `git checkout -b <branch_name>`.
3. Make your changes and commit them: `git commit -m '<commit_message>'`
4. Push to the original branch: `git push origin <branch_name>`
5. Create the pull request.

Alternatively, see the GitHub documentation on [creating a pull request](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request).

## Contributors

* [@kierblk](https://github.com/kierblk)
