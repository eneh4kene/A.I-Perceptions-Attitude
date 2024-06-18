# Reddit Data Fetching and Processing Project

## Overview

This project involves fetching, processing, and analyzing data from Reddit. It includes scripts for fetching questions, wrangling data, and preparing final submissions, likely for a competition or data analysis task.

## Features

- **Fetch Questions**: Retrieve questions or posts from Reddit using specified parameters.
- **Fetch Reddit Data**: Collect data from various subreddits or threads.
- **Data Wrangling**: Clean and preprocess Reddit data for analysis.
- **Final Submission Preparation**: Prepare the final dataset or results for submission.

## Requirements

- Python 3.x
- Libraries such as `requests`, `pandas`, `numpy`, and others as needed for data fetching and processing.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/reddit-data-project.git
   cd reddit-data-project
   ```

2. **Create a virtual environment and activate it**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Fetching Questions

Use the `fetch_questions2_2023.py` script to fetch questions from Reddit.

```bash
python fetch_questions2_2023.py
```

### Fetching Reddit Data

Use the `fetch_reddit.py` script to fetch data from various subreddits.

```bash
python fetch_reddit.py
```

### Data Wrangling

Use the `reddit_wrangle.py` script to clean and preprocess the Reddit data.

```bash
python reddit_wrangle.py
```

### Preparing Final Submission

Use the `final_submission_file.py` script to prepare the final dataset or results for submission.

```bash
python final_submission_file.py
```

## File Structure

```
reddit-data-project/
├── data/                             # Directory to store raw and processed data
├── fetch_questions2_2023.py          # Script to fetch questions from Reddit
├── fetch_reddit.py                   # Script to fetch Reddit data
├── final_submission_file.py          # Script to prepare final submission
├── reddit_wrangle.py                 # Script for data wrangling
├── requirements.txt                  # Required Python packages
└── README.md                         # Project README file
```

## Contributing

Contributions are welcome! Please fork this repository and submit a pull request for any changes you would like to make.
