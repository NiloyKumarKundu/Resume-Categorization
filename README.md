# Resume Categorization Script

## Overview

This script categorizes PDF resumes based on a pre-trained machine learning model. The categorized resumes are organized into folders, and a CSV file containing the filenames and their corresponding categories is generated.

## Setup Instructions

1. **Download Files**

   - Download the `script.py`, `requirements.txt`, and `models` folder.
   - Place all these files and the `models` folder into the same directory.

2. **Install Required Libraries**

    Open a terminal or command prompt and navigate to the directory where you placed the files. Install the necessary libraries using the following command:

    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Script**
    Execute the script with the following command:
    ```bash
    python script.py path/to/dir
    ```
    Replace `path/to/dir` with the path to the folder containing your resume PDF files.

4. **Expected Output**
    * The script will organize the PDF files into subdirectories based on their predicted categories.
    * A CSV file named `categorized_resumes.csv` will be generated in the `path/to/dir` folder. This file will contain two columns: `filename` and `category`.

    The `categorized_resumes.csv` file will have the following format:
    ```csv
    filename,category
    resume1.pdf,CategoryA
    resume2.pdf,CategoryB
    resume3.pdf,CategoryA
    ```
    In this example, `resume1.pdf` and `resume3.pdf` were categorized as `CategoryA`, while `resume2.pdf` was categorized as `CategoryB`.

By following these instructions, you will be able to categorize resumes and generate a CSV file with the results.
