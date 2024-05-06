import os

import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import euclidean_distances
import re
import openai

# Access the OPENAI_API_KEY
openai.api_key = ""
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')


def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()


def call_open_api(prompt, description):
    response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt=f"{prompt}:{description}",
        max_tokens=256,
        temperature=0.7
    )

    return response.choices[0].text.strip()


def clean_text(text):
    """Preprocesses the text by removing new lines and multiple spaces."""
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def embed_text(text, pooling='mean'):
    """Generates embeddings for the given text using BERT with different pooling strategies."""
    text = clean_text(text)
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    if pooling == 'mean':
        return outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
    elif pooling == 'max':
        return outputs.last_hidden_state.max(dim=1).values.squeeze().detach().numpy()
    elif pooling == 'cls':
        return outputs.last_hidden_state[:, 0, :].squeeze().detach().numpy()
    else:
        raise ValueError("Unsupported pooling strategy")


def expand_skills(description):
    prompt = f""" From now on, you are Artemis, a recruitment specialist working for top consulting firms (Deloitte, 
    EnY, KPMG, PWC). Generate a structured list of skills and related technologies based on the provided input.
    Please analyze the following skills: {description} 
    Analyze these above mentioned skills and define categories based on their relationships and relevance to each other. 
    The output should be formatted as a clear list, where each line contains a category, skill, and its corresponding 
    inference type. Do not add an unnecessary spaces. Strictly follow the format given here: 
    - <Category>: <Skill> - Explicit/Inferred
    Here's some example:
    - Programming Languages: Python - Explicit
    - Databases: SQL - Inferred
    - Programming Languages: Java - Inferred """
    response = call_open_api(prompt, description)
    print("this is in expand\n", response)
    return response


def infer_related_knowledge(text):
    prompt = f""" From now on, you are Artemis, a recruitment specialist working for top consulting firms (Deloitte, 
    EnY, KPMG, PWC). Generate a structured list of skills and related technologies based on the provided input.
    Please analyze the following skills: {text} 
    Analyze these above mentioned skills and define categories based on their relationships and relevance to each other. 
    The output should be formatted as a clear list, where each line contains a category, skill, and its corresponding 
    inference type. Do not add an unnecessary spaces. Strictly follow the format given here: 
    - <Category>: <Skill> - Explicit/Inferred
    Here's some example:
    - Programming Languages: Python - Explicit
    - Databases: SQL - Inferred
    - Programming Languages: Java - Inferred """
    response = call_open_api(prompt, text)
    print("this is in infer\n", response)
    return response


def parse_sections(text):
    """Parses text into sections based on specific markers."""
    sections = {'Skills': '', 'Experience': '', 'Education': ''}
    current_section = None
    for line in text.split('|'):
        line = line.strip()
        if 'Skills:' in line:
            current_section = 'Skills'
            sections[current_section] += line.replace(f'{current_section}:', '').strip() + ' '
        elif 'Experience:' in line:
            current_section = 'Experience'
            sections[current_section] += line.replace(f'{current_section}:', '').strip() + ' '
        elif 'Education:' in line:
            current_section = 'Education'
            sections[current_section] += line.replace(f'{current_section}:', '').strip() + ' '

    return sections


def parse_resume_name(text):
    line = text.split('|')
    for line in text.split('|'):
        line = line.strip()
        if 'Name:' in line:
            name_section = line.replace('Name:', '')
            break
        else:
            raise Exception("No name found")
    return name_section


def compare_sections(jd_sections, resumes_sections, pooling='mean'):
    """Compares sections of JD and multiple resumes using Euclidean distance, with binary scoring for must-have
    skills. """
    results = {}
    # Step 4.2 - Give call to open api for getting relevant skills mentioned in job description
    jd_skills_expanded = expand_skills(jd_sections['Skills'])
    for resume_name, sections in resumes_sections.items():
        # Step 4.1 - Give call to open api for getting relevant skills mentioned in resume
        resume_skills_inferred = infer_related_knowledge(sections['Skills'])

        # Step 4.2 - Now we check for common skills between job description and resume
        #skill_match = jd_skills_expanded.lower() in resume_skills_inferred.lower()
        skill_match = True
        # Step 4.3 - Now we find similarity based on embeddings. We use euclidean_distances here
        if skill_match:
            jd_embedding = embed_text(jd_skills_expanded, pooling)
            resume_embedding = embed_text(resume_skills_inferred, pooling)
            skill_distance = euclidean_distances([jd_embedding], [resume_embedding])[0][0]
            total_distance = skill_distance  # Start with skill distance
            print("Total distance : ", total_distance)
            for section in ['Experience', 'Education']:
                if jd_sections[section] and sections[section]:
                    section_embedding = embed_text(jd_sections[section], pooling)
                    resume_embedding = embed_text(sections[section], pooling)
                    distance = euclidean_distances([section_embedding], [resume_embedding])[0][0]
                    total_distance += distance
            results[resume_name] = total_distance
        else:
            results[resume_name] = float('inf')  # Penalize heavily if must-have skills are missing
    return results


def get_input_from_user():
    # Step 1 - Input job description
    job_description_input = input("Enter the full job description (including sections): ")
    # Step 2 - Derive sections like 'Skills', 'Experience' and 'Education' from the job description
    jd_sections_from_user = parse_sections(job_description_input)

    # Step 3 - Provide number of resumes that you need to give for given job description
    num_resumes = int(input("How many resumes do you want to compare? "))
    resumes_sections_from_user = {}
    for i in range(num_resumes):
        name = input(f"Enter name for Resume {i + 1}:")
        resume_input = input("Enter the full resume (including sections): ")
        # Step 3.1 - Derive sections like 'Skills', 'Experience' and 'Education' from the resume
        resumes_sections_from_user[name] = parse_sections(resume_input)
    print("Here are the jd sections from user input \n", jd_sections_from_user)
    print("Here are the resume sections from user input \n", resumes_sections_from_user)
    return jd_sections_from_user, resumes_sections_from_user


def get_input_from_files(job_description_path, resumes_directory):
    # Step 1 - Read job description from file
    job_description_input = read_file(job_description_path)
    jd_sections_from_files = parse_sections(job_description_input)

    # Get a list of all files in the resumes directory
    resume_files = os.listdir(resumes_directory)
    resume_paths = [os.path.join(resumes_directory, file) for file in resume_files]

    # Determine the number of resumes based on the number of files in resume_paths
    num_resumes = len(resume_paths)
    print("Number of resumes : ", num_resumes)

    resumes_sections_from_files = {}
    # Step 2 - Read resumes from files
    for i, resume_path in enumerate(resume_paths):
        resume_input = read_file(resume_path)
        resume_name = parse_resume_name(resume_input)
        resumes_sections_from_files[resume_name] = parse_sections(resume_input)
    #print("Here are the jd sections from files \n", jd_sections_from_files)
    #print("Here are the resume sections from files \n", resumes_sections_from_files)
    return jd_sections_from_files, resumes_sections_from_files


# Main Function
if __name__ == "__main__":

    choice = input("How would you like to provide input? (1 - User Input, 2 - File Input): ")

    if choice == '1':
        jd_sections, resumes_sections = get_input_from_user()
    elif choice == '2':
        job_description_path = "../dataset/cv-usecase/job-description-sample.txt"
        resumes_directory = "../dataset/cv-usecase/resumes"
        jd_sections, resumes_sections = get_input_from_files(job_description_path, resumes_directory)
    else:
        print("Invalid choice! Please choose either '1' or '2'.")

    if jd_sections and resumes_sections:
        # Step 4 - Compare the sections of job description with resumes
        pooling_method = input("Enter pooling method (mean, max, cls): ")
        similarities = compare_sections(jd_sections, resumes_sections, pooling=pooling_method)

        # Step 5 - Give the matching result
        print("\nMatching Scores:")
        for resume, score in similarities.items():
            print(f"{resume}: Score = {score:.4f}")
