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


def call_open_api(prompt,token_value):
    response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt=f"{prompt}",
        max_tokens=token_value,
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


def expand_skills(resume_name,description):
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
    response = call_open_api(prompt,256)
    print("this is in expand for candidate : ",resume_name, response)
    return response


def infer_skills_and_experience_knowledge(resume_name,skills, experience):
    prompt = f""" From now on, you are Artemis, a recruitment specialist working for top consulting firms (Deloitte, 
    EnY, KPMG, PWC). 
    Here's the skills of the candidate I have from the resume : {skills}
    Here's the experience I have from the same resume : {experience}
    
    1.With text coming in "Experience from resume" create a context to identify which industry this experience can
    belongs to. 
    2.With text coming in "Skills" create a context to identify which industry these skills can be best used in.
    3.Do this to identify boundary condition when comparing skills with experience. Example "A skill 
    which seems completely not related to experience needs to removed from consideration"
    4. Analyse the given skills and experience based on context created in point 1 and point 2 to find relevance. 
    4.1 Example of absolute false comparison -- "a. A sales representative generally will not have IT related skills like python.
    b. An IT Professional will not have skill related to mechanical engineering"
    4.2 Example of absolute best comparison -- "a. A digital marketing experience will have skills like Digital Marketing, SEO, Google Analytics, Content Creation, Strategic Planning" 
    
    Now, identify based on this analyses that you've carried, give me a relatable score between 0 to 1 where 
    if you find skills to experience not relevant then give score between 0.0000 to 0.2000 
    if you find skills to experience strongly relevant then give score between 0.7000 to 1.0000 
    
    
    The output should always be formatted as a single decimal number with maximum 4 decimal places and no new lines or spaces.

    Here's a example of the output format I am expecting: 0.4543 """
    print("\n")
    print("This is the prompt for candidate : ", resume_name)
    print(" ", prompt)
    response = call_open_api(prompt,256)
    print("this is in infer_skills_and_experience_knowledge for candidate : ", resume_name, response)
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
    """Compares sections of JD and multiple resumes using Euclidean distance, then normalizes the scores."""
    results = {}
    min_distance = float('inf')
    max_distance = float('-inf')

    # First, calculate distances for all resumes and find min/max distances
    jd_skills_expanded = expand_skills("JobDescription",jd_sections['Skills'])
    for resume_name, sections in resumes_sections.items():
        resume_skills_expanded = expand_skills(resume_name,sections['Skills'])
        jd_embedding = embed_text(jd_skills_expanded, pooling)
        resume_embedding = embed_text(resume_skills_expanded, pooling)
        skill_distance = euclidean_distances([jd_embedding], [resume_embedding])[0][0]
        total_distance = skill_distance
        results[resume_name] = total_distance
        # Update min and max distances found
        if total_distance < min_distance:
            min_distance = total_distance
        if total_distance > max_distance:
            max_distance = total_distance
    # Normalize distances based on the min/max found
    normalized_scores = {}
    for resume_name, distance in results.items():
        if distance == float('inf'):
            normalized_scores[resume_name] = 0.0  # Assign 0 if skills didn't match at all
        else:
            # Normalize such that closer distances score higher (closer to 1)
            normalized_scores[resume_name] = 1 - (distance - min_distance) / (
                        max_distance - min_distance) if max_distance != min_distance else 1.0
    skill_experience_relatability_score = {}
    final_normalised_score = {}
    for resume_name, sections in resumes_sections.items():
        skill_experience_relatability_score[resume_name] = float(infer_skills_and_experience_knowledge(resume_name, sections['Skills'],
                                                                                                 sections['Experience']))
        #Below formula defines the skills vs experience weightage to be 20% and other score which is job description vs skill to be 80%
        skill_experience_relatability_score[resume_name] = (skill_experience_relatability_score[resume_name]*0.2) + (normalized_scores[resume_name]*0.8)

        #final_normalised_score[resume_name] = skill_experience_relatability_score[resume_name] / 2
    return skill_experience_relatability_score


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

    # Get a list of all files in the resumes directory ensuring that only text files are read from the given
    # directory and not from subdirectories
    resume_files = [file for file in os.listdir(resumes_directory) if os.path.isfile(os.path.join(resumes_directory, file)) and file.endswith('.txt')]
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
    # print("Here are the jd sections from files \n", jd_sections_from_files)
    # print("Here are the resume sections from files \n", resumes_sections_from_files)
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
