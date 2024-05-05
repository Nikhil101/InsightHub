import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import euclidean_distances
import re
import openai

# Access the OPENAI_API_KEY
openai.api_key = ""
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')


def call_open_api(prompt, description):
    response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt=f"{prompt}:{description}",
        max_tokens=150
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
    prompt = f"""
    From now on you are Artemis, a recuirtment specilist work for big 4 (Delloite, EnY, KPMG, PWC) consulting firms hiring for varied positions.
    Generate a structured and categorized list of all skills and related technologies an individual is likely familiar with, based on the input details provided. Analyze these skills and autonomously define categories based on their relationships and relevance to each other. The output should be formatted in a clear list, with each category and its associated skills succinctly listed.
    Please analyze the following skills:
    {description}
    There are no predefined categories. Allow the categories to emerge based on the context and relationships among these skills. Ensure each category is clearly listed with relevant skills grouped together. Dont provide description of each, just provide the technical stack and category. Also dont provide any contextual information like "here is the list and category". Just put the inferences in the category "example -- if database is there then put SQL in "language" category and vice-versa but dont remove the inferred category, like database needs to be seperate category. Label skills and categories with 'explicit' or 'inferred' word only. Output format should be list containing (Category, Skill), example (Programming Languages, Python) - Explicit. 
    """
    response = call_open_api(prompt, description)
    print("this is in expand", response)
    return response


def infer_related_knowledge(text):
    prompt = f"""
    From now on you are Artemis, a recuirtment specilist work for big 4 (Delloite, EnY, KPMG, PWC) consulting firms hiring for varied positions.
    Generate a structured and categorized list of all skills and related technologies an individual is likely familiar with, based on the input details provided. Analyze these skills and autonomously define categories based on their relationships and relevance to each other. The output should be formatted in a clear list, with each category and its associated skills succinctly listed.
    Please analyze the following skills:
    {text}
    There are no predefined categories. Allow the categories to emerge based on the context and relationships among these skills. Ensure each category is clearly listed with relevant skills grouped together. Dont provide description of each, just provide the technical stack and category. Also dont provide any contextual information like "here is the list and category". Just put the inferences in the category "example -- if database is there then put SQL in "language" category and vice-versa but dont remove the inferred category, like database needs to be seperate category. Label skills and categories with 'explicit' or 'inferred' word only. Output format should be list containing (Category, Skill), example (Programming Languages, Python) - Explicit. 
    """
    response = call_open_api(prompt, text)
    print("this is in infer", response)
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


def compare_sections(jd_sections, resumes_sections, pooling='mean'):
    """Compares sections of JD and multiple resumes using Euclidean distance, with binary scoring for must-have skills."""
    results = {}
    # Step 4.2 - Give call to open api for getting relevant skills mentioned in job description
    jd_skills_expanded = expand_skills(jd_sections['Skills'])
    for resume_name, sections in resumes_sections.items():
        # Step 4.1 - Give call to open api for getting relevant skills mentioned in resume
        resume_skills_inferred = infer_related_knowledge(sections['Skills'])

        #Step 4.2 - Now we check for common skills between job description and resume
        skill_match = jd_skills_expanded.lower() in resume_skills_inferred.lower()

        #Step 4.3 - Now we find similarity based on embeddings. We use euclidean_distances here
        if skill_match:
            jd_embedding = embed_text(jd_skills_expanded, pooling)
            resume_embedding = embed_text(resume_skills_inferred, pooling)
            skill_distance = euclidean_distances([jd_embedding], [resume_embedding])[0][0]
            total_distance = skill_distance  # Start with skill distance
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


# Main Function
if __name__ == "__main__":
    # Step 1 - Input job description
    job_description_input = input("Enter the full job description (including sections): ")
    # Step 2 - Derive sections like 'Skills', 'Experience' and 'Education' from the job description
    jd_sections = parse_sections(job_description_input)

    # Step 3 - Provide number of resumes that you need to give for given job description
    num_resumes = int(input("How many resumes do you want to compare? "))
    resumes_sections = {}
    for i in range(num_resumes):
        name = input(f"Enter name for Resume {i + 1}:")
        resume_input = input("Enter the full resume (including sections): ")
        # Step 3.1 - Derive sections like 'Skills', 'Experience' and 'Education' from the resume
        resumes_sections[name] = parse_sections(resume_input)

    # Step 4 - Compare the sections of job description with resumes
    pooling_method = input("Enter pooling method (mean, max, cls): ")
    similarities = compare_sections(jd_sections, resumes_sections, pooling=pooling_method)

    # Step 5 - Give the matching result
    print("\nMatching Scores:")
    for resume, score in similarities.items():
        print(f"{resume}: Score = {score:.4f}")
