import pdfplumber
import re

def extract_skills_experience_from_pdf(pdf_path):
    skills = []
    experience = []

    with pdfplumber.open(pdf_path) as pdf:
        # Iterate through each page of the PDF
        for page in pdf.pages:
            text = page.extract_text()

            # Extract skills section
            skills_section_match = re.search(r"SKILLS\s*(.+?)(EXPERIENCE|EDUCATIONAL|PERSONAL)", text, re.DOTALL)
            if skills_section_match:
                skills_text = skills_section_match.group(1)
                # Extract individual skills
                skills.extend(re.findall(r"(?:[-●]\s*(.*?)(?=[\n\u2022\u2023-]|$))", skills_text))

            # Extract experience section
            experience_section_match = re.search(r"EXPERIENCE\s*(.+?)(EDUCATIONAL|PERSONAL)", text, re.DOTALL)
            if experience_section_match:
                experience_text = experience_section_match.group(1)
                # Split experience by dates and job titles
                experience.extend(re.findall(r"(\d{2}/\d{4})\s+to\s+([\w\s,:]+)", experience_text))

    return skills, experience

def format_skills(skills):
    formatted_skills = ""
    for skill in skills:
        formatted_skills += f"- {skill.strip()}\n"
    return formatted_skills

def format_experience(experience):
    formatted_experience = ""
    for exp in experience:
        formatted_experience += f"- {exp[0]} to {exp[1].strip()}\n"
    return formatted_experience

# Path to your PDF file
pdf_path = '../dataset/cv-usecase/resumes/Total_CVs/Pooja_Gomekar.pdf'

# Extract skills and experience
skills, experience = extract_skills_experience_from_pdf(pdf_path)

# Format the extracted information
formatted_skills = format_skills(skills)
formatted_experience = format_experience(experience)

# Print the formatted information
print("Skills:")
print(formatted_skills)

print("Experience:")
print(formatted_experience)
