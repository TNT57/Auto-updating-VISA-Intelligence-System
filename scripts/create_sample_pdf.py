"""Create a sample PDF with real 485 visa information for testing the system."""
import sys
import os
sys.stdout.reconfigure(encoding='utf-8')

from fpdf import FPDF

pdf_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'pdfs')
os.makedirs(pdf_dir, exist_ok=True)
pdf_path = os.path.join(pdf_dir, '485_visa_info.pdf')

pdf = FPDF()
pdf.add_page()
pdf.set_auto_page_break(auto=True, margin=15)

# Content based on official Home Affairs 485 visa information
sections = [
    ("Temporary Graduate Visa - Subclass 485", [
        "Source: Australian Department of Home Affairs",
        "https://immi.homeaffairs.gov.au/visas/getting-a-visa/visa-listing/temporary-graduate-485",
    ]),
    ("Overview", [
        "The Temporary Graduate visa (subclass 485) allows recent graduates to live, study and work in Australia temporarily after completing their studies. It provides an opportunity to gain work experience in Australia that can count towards a skilled migration application.",
        "There are different streams within this visa, each with specific eligibility requirements. The visa is a temporary visa and does not lead directly to permanent residency, though it can provide a pathway through skilled migration programs.",
    ]),
    ("Visa Streams", [
        "Post-Study Work stream: For students who have graduated with a higher education degree from an Australian institution. The length of the visa depends on the level of qualification completed.",
        "Graduate Work stream: For international students who have recently graduated with skills and qualifications relevant to an occupation listed on the Skilled Occupation List. This stream requires a skills assessment.",
        "Second Post-Study Work stream: A second visa for certain holders of a first Temporary Graduate visa in the Post-Study Work stream who wish to continue staying in Australia.",
        "Replacement stream: For former Temporary Graduate visa holders who were unable to travel to Australia due to COVID-19 travel restrictions.",
    ]),
    ("Post-Study Work Stream Eligibility", [
        "You must be in Australia when you apply for this visa and when the visa is granted.",
        "You must be under 50 years of age at the time of application.",
        "You must hold or have held an eligible student visa within the past 6 months.",
        "You must have completed at least one degree, diploma or trade qualification from an Australian institution that took at least 2 academic years to complete (92 weeks).",
        "Your qualification must be either a single qualification (Bachelor degree, Masters by coursework, Masters by research, or Doctoral degree) or a combination of qualifications that total at least 2 academic years of study.",
        "You must provide evidence of your English language proficiency. This can include IELTS (overall band score of at least 6.0 with no band less than 5.5), TOEFL iBT, PTE Academic, or Cambridge English Advanced.",
        "You must meet health and character requirements.",
        "You must have Overseas Student Health Cover (OSHC) from the time your student visa expires until the end of your 485 visa.",
    ]),
    ("Post-Study Work Stream Visa Duration", [
        "Bachelor degree (including Honours): 2 years from date of grant.",
        "Masters by coursework: 2 years from date of grant.",
        "Masters by research: 3 years from date of grant.",
        "Doctoral degree (PhD): 4 years from date of grant.",
        "Note: Regional study may qualify for additional visa duration. Studying and living in a regional area can provide an additional 1-2 years on your visa.",
    ]),
    ("Graduate Work Stream Eligibility", [
        "Your nominated occupation must be on the relevant Skilled Occupation List.",
        "You must have a positive skills assessment in your nominated occupation.",
        "You must have completed at least 2 academic years of study in Australia in a course closely related to your nominated occupation.",
        "You must be under 50 years of age.",
        "You must meet English language requirements (competent English minimum).",
        "The Graduate Work stream visa is granted for up to 18 months.",
    ]),
    ("English Language Requirements", [
        "For Post-Study Work stream: Functional English is required. Acceptable tests include IELTS with an overall band score of at least 6.0, with no individual band below 5.5. PTE Academic with an overall score of at least 50, with no communicative skill below 36. TOEFL iBT with a total score of at least 64, with minimum scores in each section.",
        "For Graduate Work stream: Competent English is required. This means IELTS overall 6.0 with no band below 6.0, or equivalent scores in PTE Academic, TOEFL iBT, or Cambridge English.",
        "English test results must be less than 3 years old at the time of application.",
        "Some passport holders from UK, USA, Canada, NZ, and Ireland may be exempt from English testing.",
    ]),
    ("Health Insurance Requirements", [
        "You must maintain Overseas Student Health Cover (OSHC) or Overseas Visitors Health Cover (OVHC) for the entire duration of your stay in Australia on this visa.",
        "If your student visa OSHC has expired, you must obtain appropriate health insurance cover from the date your student visa expires until the end of your 485 visa period.",
        "Failure to maintain adequate health insurance may result in visa cancellation.",
    ]),
    ("Application Process", [
        "Step 1: Gather required documents including passport, degree certificate or completion letter, transcripts, English test results, health insurance evidence, skills assessment (for Graduate Work stream).",
        "Step 2: Complete the online application through ImmiAccount on the Home Affairs website.",
        "Step 3: Pay the visa application charge. The base fee is approximately AUD $1,895 for the main applicant (fees are subject to change).",
        "Step 4: Complete health examinations if required. You will be advised if health examinations are needed after lodgement.",
        "Step 5: Provide biometrics if requested.",
        "Step 6: Wait for a decision. Processing times vary. The visa will be granted if all requirements are met.",
    ]),
    ("Visa Conditions and Obligations", [
        "Condition 8501: Maintain health insurance for the duration of your stay.",
        "Condition 8516: You must continue to meet the eligibility requirements for the visa.",
        "You must notify the Department of any changes to your contact details, passport, or address within 14 days.",
        "You must comply with all Australian laws during your stay.",
        "You cannot include family members who were not declared in your original student visa application (with limited exceptions).",
    ]),
    ("Work Rights", [
        "The Temporary Graduate visa allows you to work full-time in Australia. There are no restrictions on the type of work or the number of hours you can work.",
        "This is a key advantage over the student visa, which restricts working hours during study periods.",
        "Many graduates use this period to gain work experience in their field of study, which can support a subsequent skilled migration application.",
    ]),
    ("Pathways to Permanent Residency", [
        "While the 485 visa itself does not lead directly to permanent residency, it provides valuable time to qualify for skilled migration pathways including:",
        "Skilled Independent visa (subclass 189): For skilled workers who are not sponsored by an employer or family member. Requires meeting the points test threshold.",
        "Skilled Nominated visa (subclass 190): For skilled workers nominated by a state or territory government. Provides additional points for nomination.",
        "Skilled Work Regional (Provisional) visa (subclass 491): For skilled workers willing to live and work in regional Australia. Can lead to permanent residency after 3 years.",
        "Employer Sponsored visas: If you find an employer willing to sponsor you, you may be eligible for employer-sponsored visa options.",
        "The Australian work experience gained during your 485 visa period can contribute valuable points towards your Expression of Interest (EOI) for skilled migration.",
    ]),
    ("Regional Study Benefits", [
        "If you studied and lived in a regional area of Australia, you may be eligible for an extended 485 visa duration.",
        "Regional areas include: Category 2 (Cities and major regional centres) includes Perth, Adelaide, Gold Coast, Sunshine Coast, Canberra, Newcastle/Lake Macquarie, Wollongong/Illawarra, Geelong, and Hobart. Category 3 (Regional centres and other regional areas) includes all other areas.",
        "Additional 1 year of visa duration for studying in a Category 2 area. Additional 2 years for studying in a Category 3 area.",
    ]),
    ("Important Notes", [
        "Processing times vary and can change. Check the Home Affairs website for current processing times.",
        "Visa application charges are subject to change. Always verify current fees on the official website.",
        "Legislation and policy can change. Always refer to the official Department of Home Affairs website for the most current information.",
        "This document is for informational purposes only and does not constitute immigration advice. Consult a registered migration agent for personalised advice.",
    ]),
]

for heading, paragraphs in sections:
    # Heading
    if heading == "Temporary Graduate Visa - Subclass 485":
        pdf.set_font("Helvetica", "B", 16)
    else:
        pdf.set_font("Helvetica", "B", 13)
    pdf.multi_cell(0, 8, heading)
    pdf.ln(2)
    
    # Body text
    pdf.set_font("Helvetica", size=10)
    for para in paragraphs:
        pdf.multi_cell(0, 6, para)
        pdf.ln(2)
    pdf.ln(3)

pdf.output(pdf_path)
print(f"PDF created at: {pdf_path}")
print(f"PDF size: {os.path.getsize(pdf_path)} bytes")
print(f"\nDone! Now run: python scripts/initial_setup.py")