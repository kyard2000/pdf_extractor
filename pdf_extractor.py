import fitz  # PyMuPDF for text and image extraction
import pdfplumber  # For table extraction
import os 
from openai import AzureOpenAI
import os
import re
from openai import AzureOpenAI
from base64 import b64encode
import numpy as np
import time


#place openai endpoint info here:
azure_openai_key=""
azure_openai_endpoint=""

azure_service_region='eastus'
apiversion = "2024-02-01"
gptengine = "gpt-4o-mini"

openai_client = AzureOpenAI(
            api_key=azure_openai_key,
            api_version=apiversion,
            azure_endpoint=azure_openai_endpoint
        )

#ensures all needed folders exist
if not os.path.exists("extracted_images"):
    os.makedirs("extracted_images")

if not os.path.exists("extracted_text"):
    os.makedirs("extracted_text")

if not os.path.exists("pdfs"):
    os.makedirs("pdfs")

output_folder = r'pdf_to_image'
extracted_images=r"extracted_images"
pdf_path = r'pdfs'

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return b64encode(image_file.read()).decode('utf-8')


def text_prompt(prompt):
    try:
            response = openai_client.chat.completions.create(
                model=gptengine,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt}
                        ]
                    }
                ],
                max_tokens=300
            )
            return response.choices[0].message.content
    except Exception as e:
        print(f"An error occurred while processing")
        return None

def analyze_image(image_path, prompt):
    base64_image = encode_image(image_path)
    try:
        response = openai_client.chat.completions.create(
            model=gptengine,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"An error occurred while processing {image_path}: {str(e)}")
        return None



# List to hold all the extracted text
all_text = []

# Iterate through each page of the PDF
for pdf in os.listdir(pdf_path):
    doc = fitz.open(os.path.join(pdf_path, pdf))
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)  # Get a page
        
                                                                                                                                                                                        
        text = page.get_text("text")
        all_text.append(f"Page {page_num + 1}:\n{text}\n")
        print(page_num)

        #this is where we try to extract an table info
        try:
            with pdfplumber.open(pdf) as pdf_plumb:
            # Iterate through each page
                all_text.append(f"Tables on {page_num}:")
                tables = pdf_plumb.pages[page_num].extract_tables()
                for table in tables:
                    for row in table:
                        all_text.append(f"{row}")

        #this allows for pdfplumber to error but we still get some sort of information out of the chart, it usually doesnt though so Im not sure how necessary this is
        except:
            chartdescription = analyze_image((f'{output_folder}\\page_{page_num + 101}.jpg'), f"this is an image of a pdf page, and here is the text extracted from the page: {text}, output the columns and rows labeled as so, but do not put the data, if the page contains no tables charts or graphs, Just say: --- ")
            print(f'chart description:{chartdescription}')
            prompt = f'this is the text extracted from a pdf: {text}, and this is the table information extracted from it {chartdescription}. with this information, describe the table in detail, if the page contains no discernable tables, Just say: --- '
            table_description = text_prompt(prompt)
            print(table_description)
            all_text.append(f"Page {page_num}, Tables: {table_description}")


        # Extract images
        image_list = page.get_images(full=True)  # Get all images on the page
        if image_list:
            for img_index, img in enumerate(image_list, start=1):
                # Extract the image bytes
                xref = img[0]  # Get image reference
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                # Save the image
                image_filename = f"extracted_images/page{page_num + 1}_img{img_index}.{image_ext}"
                imagedescription = analyze_image(image_filename, f"Describe this image in detail, keep things fairly spartan but accurate, this will be fed into a vectordb so avoid any formatting...")
                #print(image)
                all_text.append(f"Page {page_num}, Image {img_index}:\n{imagedescription}\n")
                with open(image_filename, "wb") as img_file:
                    img_file.write(image_bytes)

# Close the fitz document
doc.close()

# Save all extracted text to a file
with open(os.path.join('extracted_text', f'{pdf}_extracted.txt'), 'w', encoding='utf-8') as f:
    for page_text in all_text:
        f.write(page_text)

print(f"Text extraction complete. Images saved in 'extracted_images' folder. ")
