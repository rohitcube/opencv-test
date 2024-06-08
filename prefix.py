import os
import PyPDF2

# Set the folder path where the PDF files are located
folder_path = '/Users/rohit/Desktop/chile'

# Create a new PDF file merger object
merger = PyPDF2.PdfMerger()

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    # Check if the file is a PDF
    if filename.endswith('.pdf'):
        # Open the PDF file in read-binary mode
        with open(os.path.join(folder_path, filename), 'rb') as file:
            # Create a PDF reader object
            reader = PyPDF2.PdfReader(file)

            # Add the pages of the PDF file to the merger
            merger.append(reader)

# Create a new PDF file in write-binary mode
output_filename = 'mergedfile.pdf'
with open(output_filename, 'wb') as output_file:
    # Write the merged PDF to the output file
    merger.write(output_file)

print(f'PDF files merged successfully into {output_filename}')
