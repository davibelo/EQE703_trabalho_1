import nbconvert
import nbformat

# Load the notebook
with open('res_questao_2.ipynb') as f:
    notebook_content = nbformat.read(f, as_version=4)

# Convert the notebook to PDF
pdf_exporter = nbconvert.PDFExporter()
body, resources = pdf_exporter.from_notebook_node(notebook_content)

# Save the PDF to a file
with open('res_questao_2.pdf', 'wb') as f:
    f.write(body)
