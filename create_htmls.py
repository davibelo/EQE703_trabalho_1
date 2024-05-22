import nbconvert
import nbformat

FILENAME = 'res_questao_2'

# Load the notebook
with open(f'{FILENAME}.ipynb') as f:
    notebook_content = nbformat.read(f, as_version=4)

# Convert the notebook to HTML
html_exporter = nbconvert.HTMLExporter()
body, resources = html_exporter.from_notebook_node(notebook_content)

# Save the HTML to a file
with open(f'{FILENAME}.html', 'w') as f:
    f.write(body)
