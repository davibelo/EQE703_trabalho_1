import os
import shutil
import nbformat
from nbconvert import HTMLExporter
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter

# Function to convert .py file to HTML
def convert_py_to_html(py_file):
    with open(py_file, 'r') as f:
        code = f.read()
    
    # Use Pygments to highlight the Python code
    formatter = HtmlFormatter(full=True, linenos=True)
    highlighted_code = highlight(code, PythonLexer(), formatter)
    
    html_filename = f"{os.path.splitext(py_file)[0]}.html"
    with open(html_filename, 'w') as f:
        f.write(highlighted_code)
    return html_filename

# Get a list of all .ipynb files in the current directory
notebook_files = [f for f in os.listdir('.') if f.endswith('.ipynb')]

# Get a list of all .py files in the current directory and the src folder
python_files = [f for f in os.listdir('.') if f.endswith('.py')]
python_files += [os.path.join('src', f) for f in os.listdir('src') if f.endswith('.py')]

# List to keep track of generated HTML files
html_files = []

# Process each notebook file
for notebook_file in notebook_files:
    with open(notebook_file) as f:
        notebook_content = nbformat.read(f, as_version=4)

    # Convert the notebook to HTML
    html_exporter = HTMLExporter()
    body, resources = html_exporter.from_notebook_node(notebook_content)

    # Save the HTML to a file with the same name but .html extension
    html_filename = f"{os.path.splitext(notebook_file)[0]}.html"
    with open(html_filename, 'w') as f:
        f.write(body)
    html_files.append(html_filename)

# Process each Python file
for python_file in python_files:
    html_filename = convert_py_to_html(python_file)
    html_files.append(html_filename)

# Create the htmls directory if it doesn't exist
if not os.path.exists('htmls'):
    os.makedirs('htmls')

# Move all generated HTML files to the htmls directory
for html_file in html_files:
    # Ensure the destination directory exists
    dest_dir = os.path.join('htmls', os.path.dirname(html_file))
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    shutil.move(html_file, os.path.join('htmls', html_file))
