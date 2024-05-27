import os
import shutil
import nbformat
from nbconvert import HTMLExporter

# Get a list of all .ipynb files in the current directory
notebook_files = [f for f in os.listdir('.') if f.endswith('.ipynb')]

# List to keep track of generated HTML files
html_files = []

# Process each notebook file
for notebook_file in notebook_files:
    with open(notebook_file, 'r', encoding='utf-8') as f:  # Set encoding to UTF-8
        notebook_content = nbformat.read(f, as_version=4)

    # Convert the notebook to HTML
    html_exporter = HTMLExporter()
    html_exporter.exclude_output_prompt = True
    html_exporter.exclude_input_prompt = True
    body, resources = html_exporter.from_notebook_node(notebook_content)

    # Save the HTML to a file with the same name but .html extension
    html_filename = f"{os.path.splitext(notebook_file)[0]}.html"
    with open(html_filename, 'w', encoding='utf-8') as f:  # Also set encoding here for output
        f.write(body)
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
