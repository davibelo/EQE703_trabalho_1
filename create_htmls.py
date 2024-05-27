import os
import shutil
import nbformat
from nbconvert import HTMLExporter
import re

# Get a list of all .ipynb files in the current directory
notebook_files = [f for f in os.listdir('.') if f.endswith('.ipynb')]

# List to keep track of generated HTML files
html_files = []

def copy_images(cell_source, base_path, dest_dir):
    """Copy images referenced in Markdown and HTML cells to the destination directory."""
    # Patterns to match Markdown and HTML image references
    markdown_image_pattern = re.compile(r'!\[.*?\]\((.*?)\)')
    html_image_pattern = re.compile(r'<img\s+.*?src="(.*?)".*?>')

    # Find and copy Markdown images
    for match in markdown_image_pattern.findall(cell_source):
        image_path = os.path.join(base_path, match)
        if os.path.exists(image_path):
            dest_path = os.path.join(dest_dir, match)
            dest_dir_path = os.path.dirname(dest_path)
            if not os.path.exists(dest_dir_path):
                os.makedirs(dest_dir_path)
            shutil.copy2(image_path, dest_path)

    # Find and copy HTML images
    for match in html_image_pattern.findall(cell_source):
        image_path = os.path.join(base_path, match.replace('\\', '/'))
        if os.path.exists(image_path):
            dest_path = os.path.join(dest_dir, match.replace('\\', '/'))
            dest_dir_path = os.path.dirname(dest_path)
            if not os.path.exists(dest_dir_path):
                os.makedirs(dest_dir_path)
            shutil.copy2(image_path, dest_path)

# Process each notebook file
for notebook_file in notebook_files:
    with open(notebook_file, 'r', encoding='utf-8') as f:
        notebook_content = nbformat.read(f, as_version=4)

    base_path = os.path.dirname(notebook_file)
    dest_dir = os.path.join('htmls', base_path)

    # Create destination directory if it doesn't exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Copy images referenced in Markdown and HTML cells
    for cell in notebook_content['cells']:
        if cell['cell_type'] == 'markdown':
            copy_images(cell['source'], base_path, dest_dir)
        elif cell['cell_type'] == 'code':
            if 'outputs' in cell:
                for output in cell['outputs']:
                    if 'text' in output and output['text']:
                        copy_images(output['text'], base_path, dest_dir)

    # Convert the notebook to HTML
    html_exporter = HTMLExporter()
    html_exporter.exclude_output_prompt = True
    html_exporter.exclude_input_prompt = True

    body, resources = html_exporter.from_notebook_node(notebook_content)

    # Save the HTML to a file with the same name but .html extension
    html_filename = f"{os.path.splitext(notebook_file)[0]}.html"
    with open(html_filename, 'w', encoding='utf-8') as f:
        f.write(body)
    html_files.append(html_filename)

# Create the htmls directory if it doesn't exist
if not os.path.exists('htmls'):
    os.makedirs('htmls')

# Move all generated HTML files to the htmls directory
for html_file in html_files:
    shutil.move(html_file, os.path.join('htmls', html_file))
