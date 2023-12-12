import os
import re
from graphviz import Digraph


graph_name = "includes"

# Function to extract included files from C/C++ source files
def extract_includes(file_path):
    includes = []
    with open(file_path, 'r') as file:
        for line in file:
            match = re.match(r'^\s*#\s*include\s*[<"]([^">]+)[>"]', line)
            if match:
                includes.append(match.group(1))
    return includes

# Function to recursively find included files starting from main.cpp and create the graph
def create_include_graph(start_file):
    graph = Digraph(graph_name, comment = "The project structure", format='png')
    graph.attr(rankdir='BT')  # Set direction of the graph (left to right)

    # Store unique files to avoid duplicate nodes in the graph
    global unique_files
    global current_id

    unique_files = {}
    current_id = 0

    # Function to recursively find included files
    def find_includes(file_path):
        global current_id

        if not os.path.exists(file_path):
            return

        if file_path in unique_files:
            return unique_files[file_path]

        unique_files[file_path] = str(current_id)
        current_id += 1

        graph.node(unique_files[file_path], os.path.basename(file_path))  # Use basename for node label

        included_files = extract_includes(file_path)
        current_dir = os.path.dirname(file_path)

        for included_file in included_files:
            included_file_path = os.path.join(current_dir, included_file)

            # Check if the file exists and is not already added to the graph
            if os.path.exists(included_file_path):
                included_file_node = find_includes(included_file_path)
                graph.edge(unique_files[file_path], included_file_node)

        return unique_files[file_path]

    find_includes(start_file)
    return graph

# Replace 'your_directory_path/main.cpp' with the path to your main.cpp file
main_file_path = './src/main.cpp'
include_graph = create_include_graph(main_file_path)

# Save the graph to a file
# include_graph.render(format="png", cleanup=True) # do not save the png file
include_graph.view(cleanup=True)