#!/bin/bash

concat_cpp_project() {
    project_dir="$1"
    output_file="$2"

    # Find all .h and .cpp files in the project directory
    cpp_files=$(find "$project_dir" -name "*.cpp")
    header_files=$(find "$project_dir" -name "*.h")

    # Write main.cpp to the output file
    main_file="$project_dir/main.cpp"
    if [ -f "$main_file" ]; then
        cat "$main_file" >"$output_file"
    else
        echo "main.cpp not found in the project directory."
        return 1
    fi

    # Write header files to the output file
    for header in $header_files; do
        echo "" >>"$output_file"
        echo "// Header: $header" >>"$output_file"
        cat "$header" >>"$output_file"
    done

    # Write remaining .cpp files to the output file
    for cpp_file in $cpp_files; do
        if [ "$cpp_file" != "$main_file" ]; then
            echo "" >>"$output_file"
            echo "// Source: $cpp_file" >>"$output_file"
            cat "$cpp_file" >>"$output_file"
        fi
    done

    echo "C++ project concatenated into $output_file"
}

if [ $# -eq 0 ]; then
    echo "Please provide the path to the directory containing the C++ project."
    return 1
fi

if [ $# -eq 1 ]; then
    echo "Please provide filename to save to."
    return 1
fi

concat_cpp_project "$1" "$2"
