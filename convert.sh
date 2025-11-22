#!/bin/bash

# One file to keep the papers that I have already ingested
# One dir to store newly added papers
# A temporary dir for image-based pdfs.
output_dir="./data/new"
temp_dir="./data/temp"

counter=0

total=$(find ./papers -type f -name "*.pdf" | wc -l)

find ./papers -type f -name "*.pdf" | while read -r file
do
    base_name=$(basename "$file" .pdf)

    pdftotext -enc UTF-8 "$file" "$output_dir/$base_name.txt"
	
    counter=$((counter + 1))
    echo -ne "Processed $counter out of $total PDFs.\r"
    
done
