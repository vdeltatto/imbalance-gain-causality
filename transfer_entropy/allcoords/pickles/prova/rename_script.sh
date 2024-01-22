for file in *; do    
    if [[ $file != rename_script.sh ]]; then
        mv "$file" "${file/.dat/_new.dat}"
        #echo "Renamed $file to ${file/_dat/_new.dat}"
    fi
done