for file in *; do    
    if [[ $file != rename_script.sh ]]; then
        mv "$file" "${file/.p/_k3_minzeroTrue.p}"
        #echo "Renamed $file to ${file/_dat/_new.dat}"
    fi
done