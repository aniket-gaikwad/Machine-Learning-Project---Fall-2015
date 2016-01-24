TARGET=$1
cat nonNumericFields.txt | 
	while read -r line; do 
		field=$(echo $line | cut -d' ' -f1)
		fieldid=$(echo $line | cut -d' ' -f2)
		count=1
		cut -d',' -f$fieldid training.csv | sort -u | 
			sed "/${field}\|NULL/d" | 
				while read -r fieldVal; do 
					echo $count" "$fieldVal; count=$(($count+1))
				done 
		> ${TARGET}/${field}.txt
	done
