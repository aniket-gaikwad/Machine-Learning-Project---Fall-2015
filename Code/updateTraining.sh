cat training.csv | 
	while read -r line; do 
		str=""
		for i in {1..34}; do 
			field=$(grep "^$i " allFields.txt | cut -d' ' -f2)
			fieldval=$(echo $line | cut -d',' -f$i)
			if [ "$field" == "WheelType" ]; then 
				continue
			fi
			if [ -f "fields/${field}.txt" ] && [ ! -z "$fieldval" ] && [ "$field" != "$fieldval" ]; then 
				newFieldval=$(grep " ${fieldval}$" fields/${field}.txt | cut -d' ' -f1)
				str+=,$newFieldval
			else
				if [ "$fieldval" == "PurchDate" ]; then 
					fieldval="PurchMonth,PurchDay,PurchYear"
				elif [ ! -z "$(echo $fieldval | grep "/")" ]; then 
					fieldval=$(echo $fieldval | cut -d'/' -f1),$(echo $fieldval | cut -d'/' -f2),$(echo $fieldval | cut -d'/' -f3)
				fi
				str+=,$fieldval
			fi
		done
		echo ${str:1:${#str}}
	done
