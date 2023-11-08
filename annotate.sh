#!bin/bash

annotations=()

readarray -d '' fileNameArray < <(find ~+ -name "*.wav" -print0)

for f in ${!fileNameArray[@]}; do
        s=${fileNameArray[$f]}
        char=${s: -5: 1}
	duration=$(ffprobe -i $s -show_entries format=duration -v quiet -of csv="p=0")
        case $char in

                a)
                        annotations+=("$s,0,$duration")
                        ;;
                e)
                        annotations+=("$s,1,$duration")
                        ;;
                i)
                        annotations+=("$s,2,$duration")
                        ;;
                o)
                        annotations+=("$s,3,$duration")
                        ;;
                u)
                        annotations+=("$s,4,$duration")
                        ;;
        esac

done

printf "%s\n" "${annotations[@]}" > annotations.txt

exit 0
