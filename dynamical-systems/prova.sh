start_eps=0.0
end_eps=3.0
n_eps=16
stride=$(bc <<< "scale=20; (${end_eps}-${start_eps})/(${n_eps}-1)")

i=${1}
eps=$(bc <<< "scale=10; ${start_eps} + ${i} * ${stride}")
echo ${eps}