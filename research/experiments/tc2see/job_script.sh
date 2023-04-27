salloc \
--nodes=1 \
--time=0-3:00:0 \
--ntasks-per-node=1 \
--cpus-per-task=3 \
--mem=48G \
--account=def-uofavis-ab

module load python/3.9
source ~/ENVS/glmsingle/bin/activate
$VIRTUAL_ENV/bin/notebook.sh

ssh -L 8888:cdr861.int.cedar.computecanada.ca:8889 efirdc@cedar.computecanada.ca

http://cdr861.int.cedar.computecanada.ca:8889/?token=d6140a6af89acc7f16e885a2b195610e1b205dd119dc3c22