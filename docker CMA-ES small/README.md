`docker build -t nevergrad_1 .
docker run -it --name Nevergrad_R nevergrad_1
chmod +x target-runner-linux
Rscript r-runner.R`

python target-runner.py 4 8 1101047833 yasmallbbob --name=hm --rotation=True --d=50 --budget=20   --solver=CMA-ES --scale=5.8196 --popsize_factor=5 --elitist=True --diagonal=False --fcmaes=True --random_init=False