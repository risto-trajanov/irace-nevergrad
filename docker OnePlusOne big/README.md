`docker build -t nevergrad_1 .
docker run -it --name Nevergrad_R nevergrad_1
chmod +x target-runner-linux
Rscript r-runner.R`