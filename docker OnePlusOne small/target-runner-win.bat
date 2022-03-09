@echo off
call conda activate C:\Users\Risto\.conda\envs\newEnv
set parameters=%*
python target-runner.py  %parameters%
