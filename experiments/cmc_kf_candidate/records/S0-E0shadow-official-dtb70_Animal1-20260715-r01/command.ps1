$python = 'D:\Anaconda\envs\track\python.exe'
$repo = 'D:\PyCharm\Projects\OSTrack-main-cmc-kf-candidate'
Set-Location $repo
& $python -u tracking\test.py `
  ostrack_cmc_kf_assoc e0__official_vitb256_ce_ep300__v1 `
  --dataset_name dtb70 --sequence Animal1 --threads 0 --num_gpus 1
