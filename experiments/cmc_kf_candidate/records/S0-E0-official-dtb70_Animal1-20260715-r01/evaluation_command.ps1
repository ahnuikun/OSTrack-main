$python = 'D:\Anaconda\envs\track\python.exe'
$repo = 'D:\PyCharm\Projects\OSTrack-main-cmc-kf-candidate'

Set-Location $repo
& $python -u tracking\analyze_uav_suite.py `
  --tracker_name ostrack `
  --tracker_param e0__official_vitb256_ce_ep300__v1 `
  --dataset dtb70 `
  --force_evaluation `
  --skip_missing_seq `
  --per_sequence
