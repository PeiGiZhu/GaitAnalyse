## Folder structure

- library dependencies files
  - lib
  - ProcessVideoLib
  - FeatureAnalyseLib
  - fitness_poses_csvs_out
  - Roboto-Regular.ttf

- input/output folder
  - data_record
  - video_input
  - video_output
  - PCAmodel
- Top code
  - VideoAnalyse.py
  - FeatureExtract.py
  - PCAModelGenerator.py

## Command example

- Firstly, get joint angle from video, which is corresponding to 'Human pose detection' and 'Get joint angle signal' in the overview.pdf. This step will generate a rendered video in ***video_output*** folder and record some pre-processed data in ***data_record*** folder

```python
python .\VideoAnalyse.py -n video_fede.mp4 -m -pn 10 -mn 10.0 -pf 10 -mf 10.0
```

1. -n is the sample name in the ***video_input*** folder
2. Add -m to apply segmentation mask.
3. -mn -pn -mf -pf Kalman filter related parameters.



- Secondly,  use FFT to filters gain signal and extracts key features, which is corresponding to 'Get filtered signal' and 'Extract gait features'. This step will generate filtered signal figures and extracted feature csv file in the  ***data_record*** folder

```
python .\FeatureExtract.py -n video_fede.mp4 -ns right -c 0.15 -p 5
```

1. -n is the file name in the ***data_record*** folder
2. Set -c to cut slice from prefix of signal (0.15 is 15% prefix time of the video) corresponding to the red dash line of filtered signal figures
3. -ns is the side of the body that close to the camera in the target video
4. -p the frequency components with top 'p' amplitude will be selected to recover filtered signal in FFT.
5. -d (not be shown in example) only need when the extracted features are needed to build PCA model. 



- Thirdly, generate PCA Model. Only be used when the data_record has sufficient sample. It need a csv file to build PCA model (An example file 'all_features.csv' is provided)

```
python .\PCAModelGenerator.py
```



- Finally, analyse patients recovery level. The hotelling'T2 value reflect the distance between target and normal samples from 'all_features.csv'.

```
python .\FeatureExtract.py -n video_fede.mp4 -ns right -t2 -a age -he height -we weight -s sex
```

1. -n patients sample must be loaded to ***data_record*** folder.
2. -t2 add this symbol to calculate hotelling'T2 value for given video.
3.  -a -he -we -s are patient's basic information.