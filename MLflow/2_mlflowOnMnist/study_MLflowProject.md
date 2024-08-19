Reference:  
https://dailyheumsi.tistory.com/263    
https://pupbani.tistory.com/233

## MLflow Projects
mmdet에서 config로 하이퍼 파라미터 관리하는 것과 유사하다.  

```yaml
name: tutorial

conda_env: conda.yaml

entry_points:
  main:
    parameters:
      alpha: {type: float, default: 0.5}
      l1_ratio: {type: float, default: 0.1}
    command: "python train.py {alpha} {l1_ratio}"
```

```
$ mlflow run -e main sklearn_elastic_wine -P alpha=0.1 -P l1_ratio=0.5
$ python train.py 0.1 0.5

# github에 올려둔 MLProject을 실행시키는 것도 가능하다.
$ mlflow run git@github.com:mlflow/mlflow-example.git -P alpha=0.5 --no-conda
```
