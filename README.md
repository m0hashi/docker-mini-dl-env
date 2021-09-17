# 深層学習用イメージ
主な深層学習ライブラリ
- rapids-blazing==21.8
- tensorflow==2.6.0
- pytorch==1.9.0

## Build 
```
docker-compose build
```

## RUN
jupyter-labが起動します。
```
docker-compose up
```

## Available Services/Ports
- juptyer:8879
- tensorboard: 6006
- mlflow: 500

## Shared volumes
- ~/workspace:/home/mluser/workspace
