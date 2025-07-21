# Classificador de manobras de Surf

Esse projeto é um classificador de vídeo, utilizando um modelo de 3D convolutional neural network(CNN) feito no tensorflow, ele consegue classificar a manobra feita num vídeo de surf.

## tabela de conteúdo

- [Sobre o projeto](#Sobre_o_Projeto)
  -[Arquitetura da Rede](#Arquitetura_da_Rede)
  -[Treinamento](#Treinamento)
    -[Dataset](#Dataset)
    -[Resultados e Metricas](#Resultados_e_Metricas)
- [Como Baixar](#Como_Baixar)
- [Proximos Passos](#Proximos_Passos)
- [Como contribuir](#como_contribuir)
- [Licensa](#licensa)

## Sobre o Projeto

Falando mais sobre a escolha do projeto, ele se baseou em um [tutorial do tensorflow de classificação com videos](https://www.tensorflow.org/tutorials/video/video_classification), mas utilizando como base um novo dataset e novas classes, treinando a rede do zero.  
[foto que representa o projeto]

### Arquitetura da rede
Nesse tutorial eles utilizam uma camada de Convolução 3D, que na realidade é (2+1)D, técnica detalhada no artigo [A Closer Look at Spatiotemporal Convolutions for Action Recognition](https://arxiv.org/abs/1711.11248v3). Como estamos lidando com vídeos, essa primeira camada recebe entradas com o formato (tempo, altura, largura, canais). Decompondo o 3D em 2+1 lideamos com o espaço e o tempo de formas separadas. Graficamente podemos vizualizar da seguinte forma:
![Representação da convolução](https://www.tensorflow.org/images/tutorials/video/2plus1CNN.png "Convolução 2+1")
A vantagem disso é a facilidade dos calculos, pois reduz o numero de parâmetros por alterar a dimensão da matriz de pesos

Mais adiante ainda temos uma sequencia de blocos residuais, por ser uma rede recorrente, e a arquitetura da rede ficou assim
![classificador keras](https://github.com/user-attachments/assets/e2c48cf2-2ae6-42df-850a-f6c5a3b1fc83)

### Treinamento 
Foram feitos vários treinamentos, variando: epocas, versão do dataset, preprocessamento, learning rates, etc. Todos feitos usando google colab ou kaggle notebook
#### Dataset
Essa foi sem dúvida a parte mais trabalhosa do projeto, criar um dataset do zero não é facil! Utilizamos recortes de +/- 3 segundos (tempo médio de uma manobra), a maioria de surfistas profissionais perfomando nas competições mais importantes da cena do surf mundial. Para realizar esses cortes, a ferramenta ffmpeg foi perfeita, tanto para ajustar o formato ideal do vídeo, tanto para o cortes.

Exemplo de como recortar um trecho de um video.webm e deixar no formato .avi(utilizado no dataset)
```bash
ffmpeg -i wsl_miami_p1.webm -ss 3:14 -to 3:18 -c:v mpeg4 -q:v 2 -vtag xvid -c:a libmp3lame -q:a 2 rasgada_93.avi
```
No fim das contas o dataset tem 2 classes apenas: Aéreos e Rasgadas. Duas manobras distintas entre si e muito comuns na elite do surf mundial
![rasgada_37](https://github.com/user-attachments/assets/6b796612-0fec-40f3-8b78-bf75babf3eff)
![aerial_89](https://github.com/user-attachments/assets/b612bc72-b109-423d-aabc-ba9dcc0bf252)

e para aumentar o dataset foram usadas algumas recursos como: duplicar e inverter os videos
![rasgada_116](https://github.com/user-attachments/assets/d896e502-bbe8-44d5-b51e-e80c1f99280e)
![rasgada_116_hflip](https://github.com/user-attachments/assets/158604ed-d6c6-49d5-93bd-872f3f7c697a)

e além disso, os videos passaram por um preprocessamento para segmentar (recortar so o surfista) os videos, feito com [yolov8](https://docs.ultralytics.com/models/yolov8/),já visando uma melhoria futura de adicionar os keypoints do surfista e tembém por ter muita informação na imagem (mar balaçando, outros possiveis surfistas etc.)

exemplo de video segmentado
![cropped_01](https://github.com/user-attachments/assets/08117358-8627-4635-be1d-ff24ab5a6b9d)

#### Resultados e Metricas
Depois de varios testes e treinamentos, foi obtido um resultado satisfatório, nas métricas escolhidas. Para o treinamento foi analisado ACURACCY e LOSS, do treinamento e da validação
<img width="1364" height="772" alt="Screenshot from 2025-07-21 14-31-47" src="https://github.com/user-attachments/assets/507238e8-0659-4650-ab16-b68c25bbb843" />
Já na matriz de confusão esses foram os resultados
![matriz_conf_test](https://github.com/user-attachments/assets/5e84acff-b658-47d7-bee1-aaaa983dfe18)
Foram bastantes satisfatórios, mas alguns aéreos foram confundidos com rasgadas, talvez uns com pouca rotação e sem muita altura. 




