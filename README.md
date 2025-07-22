# Classificador de manobras de Surf

Esse projeto é um classificador de vídeo, utilizando um modelo de 3D convolutional neural network(CNN) feito no tensorflow, ele consegue classificar a manobra feita num vídeo de surf.

## tabela de conteúdo

- [Sobre o projeto](##Sobre-o-Projeto)
- [Arquitetura da Rede](##Arquitetura-de-Rede)
- [Treinamento](##Treinamento)
 - [Dataset](#Dataset)
 - [Resultados e Metricas](#Resultados-e-Metricas)
- [Como Baixar](#Como-Baixar)
- [Proximos Passos](#Proximos-Passos)
- [Como contribuir](#como-contribuir)
- [Licensa](#licensa)

## Sobre o Projeto

Falando mais sobre a escolha do projeto, ele se baseou em um [tutorial do tensorflow de classificação com videos](https://www.tensorflow.org/tutorials/video/video_classification), mas utilizando como base um novo dataset e novas classes, treinando a rede do zero. 

Foi decidido o nicho do esporte, visto que a analise de vídeo hoje está muito presente no meio esportivo, sendo muito importante para atletas de alto nível analisarem o próprio desempenho e evoluirem com a ajuda de ferramentas. E especificamente o surf é uma modalidade de muito plástica de movimentos distintos entre si, facilitanto por um lado a classificação.

Além disso o projeto é escalável e pretendemos fornecer análises cada vez mais detalhadas dos vídeos.

[foto que representa o projeto]

## Arquitetura da rede
Nesse tutorial é utilizada uma camada de Convolução 3D, que na realidade é (2+1)D, técnica detalhada no artigo [A Closer Look at Spatiotemporal Convolutions for Action Recognition](https://arxiv.org/abs/1711.11248v3) de D. Tran et al. (2017). Como estamos lidando com vídeos, essa primeira camada recebe entradas com tempo * altura * largura * canais), no caso dos videos do dataset recebe arrays com (numero de frames, altura do vídeo, largura do vídeo, red, green, blue).

Decompondo o 3D em 2+1 lideamos com o espaço e o tempo de formas separadas. Graficamente podemos vizualizar da seguinte forma:

![Representação da convolução](https://www.tensorflow.org/images/tutorials/video/2plus1CNN.png "Convolução 2+1")

A vantagem disso, segundo o artigo, é a facilidade dos calculos, pois reduz a quantidade de parâmetros por alterar a dimensão da matriz de pesos de (27 * canais ** 2) para (9 * canais ** 2) + (3 * canais ** 2)

O núcleo do modelo é construído em torno de blocos residuais, implementados pela função add_residual_block. Cada bloco utiliza uma camada principal ResidualMain para realizar convoluções e normalizações, mas sua característica essencial é a conexão residual. Essa conexão soma a entrada do bloco à sua saída, permitindo que o gradiente flua mais facilmente durante o treinamento e ajudando a rede a aprender características mais complexas. Para garantir que essa soma seja possível mesmo quando o número de filtros aumenta entre os blocos, a camada customizada Project é usada para ajustar as dimensões da conexão residual. A arquitetura processa o vídeo de forma progressiva: após a maioria dos blocos residuais, a camada ResizeVideo reduz a altura e a largura dos frames, diminuindo o custo computacional enquanto os blocos aumentam a profundidade das características (o número de filtros). Por fim, após a extração de características, uma camada GlobalAveragePooling3D condensa as informações espaciais e temporais em um único vetor, que é então achatado (Flatten) e passado para a camada Dense final, responsável por gerar a classificação.

<img src="[path/to/your/image.svg](https://github.com/user-attachments/assets/e2c48cf2-2ae6-42df-850a-f6c5a3b1fc83)" alt="Description of SVG" style="width:[200]; height:auto;">

## Treinamento 
O processo de treinamento envolveu múltiplas iterações, nas quais ajustamos diversos parâmetros como o número de épocas, a versão do dataset utilizada, as técnicas de pré-processamento e as taxas de aprendizado. Todos esses experimentos foram executados em ambientes de notebook como Google Colab e Kaggle Notebooks. 

### Dataset

Essa foi sem dúvida a parte mais trabalhosa do projeto, criar um dataset do zero não é facil! Utilizamos recortes de +/- 3 segundos (tempo médio de uma manobra), a maioria de surfistas profissionais perfomando nas competições mais importantes da cena do surf mundial. Foi utilizada a ferramenta [ffmpeg](https://ffmpeg.org/), tanto para ajustar o formato ideal do vídeo, tanto para o cortes. Existe também uma GUI feita com base em ffmpeg que ajudou bastante chamada [lossless cut](https://github.com/mifi/lossless-cut), ela mantém o formato do vídeo e ajuda a ter uma precisão maior no corte.

Exemplo de como recortar um trecho de um video.webm e deixar no formato .avi(utilizado no dataset)

```bash
ffmpeg -i wsl_miami_p1.webm -ss 3:14 -to 3:18 -c:v mpeg4 -q:v 2 -vtag xvid -c:a libmp3lame -q:a 2 rasgada_93.avi
```

Em sua configuração final, o dataset trabalha com duas classes: Aéreos e Rasgadas. O aéreo, como o nome ja diz, é uma manobra em na qual o surfista decola, saindo da onda, ele tem diversas variações de rotação, pegada na prancha(grab), etc. O que pode dificultar a identificação. Já a rasgada é uma manobra mais simples, o surfista ataca o lip (parte de cima da onda) jogando a rabeta da prancha e levantando bastante água. 
São duas manobras bem distintas entre si e bastante utilizadas na elite do surf mundial.

![rasgada_37](https://github.com/user-attachments/assets/6b796612-0fec-40f3-8b78-bf75babf3eff "rasgada")
![aerial_89](https://github.com/user-attachments/assets/b612bc72-b109-423d-aabc-ba9dcc0bf252 "aéreo")

Para aumentar o tamanho do dataset, um dos recursos utilizados foi o de duplicar um vídeo e inverter ele. Essa solução bem simples duplicou o número de vídeos no dataset

![rasgada_116](https://github.com/user-attachments/assets/d896e502-bbe8-44d5-b51e-e80c1f99280e)
![rasgada_116_hflip](https://github.com/user-attachments/assets/158604ed-d6c6-49d5-93bd-872f3f7c697a)

Além disso, os videos passaram por um preprocessamento para segmentar (recortar so o surfista) os videos, feito com um modelo já pronto [yolov8](https://docs.ultralytics.com/models/yolov8/). A ideia por trás disso foi reduzir a quantidade informações no video, visto que eles poderiam ter: mar balaçando, espuma da onda, outros possiveis surfistas etc. 

exemplo de video segmentado

![cropped_01](https://github.com/user-attachments/assets/08117358-8627-4635-be1d-ff24ab5a6b9d)

## Resultados e Metricas
Após diversas rodadas de testes e treinamentos, alcançamos resultados satisfatórios nas métricas escolhidas. Durante o processo, acompanhamos de perto a acurácia e a loss tanto do conjunto de treinamento quanto do conjunto de validação, culminando em um desempenho final robusto.

<img width="1364" height="772" alt="Screenshot from 2025-07-21 14-31-47" src="https://github.com/user-attachments/assets/507238e8-0659-4650-ab16-b68c25bbb843" />

A matriz de confusão resultante dos videos de teste foi satisfatória e houve uma mudança significativa em relação a matriz gerada com os arquivos de treino. Entretanto alguns aéreos estão sendo confundidos com rasgadas, isso está acontecendo, provavelmente, em alguns aéreos com pouca rotação, ou em que o surfista não saiu muito da água. Esse erro talvez fosse resolvido adicionando os kepoints como entrada no modelo.

![matriz_conf_train](https://github.com/user-attachments/assets/4f095c7e-f40d-40ee-b992-3c0799386a1d "Matriz de Confusão Treino")

![matriz_conf_test](https://github.com/user-attachments/assets/5e84acff-b658-47d7-bee1-aaaa983dfe18 "Matriz de Confusão Teste")






